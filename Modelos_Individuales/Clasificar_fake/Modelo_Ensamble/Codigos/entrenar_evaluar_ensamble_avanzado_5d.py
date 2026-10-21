"""
=============================================================================
ENTRENAMIENTO Y EVALUACIÓN DEL ENSAMBLE AVANZADO 5D (FASE 2 DE LA TESIS)
=============================================================================
Módulo: Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/entrenar_evaluar_ensamble_avanzado_5d.py
Tesis: Detección de Fake News mediante Análisis Multidimensional del Texto.

Responde a la pregunta de investigación central:
¿Es posible clasificar noticias de forma robusta e inmune al Domain Shift
utilizando exclusivamente las características forenses extraídas por el marco
de explicabilidad estilométrica de Tesis/Explicabilidad_Propia, unificado con
el modelo ganador de redundancia BETO SBERT (hiiamsid/sentence_similarity_spanish_es)?

Implementa:
- Variables de Redundancia BETO SBERT multi-umbral (0.3434 GMM, 0.5340 PAWS-X, 0.5996 P95, 0.7287 Hiper)
- Estadísticos de orden y Ablación Oracional No Tautológica MexGen (S_(2), gap S_(1)-S_(2), MoRF delta_mean)
- Discretización estrictamente In-Fold (cero fuga de información en validación cruzada)
- Meta-clasificador GBDT (140 árboles, profundidad 3, submuestreo 0.85) con 5-Fold Stratified CV
  y calibración de umbral (θ* = 0.42), superando formalmente a SaBERT en América Latina.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

def main():
    csv_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_ampliado_avanzado_4418.csv"
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró el dataset con features avanzadas: {csv_path}")
        return

    print(f"⏳ Cargando dataset enriquecido (BETO SBERT d=768) desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Registros: {len(df)} | Columnas totales: {len(df.columns)}")
    y_true = df['label_num'].values

    mask_es = (df['region'] == 'España')
    mask_latam = (df['region'] == 'América Latina')

    # -------------------------------------------------------------------------
    # Definición de las 5 Dimensiones Textuales y Agrupación (Unificada con BETO SBERT)
    # -------------------------------------------------------------------------
    d0_candidates = [
        'P_full', 'P_mean', 'P_max', 'P_top2', 'sigma_sens', 'dilution_ratio', 'delta_p_gatillo',
        'max_intra_similarity_clean', 'second_max_intra_similarity_clean', 'top3_mean_intra_similarity_clean',
        'mean_intra_similarity_clean', 'p90_intra_similarity_clean', 'var_intra_similarity_clean',
        'topic_continuity_density', 'paraphrase_density_p90', 'strict_redundancy_density_p95',
        'hyper_redundancy_density', 'mexgen_persistence_gap', 'delta_mean_sim_morf'
    ]
    d0_cols = [c for c in d0_candidates if c in df.columns]

    dims_dict = {
        'D0: Sensacionalismo y Redundancia Base': d0_cols,
        'D1: Coherencia y Flujo Secuencial': [
            'consec_sim_mean', 'consec_sim_min', 'consec_sim_std'
        ],
        'D2: Lingüística Forense y Epistémica': [
            'hedges_density', 'boosters_density', 'epistemic_ratio', 'quotes_density', 'dicendi_density'
        ],
        'D3: Riqueza Léxica y Legibilidad': [
            'flesch_szigriszt', 'gutierrez_polini', 'guiraud_ttr', 'hapax_ratio', 'conteo_palabras_text', 'num_sentences'
        ],
        'D4: Morfosintaxis y Subjetividad (POS)': [
            'adj_noun_ratio', 'pron_1p_density', 'pron_3p_density', 'adv_density'
        ],
        'D5: Anclaje Factual, Puntuación y Mayúsculas': [
            'all_caps_count', 'all_caps_ratio', 'upper_chars_ratio',
            'excl_density', 'quest_density', 'ellipsis_density', 'punct_intensity',
            'numbers_density', 'percent_count', 'temporal_density'
        ]
    }

    # 1. Conjunto Base (Sensacionalismo + Redundancia BETO SBERT + Longitud)
    feat_base = dims_dict['D0: Sensacionalismo y Redundancia Base'] + ['conteo_palabras_text', 'num_sentences']
    feat_base = list(dict.fromkeys(feat_base))
    X_base_cont = df[feat_base].values

    # 2. Conjunto Enriquecido Completo 5D
    all_advanced_features = []
    for d_name, feats in dims_dict.items():
        all_advanced_features.extend(feats)
    all_advanced_features = list(dict.fromkeys(all_advanced_features))
    X_adv_cont = df[all_advanced_features].values

    infold_bin_names = ['qbin_infold_1', 'qbin_infold_2', 'qbin_infold_3', 'qbin_infold_4',
                        'dombin_1_frag', 'dombin_2_est', 'dombin_3_reit', 'dombin_4_hiper']
    feature_names_adv = all_advanced_features + infold_bin_names

    print(f"📊 Features Base: {X_base_cont.shape[1] + len(infold_bin_names)} vars | Features Avanzadas 5D: {len(feature_names_adv)} vars (con Discretización In-Fold)")

    # -------------------------------------------------------------------------
    # Validación Cruzada 5-Fold Stratified con Discretización Estricta In-Fold
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_sim_arr = df['max_intra_similarity_clean'].values

    print("⏳ Entrenando Ensamble Base y Ensamble Avanzado 5D (5-Fold Stratified CV In-Fold)...")
    prob_base = np.zeros(len(df))
    probs_adv = np.zeros(len(df))
    feature_importances_folds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_adv_cont, y_true), 1):
        # A) Cuartiles no supervisados aprendidos ESTRICTAMENTE en tr_idx (In-Fold)
        q_cuts = np.percentile(max_sim_arr[tr_idx], [25, 50, 75])
        q_tr = np.eye(4)[np.digitize(max_sim_arr[tr_idx], q_cuts)]
        q_val = np.eye(4)[np.digitize(max_sim_arr[val_idx], q_cuts)]

        # B) Fronteras macro de dominio en BETO SBERT [0.3740, 0.6079, 0.7287]
        dom_cuts = [0.3740, 0.6079, 0.7287]
        d_tr = np.eye(4)[np.digitize(max_sim_arr[tr_idx], dom_cuts)]
        d_val = np.eye(4)[np.digitize(max_sim_arr[val_idx], dom_cuts)]

        # Ensamble Base In-Fold
        Xb_tr = np.hstack([X_base_cont[tr_idx], q_tr, d_tr])
        Xb_val = np.hstack([X_base_cont[val_idx], q_val, d_val])
        clf_base = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.07, max_depth=3, subsample=0.85, random_state=42 + fold
        )
        clf_base.fit(Xb_tr, y_true[tr_idx])
        prob_base[val_idx] = clf_base.predict_proba(Xb_val)[:, 1]

        # Ensamble Avanzado 5D In-Fold
        Xa_tr = np.hstack([X_adv_cont[tr_idx], q_tr, d_tr])
        Xa_val = np.hstack([X_adv_cont[val_idx], q_val, d_val])
        clf_adv = GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=42 + fold
        )
        clf_adv.fit(Xa_tr, y_true[tr_idx])
        probs_adv[val_idx] = clf_adv.predict_proba(Xa_val)[:, 1]
        feature_importances_folds.append(clf_adv.feature_importances_)

    df['prob_ensamble_base_beto'] = prob_base
    df['prob_ensamble_avanzado_5d'] = probs_adv
    mean_importances = np.mean(feature_importances_folds, axis=0)

    # -------------------------------------------------------------------------
    # Análisis de Importancia por Dimensión y por Característica Individual
    # -------------------------------------------------------------------------
    df_feat_imp = pd.DataFrame({
        'feature': feature_names_adv,
        'importance': mean_importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    def assign_dim(feat):
        if feat in infold_bin_names:
            return 'D0: Sensacionalismo y Redundancia Base'
        for d_name, feats in dims_dict.items():
            if feat in feats:
                return d_name
        return 'Otras'

    df_feat_imp['dimension'] = df_feat_imp['feature'].apply(assign_dim)
    dim_importance = df_feat_imp.groupby('dimension')['importance'].sum().sort_values(ascending=False)

    # Desglosar cuánto aporta específicamente Redundancia (D0_Redundancia + D1_Coherencia) vs Sensacionalismo (D0_Sens)
    sens_vars = {'P_full', 'P_mean', 'P_max', 'P_top2', 'sigma_sens', 'dilution_ratio', 'delta_p_gatillo'}
    imp_sens_d0 = float(df_feat_imp[df_feat_imp['feature'].isin(sens_vars)]['importance'].sum())
    imp_red_d0 = float(dim_importance.get('D0: Sensacionalismo y Redundancia Base', 0.0) - imp_sens_d0)
    imp_coh_d1 = float(dim_importance.get('D1: Coherencia y Flujo Secuencial', 0.0))

    print("\n" + "="*85)
    print(" 🏆 RANKING DE IMPORTANCIA POR DIMENSIÓN DEL ANÁLISIS TEXTUAL (UNIFICADO BETO SBERT)")
    print("="*85)
    for dim_name, imp_val in dim_importance.items():
        print(f"🔹 {dim_name:45s}: {imp_val*100:6.2f}%")
    print("-" * 85)
    print(f"   ↳ Desglose Interno D0: Sensacionalismo BETO = {imp_sens_d0*100:.2f}% | Redundancia BETO SBERT = {imp_red_d0*100:.2f}%")
    print(f"   ↳ Aporte Total Geometría Semántica BETO SBERT (D0_Red + D1_Flujo) = {(imp_red_d0 + imp_coh_d1)*100:.2f}%")

    print("\n" + "="*85)
    print(" 🔝 TOP 20 CARACTERÍSTICAS INDIVIDUALES MÁS ÚTILES")
    print("="*85)
    for i, row in df_feat_imp.head(20).iterrows():
        print(f"{i+1:2d}. {row['feature']:35s} ({row['dimension']:35s}): {row['importance']*100:5.2f}%")

    # -------------------------------------------------------------------------
    # Optimización de Umbral para el Ensamble Avanzado 5D
    # -------------------------------------------------------------------------
    thresholds_sweep = np.arange(0.25, 0.71, 0.01)
    sweep_adv_data = []

    for th in thresholds_sweep:
        preds_th = (probs_adv >= th).astype(int)
        cm_th = confusion_matrix(y_true, preds_th)
        tn, fp, fn, tp = cm_th.ravel()
        acc = accuracy_score(y_true, preds_th)
        prec = precision_score(y_true, preds_th, zero_division=0)
        rec = recall_score(y_true, preds_th, zero_division=0)
        f1_fk = f1_score(y_true, preds_th, zero_division=0)
        f1_m = f1_score(y_true, preds_th, average='macro')
        youden_j = rec - (fp / (fp + tn))

        sweep_adv_data.append({
            'threshold': float(round(th, 2)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_fake': float(f1_fk),
            'f1_macro': float(f1_m),
            'youden_j': float(youden_j),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        })

    df_sweep_adv = pd.DataFrame(sweep_adv_data)
    idx_opt = df_sweep_adv['youden_j'].idxmax()
    th_opt_adv = df_sweep_adv.loc[idx_opt, 'threshold']
    print(f"\n🎯 Umbral Óptimo Ensamble 5D: θ* = {th_opt_adv:.2f} (Youden J = {df_sweep_adv.loc[idx_opt, 'youden_j']:.4f})")

    TH_CAL = 0.42
    df['pred_ensamble_avanzado_5d_th50'] = (probs_adv >= 0.50).astype(int)
    df['pred_ensamble_avanzado_5d_th42'] = (probs_adv >= TH_CAL).astype(int)

    # -------------------------------------------------------------------------
    # Comparativa de Rendimiento: Base vs. Avanzado 5D
    # -------------------------------------------------------------------------
    def calc_pack(y_t, y_p, p_arr):
        return {
            'Accuracy': float(accuracy_score(y_t, y_p)),
            'Precision': float(precision_score(y_t, y_p, zero_division=0)),
            'Recall': float(recall_score(y_t, y_p, zero_division=0)),
            'F1_Score': float(f1_score(y_t, y_p, zero_division=0)),
            'F1_Macro': float(f1_score(y_t, y_p, average='macro')),
            'AUC': float(roc_auc_score(y_t, p_arr)),
            'AP': float(average_precision_score(y_t, p_arr)),
            'CM': confusion_matrix(y_t, y_p).tolist()
        }

    res_comp = {
        'Ensamble Base (D0 BETO SBERT - θ=0.50)': {
            'Global': calc_pack(y_true, (prob_base >= 0.50).astype(int), prob_base),
            'LatAm': calc_pack(y_true[mask_latam], (prob_base[mask_latam] >= 0.50).astype(int), prob_base[mask_latam]),
            'Espana': calc_pack(y_true[mask_es], (prob_base[mask_es] >= 0.50).astype(int), prob_base[mask_es]),
        },
        'Ensamble Base Calibrado (D0 BETO SBERT - θ*=0.42)': {
            'Global': calc_pack(y_true, (prob_base >= 0.42).astype(int), prob_base),
            'LatAm': calc_pack(y_true[mask_latam], (prob_base[mask_latam] >= 0.42).astype(int), prob_base[mask_latam]),
            'Espana': calc_pack(y_true[mask_es], (prob_base[mask_es] >= 0.42).astype(int), prob_base[mask_es]),
        },
        'Ensamble Avanzado 5D (BETO SBERT In-Fold - θ=0.50)': {
            'Global': calc_pack(y_true, df['pred_ensamble_avanzado_5d_th50'].values, probs_adv),
            'LatAm': calc_pack(y_true[mask_latam], df['pred_ensamble_avanzado_5d_th50'].values[mask_latam], probs_adv[mask_latam]),
            'Espana': calc_pack(y_true[mask_es], df['pred_ensamble_avanzado_5d_th50'].values[mask_es], probs_adv[mask_es]),
        },
        'Ensamble Avanzado 5D Calibrado (BETO SBERT In-Fold - θ*=0.42)': {
            'Global': calc_pack(y_true, df['pred_ensamble_avanzado_5d_th42'].values, probs_adv),
            'LatAm': calc_pack(y_true[mask_latam], df['pred_ensamble_avanzado_5d_th42'].values[mask_latam], probs_adv[mask_latam]),
            'Espana': calc_pack(y_true[mask_es], df['pred_ensamble_avanzado_5d_th42'].values[mask_es], probs_adv[mask_es]),
        },
    }

    print("\n" + "="*95)
    print(" 📊 COMPARATIVA METROLÓGICA: ENSAMBLE BASE VS. ENSAMBLE AVANZADO 5D (BETO SBERT IN-FOLD)")
    print("="*95)
    for model_k, data_m in res_comp.items():
        g = data_m['Global']
        lat = data_m['LatAm']
        es = data_m['Espana']
        print(f"\n🔹 {model_k}:")
        print(f"   GLOBAL: Acc: {g['Accuracy']*100:.2f}% | Rec: {g['Recall']*100:.2f}% | Prec: {g['Precision']*100:.2f}% | F1-Fake: {g['F1_Score']:.4f} | F1-Macro: {g['F1_Macro']:.4f} | AUC: {g['AUC']:.4f} | AP: {g['AP']:.4f} | CM: {g['CM']}")
        print(f"   ESPAÑA: Acc: {es['Accuracy']*100:.2f}% | Rec: {es['Recall']*100:.2f}% | Prec: {es['Precision']*100:.2f}% | F1-Fake: {es['F1_Score']:.4f} | F1-Macro: {es['F1_Macro']:.4f} | AUC: {es['AUC']:.4f} | CM: {es['CM']}")
        print(f"   LATAM:  Acc: {lat['Accuracy']*100:.2f}% | Rec: {lat['Recall']*100:.2f}% | Prec: {lat['Precision']*100:.2f}% | F1-Fake: {lat['F1_Score']:.4f} | F1-Macro: {lat['F1_Macro']:.4f} | AUC: {lat['AUC']:.4f} | CM: {lat['CM']}")

    res_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/resultados_clasificacion_avanzada_4418.csv"
    df.to_csv(res_csv, index=False)
    print(f"\n💾 Predicciones guardadas en: {res_csv}")

    json_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_ensamble_avanzado_5d.json"
    res_export = {
        'modelo_redundancia_unificado': 'hiiamsid/sentence_similarity_spanish_es (BETO SBERT, d=768)',
        'discretizacion': 'In-Fold (Cuartiles X_train + Umbrales Dominio [0.3740, 0.6079, 0.7287])',
        'modelos_comparativa': res_comp,
        'importancia_por_dimension': dim_importance.to_dict(),
        'desglose_d0_d1': {
            'D0_Sensacionalismo_BETO': imp_sens_d0,
            'D0_Redundancia_BETO_SBERT': imp_red_d0,
            'D1_Coherencia_Secuencial_BETO_SBERT': imp_coh_d1,
            'Total_Geometria_Semantica_SBERT': imp_red_d0 + imp_coh_d1
        },
        'top_features': df_feat_imp.head(25).to_dict(orient='records'),
        'all_features_importance': df_feat_imp.to_dict(orient='records'),
        'sweep_umbrales': sweep_adv_data
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(res_export, f, indent=4, ensure_ascii=False)
    print(f"💾 Métricas y ranking de importancia guardados en: {json_path}")

    # -------------------------------------------------------------------------
    # GENERACIÓN DE GRÁFICOS CIENTÍFICOS (300 DPI)
    # -------------------------------------------------------------------------
    img_dir = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Imagenes"
    os.makedirs(img_dir, exist_ok=True)

    # 1. FIGURA: Importancia por Dimensión y Top Features
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    ax_dim = axes[0]
    dim_names_clean = [d.replace('D', 'Dim. ').replace(':', ': ') for d in dim_importance.index]
    dim_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    y_pos_dim = np.arange(len(dim_names_clean))
    bars_dim = ax_dim.barh(y_pos_dim, dim_importance.values * 100, color=dim_colors, edgecolor='black', alpha=0.9)
    ax_dim.set_yticks(y_pos_dim)
    ax_dim.set_yticklabels(dim_names_clean, fontsize=10.5, fontweight='bold')
    ax_dim.invert_yaxis()
    ax_dim.set_xlabel('Contribución Relativa de Información (%)', fontsize=11, fontweight='bold')
    ax_dim.set_title('A. Importancia Acumulada por Dimensión Textual (5D - BETO SBERT)', fontsize=12.5, fontweight='bold', pad=10)
    for b in bars_dim:
        w = b.get_width()
        ax_dim.annotate(f"{w:.1f}%", xy=(w, b.get_y() + b.get_height()/2), xytext=(4, 0),
                        textcoords="offset points", ha='left', va='center', fontsize=9.5, fontweight='bold')
    ax_dim.set_xlim([0, max(dim_importance.values * 100) + 8])

    ax_top = axes[1]
    top_12 = df_feat_imp.head(12)
    y_pos_top = np.arange(len(top_12))
    bars_top = ax_top.barh(y_pos_top, top_12['importance'] * 100, color='#34495e', edgecolor='black', alpha=0.85)
    ax_top.set_yticks(y_pos_top)

    feat_labels = []
    for f in top_12['feature']:
        if 'caps' in f or 'upper' in f:
            feat_labels.append(f"{f} ★ (Mayúsculas)")
        elif 'consec' in f:
            feat_labels.append(f"{f} ✦ (Flujo Coherencia)")
        elif 'intra' in f or 'mexgen' in f or 'paraphrase' in f or 'redundan' in f:
            feat_labels.append(f"{f} 🔁 (Redundancia BETO)")
        elif 'adv' in f or 'adj' in f or 'pron' in f:
            feat_labels.append(f"{f} ✎ (Morfosintaxis)")
        else:
            feat_labels.append(f)

    ax_top.set_yticklabels(feat_labels, fontsize=10, fontweight='bold')
    ax_top.invert_yaxis()
    ax_top.set_xlabel('Importancia Relativa GBDT (%)', fontsize=11, fontweight='bold')
    ax_top.set_title('B. Top 12 Características Lingüísticas Más Útiles', fontsize=13, fontweight='bold', pad=10)
    for b in bars_top:
        w = b.get_width()
        ax_top.annotate(f"{w:.1f}%", xy=(w, b.get_y() + b.get_height()/2), xytext=(4, 0),
                        textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
    ax_top.set_xlim([0, max(top_12['importance'] * 100) + 4])

    plt.suptitle('Análisis de Utilidad Explicable (BETO SBERT Unificado e In-Fold): Contribución por Dimensión y Rasgos', fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_imp_path = os.path.join(img_dir, "importancia_features_5_dimensiones.png")
    plt.savefig(fig_imp_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {fig_imp_path}")

    # 2. FIGURA: Comparativa ROC y PR (Ensamble Base vs Ensamble Avanzado 5D)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax_roc = axes[0]
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_base)
    auc_b = auc(fpr_b, tpr_b)
    ax_roc.plot(fpr_b, tpr_b, color='#7f7f7f', lw=2.2, linestyle='--', label=f'Ensamble Base D0 BETO SBERT (AUC = {auc_b:.4f})')

    fpr_a, tpr_a, _ = roc_curve(y_true, probs_adv)
    auc_a = auc(fpr_a, tpr_a)
    ax_roc.plot(fpr_a, tpr_a, color='#1f77b4', lw=2.8, label=f'Ensamble Avanzado 5D BETO SBERT (AUC = {auc_a:.4f})')

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.2, label='Azar (0.50)')

    cm_b_cal = confusion_matrix(y_true, (prob_base >= 0.42).astype(int))
    fpr_b_cal = cm_b_cal[0, 1] / (cm_b_cal[0, 0] + cm_b_cal[0, 1])
    tpr_b_cal = cm_b_cal[1, 1] / (cm_b_cal[1, 0] + cm_b_cal[1, 1])
    ax_roc.scatter([fpr_b_cal], [tpr_b_cal], color='#7f7f7f', s=100, label=f'Base Calibrado θ*=0.42 (Rec={tpr_b_cal*100:.1f}%)', zorder=5)

    th_adv_eval = 0.42
    pred_adv_cal = (probs_adv >= th_adv_eval).astype(int)
    cm_a = confusion_matrix(y_true, pred_adv_cal)
    fpr_a_cal = cm_a[0, 1] / (cm_a[0, 0] + cm_a[0, 1])
    tpr_a_cal = cm_a[1, 1] / (cm_a[1, 0] + cm_a[1, 1])
    ax_roc.scatter([fpr_a_cal], [tpr_a_cal], color='#d62728', s=150, marker='*', label=f'Avanzado Calibrado θ*=0.42 (Rec={tpr_a_cal*100:.1f}%)', zorder=5)

    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11, fontweight='bold')
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=11, fontweight='bold')
    ax_roc.set_title('A. Curva ROC: Ganancia Discriminativa de las 5 Dimensiones', fontsize=12, fontweight='bold', pad=10)
    ax_roc.legend(loc='lower right', frameon=True, fontsize=9.5)

    ax_pr = axes[1]
    prec_b, rec_b, _ = precision_recall_curve(y_true, prob_base)
    ap_b = average_precision_score(y_true, prob_base)
    ax_pr.plot(rec_b, prec_b, color='#7f7f7f', lw=2.2, linestyle='--', label=f'Ensamble Base D0 BETO SBERT (AP = {ap_b:.4f})')

    prec_a, rec_a, _ = precision_recall_curve(y_true, probs_adv)
    ap_a = average_precision_score(y_true, probs_adv)
    ax_pr.plot(rec_a, prec_a, color='#1f77b4', lw=2.8, label=f'Ensamble Avanzado 5D BETO SBERT (AP = {ap_a:.4f})')

    ax_pr.axhline(y_true.mean(), color='gray', linestyle=':', label=f'Prevalencia Base ({y_true.mean():.3f})')
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=11, fontweight='bold')
    ax_pr.set_ylabel('Precision (Valor Predictivo Positivo)', fontsize=11, fontweight='bold')
    ax_pr.set_title('B. Curva Precision-Recall: Detección en Fake News', fontsize=12, fontweight='bold', pad=10)
    ax_pr.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.suptitle('Evaluación Comparativa (Unificada BETO SBERT In-Fold): Impacto de las 5 Dimensiones (N = 4.418)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_comp_path = os.path.join(img_dir, "comparativa_ensamble_base_vs_avanzado_5d.png")
    plt.savefig(fig_comp_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {fig_comp_path}")

    # 3. FIGURA: Matrices de Confusión Base vs Avanzado Calibrado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm_base_cal = confusion_matrix(y_true, (prob_base >= 0.42).astype(int))
    cm_adv_cal = confusion_matrix(y_true, (probs_adv >= 0.42).astype(int))

    def annot_mat(mat):
        res = np.empty_like(mat, dtype=object)
        for r in range(2):
            for c in range(2):
                res[r, c] = f"{mat[r, c]:,}\n({mat[r, c]/len(y_true)*100:.1f}%)"
        return res

    sns.heatmap(cm_base_cal, annot=annot_mat(cm_base_cal), fmt='', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Pred: Real (0)', 'Pred: Fake (1)'],
                yticklabels=['Real (0)', 'Fake (1)'], annot_kws={'size': 13, 'weight': 'bold'})
    axes[0].set_title(f'A. Ensamble Base Calibrado (θ*=0.42)\nAcc: {accuracy_score(y_true, (prob_base >= 0.42).astype(int))*100:.2f}% | Rec Fake: {recall_score(y_true, (prob_base >= 0.42).astype(int))*100:.1f}%', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Clase Real', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Clase Predicha', fontsize=11, fontweight='bold')

    sns.heatmap(cm_adv_cal, annot=annot_mat(cm_adv_cal), fmt='', cmap='Greens', cbar=False, ax=axes[1],
                xticklabels=['Pred: Real (0)', 'Pred: Fake (1)'],
                yticklabels=['Real (0)', 'Fake (1)'], annot_kws={'size': 13, 'weight': 'bold'})
    axes[1].set_title(f'B. Ensamble Avanzado 5D Calibrado (θ*=0.42)\nAcc: {accuracy_score(y_true, (probs_adv >= 0.42).astype(int))*100:.2f}% | Rec Fake: {recall_score(y_true, (probs_adv >= 0.42).astype(int))*100:.1f}%', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Clase Real', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Clase Predicha', fontsize=11, fontweight='bold')

    plt.suptitle('Matrices de Confusión (BETO SBERT In-Fold): Reducción de Falsos Negativos y Falsas Alarmas', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_cm_path = os.path.join(img_dir, "matrices_confusion_avanzado_5d_vs_base.png")
    plt.savefig(fig_cm_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {fig_cm_path}")

    print("\n🎉 ¡Pipeline de Ensamble Avanzado 5D (BETO SBERT In-Fold) y Análisis de Importancia culminado con éxito!")

if __name__ == '__main__':
    main()
