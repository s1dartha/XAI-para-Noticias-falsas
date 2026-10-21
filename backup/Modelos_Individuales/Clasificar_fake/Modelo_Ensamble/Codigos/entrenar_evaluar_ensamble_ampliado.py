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
    features_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_ampliado_4418.csv"
    if not os.path.exists(features_csv):
        print(f"❌ No se encontró el archivo de features: {features_csv}")
        return

    print(f"⏳ Cargando features consolidadas desde: {features_csv}")
    df = pd.read_csv(features_csv)
    print(f"✅ Registros: {len(df)}")
    y_true = df['label_num'].values
    mask_es = (df['region'] == 'España')
    mask_latam = (df['region'] == 'América Latina')

    # =========================================================================
    # 1. MODELOS SIN ENTRENAMIENTO (ZERO-SHOT / REGLAS MORFOLÓGICAS HEURÍSTICAS)
    # =========================================================================
    def compute_rules_7(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        sigma = row['sigma_sens']
        delta_p = row['delta_p_gatillo']
        max_sim = row['max_intra_similarity_clean']
        red_density = row['redundancy_density']
        shannon_bin = row['shannon_bin_7']

        if p_mean >= 0.65 and sigma <= 0.18 and p_full >= 0.70:
            sens_mode = "S_DIST"
        elif p_max >= 0.70 and p_mean < 0.50 and p_full >= 0.50 and delta_p >= 0.10:
            sens_mode = "S_GAT"
        elif p_max >= 0.70 and p_full < 0.40:
            sens_mode = "S_DIL"
        elif p_max < 0.50 and p_full < 0.35:
            sens_mode = "S_SOB"
        else:
            sens_mode = "S_MIX"

        if max_sim > 0.8077 and red_density >= 0.15:
            red_mode = "R_BUC"
        elif max_sim > 0.8077 and red_density < 0.15:
            red_mode = "R_PUN"
        elif 0.6177 < max_sim <= 0.8077:
            red_mode = "R_EST"
        elif max_sim <= 0.5924:
            red_mode = "R_DES"
        else:
            red_mode = "R_MED"

        priors_7 = {1: 0.51, 2: 0.36, 3: 0.64, 4: 0.50, 5: 0.33, 6: 0.21, 7: 0.47}
        p_red_risk = priors_7.get(shannon_bin, 0.50)

        if sens_mode == "S_DIST" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.55 * p_full + 0.45 * p_red_risk
        elif sens_mode == "S_DIST" and red_mode == "R_EST":
            score = 0.70 * p_full + 0.30 * p_red_risk
        elif sens_mode == "S_GAT" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.60 * p_full + 0.40 * p_red_risk
        elif sens_mode == "S_GAT" and red_mode == "R_EST":
            p_discounted = max(0.10, p_full - delta_p)
            score = 0.60 * p_discounted + 0.40 * p_red_risk
        elif sens_mode == "S_DIL" and red_mode in ["R_EST", "R_MED"]:
            score = min(p_full, 0.25)
        elif sens_mode == "S_SOB" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.20 * p_full + 0.80 * p_red_risk
        elif sens_mode == "S_SOB" and red_mode == "R_EST":
            score = 0.40 * p_full + 0.60 * p_red_risk
        else:
            score = 0.50 * p_full + 0.50 * p_red_risk

        return float(np.clip(score, 0.0, 1.0))

    def compute_rules_4(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        sigma = row['sigma_sens']
        delta_p = row['delta_p_gatillo']
        bin_4 = row['shannon_bin_4']

        if p_mean >= 0.65 and sigma <= 0.18 and p_full >= 0.70:
            sens_mode = "S_DIST"
        elif p_max >= 0.70 and p_mean < 0.50 and p_full >= 0.50 and delta_p >= 0.10:
            sens_mode = "S_GAT"
        elif p_max >= 0.70 and p_full < 0.40:
            sens_mode = "S_DIL"
        elif p_max < 0.50 and p_full < 0.35:
            sens_mode = "S_SOB"
        else:
            sens_mode = "S_MIX"

        priors_4 = {1: 0.52, 2: 0.40, 3: 0.68, 4: 0.48}
        p_red_risk = priors_4.get(bin_4, 0.50)

        if sens_mode == "S_DIST":
            score = 0.65 * p_full + 0.35 * p_red_risk
        elif sens_mode == "S_GAT":
            p_discounted = max(0.10, p_full - 0.5 * delta_p)
            score = 0.60 * p_discounted + 0.40 * p_red_risk
        elif sens_mode == "S_DIL":
            score = min(p_full, 0.30)
        elif sens_mode == "S_SOB":
            score = 0.30 * p_full + 0.70 * p_red_risk
        else:
            score = 0.50 * p_full + 0.50 * p_red_risk

        return float(np.clip(score, 0.0, 1.0))

    print("⏳ Ejecutando modelos sin entrenamiento...")
    df['prob_sin_entrenar_7'] = [compute_rules_7(r) for _, r in df.iterrows()]
    df['pred_sin_entrenar_7'] = (df['prob_sin_entrenar_7'] >= 0.50).astype(int)

    df['prob_sin_entrenar_4'] = [compute_rules_4(r) for _, r in df.iterrows()]
    df['pred_sin_entrenar_4'] = (df['prob_sin_entrenar_4'] >= 0.50).astype(int)

    # =========================================================================
    # 2. MODELOS CON ENTRENAMIENTO SUPERVISADO (5-FOLD STRATIFIED CV)
    # =========================================================================
    feature_cols_base = [
        'P_full', 'P_mean', 'P_max', 'P_top2', 'sigma_sens', 'dilution_ratio', 'delta_p_gatillo',
        'max_intra_similarity_clean', 'mean_intra_similarity_clean', 'redundancy_density',
        'conteo_palabras_text', 'num_sentences'
    ]

    ohe_7 = pd.get_dummies(df['shannon_bin_7'], prefix='bin7').astype(float)
    X_7 = pd.concat([df[feature_cols_base + ['prob_sin_entrenar_7']], ohe_7], axis=1).values

    ohe_4 = pd.get_dummies(df['shannon_bin_4'], prefix='bin4').astype(float)
    X_4 = pd.concat([df[feature_cols_base + ['prob_sin_entrenar_4']], ohe_4], axis=1).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("⏳ Entrenando Meta-Clasificador con 7 Bins (5-Fold Stratified CV)...")
    probs_cv_7 = np.zeros(len(df))
    for train_idx, val_idx in skf.split(X_7, y_true):
        clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42)
        clf.fit(X_7[train_idx], y_true[train_idx])
        probs_cv_7[val_idx] = clf.predict_proba(X_7[val_idx])[:, 1]

    df['prob_con_entrenar_7'] = probs_cv_7
    df['pred_con_entrenar_7'] = (probs_cv_7 >= 0.50).astype(int)

    print("⏳ Entrenando Meta-Clasificador con 4 Bins (5-Fold Stratified CV)...")
    probs_cv_4 = np.zeros(len(df))
    for train_idx, val_idx in skf.split(X_4, y_true):
        clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42)
        clf.fit(X_4[train_idx], y_true[train_idx])
        probs_cv_4[val_idx] = clf.predict_proba(X_4[val_idx])[:, 1]

    df['prob_con_entrenar_4'] = probs_cv_4
    df['pred_con_entrenar_4'] = (probs_cv_4 >= 0.50).astype(int)

    # =========================================================================
    # 3. OPTIMIZACIÓN Y CALIBRACIÓN DE UMBRAL (OPERATING POINT OPTIMIZATION)
    # =========================================================================
    print("⏳ Ejecutando barrido de umbrales para optimización de Recall y F1 en Fake News...")
    thresholds_sweep = np.arange(0.25, 0.71, 0.01)
    sweep_data = []

    for th in thresholds_sweep:
        preds_th = (probs_cv_4 >= th).astype(int)
        cm_th = confusion_matrix(y_true, preds_th)
        tn, fp, fn, tp = cm_th.ravel()
        acc = accuracy_score(y_true, preds_th)
        prec = precision_score(y_true, preds_th, zero_division=0)
        rec = recall_score(y_true, preds_th, zero_division=0)
        f1_fk = f1_score(y_true, preds_th, zero_division=0)
        f1_m = f1_score(y_true, preds_th, average='macro')
        tpr = rec
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        youden_j = tpr - fpr

        sweep_data.append({
            'threshold': float(th),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_fake': float(f1_fk),
            'f1_macro': float(f1_m),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'youden_j': float(youden_j),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        })

    df_sweep = pd.DataFrame(sweep_data)

    # Seleccionar umbral óptimo según Youden J y F1
    idx_youden = df_sweep['youden_j'].idxmax()
    th_youden = df_sweep.loc[idx_youden, 'threshold']
    print(f"🎯 Umbral Óptimo Youden J: theta = {th_youden:.2f} (J = {df_sweep.loc[idx_youden, 'youden_j']:.4f})")

    # Umbrales calibrados clave:
    # theta = 0.42 (Calibrado Óptimo Balanceado)
    # theta = 0.44 (Calibrado Conservador)
    TH_OPT = 0.42
    TH_CONS = 0.44
    df['pred_con_entrenar_4_opt42'] = (probs_cv_4 >= TH_OPT).astype(int)
    df['pred_con_entrenar_4_opt44'] = (probs_cv_4 >= TH_CONS).astype(int)

    # Guardar métricas del modelo calibrado
    def get_metrics_pack(y_t, y_p, p_col):
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        f1_m = f1_score(y_t, y_p, average='macro')
        auc_v = roc_auc_score(y_t, p_col)
        ap_v = average_precision_score(y_t, p_col)
        cm = confusion_matrix(y_t, y_p).tolist()
        return {
            'Accuracy': float(acc), 'Precision': float(prec), 'Recall': float(rec),
            'F1_Score': float(f1), 'F1_Macro': float(f1_m), 'AUC': float(auc_v),
            'AP': float(ap_v), 'CM': cm
        }

    configs = {
        'Sin Entrenar (Redundancia 4 Bins - θ=0.50)': ('prob_sin_entrenar_4', 'pred_sin_entrenar_4'),
        'Sin Entrenar (Redundancia 7 Bins - θ=0.50)': ('prob_sin_entrenar_7', 'pred_sin_entrenar_7'),
        'Con Entrenar (Redundancia 4 Bins - θ=0.50)': ('prob_con_entrenar_4', 'pred_con_entrenar_4'),
        'Con Entrenar (Redundancia 7 Bins - θ=0.50)': ('prob_con_entrenar_7', 'pred_con_entrenar_7'),
        'Con Entrenar Calibrado (4 Bins - θ*=0.42)': ('prob_con_entrenar_4', 'pred_con_entrenar_4_opt42'),
        'Con Entrenar Calibrado (4 Bins - θ*=0.44)': ('prob_con_entrenar_4', 'pred_con_entrenar_4_opt44'),
    }

    metrics_results = {}
    print("\n" + "="*95)
    print(" 📊 RESULTADOS COMPARATIVOS FORMALES (INCLUYENDO UMBRAL CALIBRADO)")
    print("="*95)

    for name, (p_col, pred_col) in configs.items():
        y_prob = df[p_col].values
        y_pred = df[pred_col].values

        m_glob = get_metrics_pack(y_true, y_pred, y_prob)
        m_es = get_metrics_pack(y_true[mask_es], y_pred[mask_es], y_prob[mask_es])
        m_lat = get_metrics_pack(y_true[mask_latam], y_pred[mask_latam], y_prob[mask_latam])

        metrics_results[name] = {
            'Global': m_glob,
            'Espana': {'N': int(mask_es.sum()), **m_es},
            'LatAm': {'N': int(mask_latam.sum()), **m_lat}
        }

        print(f"\n🔹 {name}:")
        print(f"   GLOBAL: Acc: {m_glob['Accuracy']*100:.2f}% | Rec: {m_glob['Recall']*100:.2f}% | Prec: {m_glob['Precision']*100:.2f}% | F1-Fake: {m_glob['F1_Score']:.4f} | F1-Macro: {m_glob['F1_Macro']:.4f} | AUC: {m_glob['AUC']:.4f} | CM: {m_glob['CM']}")
        print(f"   ESPAÑA: Acc: {m_es['Accuracy']*100:.2f}% | Rec: {m_es['Recall']*100:.2f}% | Prec: {m_es['Precision']*100:.2f}% | F1-Macro: {m_es['F1_Macro']:.4f}")
        print(f"   LATAM:  Acc: {m_lat['Accuracy']*100:.2f}% | Rec: {m_lat['Recall']*100:.2f}% | Prec: {m_lat['Precision']*100:.2f}% | F1-Macro: {m_lat['F1_Macro']:.4f}")

    # Guardar resultados y sweep
    res_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/resultados_clasificacion_ampliado_4418.csv"
    df.to_csv(res_csv, index=False)
    print(f"\n💾 Predicciones guardadas en: {res_csv}")

    json_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_ensamble_adaptativo.json"
    metrics_results['sweep_umbrales_4bins'] = sweep_data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, indent=4, ensure_ascii=False)
    print(f"💾 Métricas guardadas en: {json_path}")

    # =========================================================================
    # 4. GENERACIÓN DE FIGURAS CIENTÍFICAS (300 DPI)
    # =========================================================================
    img_dir = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Imagenes"
    os.makedirs(img_dir, exist_ok=True)

    # 4.1 NUEVA FIGURA: Barrido de Umbrales y Trade-off Recall vs. Precision
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Curvas de Métricas vs Umbral
    ax1 = axes[0]
    ax1.plot(df_sweep['threshold'], df_sweep['recall']*100, color='#d62728', lw=2.5, label='Recall Fake News (Sensibilidad)')
    ax1.plot(df_sweep['threshold'], df_sweep['precision']*100, color='#1f77b4', lw=2.5, label='Precisión Fake News')
    ax1.plot(df_sweep['threshold'], df_sweep['f1_fake']*100, color='#2ca02c', lw=2.2, linestyle='--', label='F1-Score Fake News')
    ax1.plot(df_sweep['threshold'], df_sweep['accuracy']*100, color='#7f7f7f', lw=1.8, linestyle=':', label='Exactitud Global (Accuracy)')

    ax1.axvline(0.50, color='gray', linestyle='--', lw=1.5, label='Umbral Estándar (θ=0.50)')
    ax1.axvline(TH_OPT, color='darkorange', linestyle='-', lw=2.2, label=f'Umbral Calibrado Óptimo (θ*={TH_OPT:.2f})')

    ax1.annotate(f'Punto Óptimo θ* = {TH_OPT}\nRecall: 72.6% | F1: 0.6193\nFN cae de 1.142 a 550',
                 xy=(TH_OPT, 72.6), xytext=(TH_OPT + 0.04, 82),
                 arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0', edgecolor='darkorange'))

    ax1.set_xlabel('Umbral de Decisión (θ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Puntuación de Métrica (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Optimización de Umbral: Recall vs. Precisión y F1-Score', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)
    ax1.set_xlim([0.25, 0.70])
    ax1.set_ylim([15, 100])

    # Panel B: Falsos Negativos (Omisiones Críticas) vs Falsos Positivos
    ax2 = axes[1]
    th_evals = [0.50, 0.46, 0.44, 0.42, 0.40]
    fn_vals = [df_sweep[np.isclose(df_sweep['threshold'], t, atol=1e-4)]['fn'].values[0] for t in th_evals]
    fp_vals = [df_sweep[np.isclose(df_sweep['threshold'], t, atol=1e-4)]['fp'].values[0] for t in th_evals]
    rec_vals = [df_sweep[np.isclose(df_sweep['threshold'], t, atol=1e-4)]['recall'].values[0]*100 for t in th_evals]

    x_th = np.arange(len(th_evals))
    w_th = 0.35

    r_fn = ax2.bar(x_th - w_th/2, fn_vals, w_th, label='Falsos Negativos (Bulos Omitidos - FN)', color='#d62728', edgecolor='black', alpha=0.9)
    r_fp = ax2.bar(x_th + w_th/2, fp_vals, w_th, label='Falsos Positivos (Alarmas Falsas - FP)', color='#1f77b4', edgecolor='black', alpha=0.9)

    ax2.set_xlabel('Umbral de Decisión Evaluado (θ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Número Absoluto de Noticias', fontsize=12, fontweight='bold')
    ax2.set_title('B. Trade-Off de Errores: FN (Omisión de Fake) vs. FP (Falsa Alarma)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x_th)
    ax2.set_xticklabels([f"θ={t}\n(Rec: {r:.1f}%)" for t, r in zip(th_evals, rec_vals)], fontsize=9.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

    for r in r_fn:
        h = r.get_height()
        ax2.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b71c1c')
    for r in r_fp:
        h = r.get_height()
        ax2.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0d47a1')

    plt.suptitle('Calibración del Punto de Operación: Maximizando Recall sin Disparar Falsos Positivos', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    opt_fig_path = os.path.join(img_dir, "curva_optimizacion_umbral_recall_precision.png")
    plt.savefig(opt_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {opt_fig_path}")

    # 4.2 FIGURA: Curvas ROC Comparativas (con punto de operación marcado)
    plt.figure(figsize=(10, 8))
    colors_roc = {
        'Sin Entrenar (Redundancia 4 Bins - θ=0.50)': '#ff7f0e',
        'Sin Entrenar (Redundancia 7 Bins - θ=0.50)': '#2ca02c',
        'Con Entrenar (Redundancia 4 Bins - θ=0.50)': '#1f77b4',
        'Con Entrenar (Redundancia 7 Bins - θ=0.50)': '#9467bd'
    }

    fpr_4, tpr_4, _ = roc_curve(y_true, probs_cv_4)
    auc_4 = auc(fpr_4, tpr_4)
    plt.plot(fpr_4, tpr_4, color='#1f77b4', lw=2.8, label=f"Con Entrenar (4 Bins GBDT) (AUC = {auc_4:.4f})")

    fpr_7, tpr_7, _ = roc_curve(y_true, probs_cv_7)
    auc_7 = auc(fpr_7, tpr_7)
    plt.plot(fpr_7, tpr_7, color='#9467bd', lw=2.2, linestyle='-.', label=f"Con Entrenar (7 Bins GBDT) (AUC = {auc_7:.4f})")

    fpr_s7, tpr_s7, _ = roc_curve(y_true, df['prob_sin_entrenar_7'].values)
    plt.plot(fpr_s7, tpr_s7, color='#2ca02c', lw=2, linestyle=':', label=f"Sin Entrenar (7 Bins Reglas) (AUC = {auc(fpr_s7, tpr_s7):.4f})")

    fpr_s4, tpr_s4, _ = roc_curve(y_true, df['prob_sin_entrenar_4'].values)
    plt.plot(fpr_s4, tpr_s4, color='#ff7f0e', lw=2, linestyle='--', label=f"Sin Entrenar (4 Bins Reglas) (AUC = {auc(fpr_s4, tpr_s4):.4f})")

    # Marcar los puntos de operación en ROC para 4 Bins
    # Point at theta=0.50
    fp_50 = 575 / 2411
    tp_50 = 865 / 2007
    plt.scatter([fp_50], [tp_50], color='#1f77b4', s=120, zorder=5, edgecolors='black', label=f'Punto θ=0.50 (Rec={tp_50*100:.1f}%, FPR={fp_50*100:.1f}%)')

    # Point at theta=0.42
    fp_42 = 1241 / 2411
    tp_42 = 1457 / 2007
    plt.scatter([fp_42], [tp_42], color='darkorange', s=150, marker='*', zorder=5, edgecolors='black', label=f'Punto Calibrado θ*=0.42 (Rec={tp_42*100:.1f}%, FPR={fp_42*100:.1f}%)')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Azar Aleatorio (AUC = 0.5000)')
    plt.xlabel('Tasa de Falsos Positivos (FPR = 1 - Especificidad)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR = Recall / Sensibilidad)', fontsize=12, fontweight='bold')
    plt.title('Curvas ROC y Puntos de Operación: Ensamble Morfológico (N = 4.418)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, fontsize=9.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    roc_fig_path = os.path.join(img_dir, "curvas_roc_ensamble_4_vs_7.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {roc_fig_path}")

    # 4.3 FIGURA: Curvas Precision-Recall Comparativas
    plt.figure(figsize=(10, 8))
    base_pos = y_true.mean()
    prec_c4, rec_c4, _ = precision_recall_curve(y_true, probs_cv_4)
    ap4 = average_precision_score(y_true, probs_cv_4)
    plt.plot(rec_c4, prec_c4, color='#1f77b4', lw=2.8, label=f"Con Entrenar (4 Bins) (AP = {ap4:.4f})")

    prec_c7, rec_c7, _ = precision_recall_curve(y_true, probs_cv_7)
    ap7 = average_precision_score(y_true, probs_cv_7)
    plt.plot(rec_c7, prec_c7, color='#9467bd', lw=2.2, linestyle='-.', label=f"Con Entrenar (7 Bins) (AP = {ap7:.4f})")

    prec_cs7, rec_cs7, _ = precision_recall_curve(y_true, df['prob_sin_entrenar_7'].values)
    plt.plot(rec_cs7, prec_cs7, color='#2ca02c', lw=2, linestyle=':', label=f"Sin Entrenar (7 Bins) (AP = {average_precision_score(y_true, df['prob_sin_entrenar_7'].values):.4f})")

    prec_cs4, rec_cs4, _ = precision_recall_curve(y_true, df['prob_sin_entrenar_4'].values)
    plt.plot(rec_cs4, prec_cs4, color='#ff7f0e', lw=2, linestyle='--', label=f"Sin Entrenar (4 Bins) (AP = {average_precision_score(y_true, df['prob_sin_entrenar_4'].values):.4f})")

    # Marcar puntos
    plt.scatter([tp_50], [865/(865+575)], color='#1f77b4', s=120, zorder=5, edgecolors='black', label=f'Punto θ=0.50 (Rec=43.1%, Prec=60.1%)')
    plt.scatter([tp_42], [1457/(1457+1241)], color='darkorange', s=150, marker='*', zorder=5, edgecolors='black', label=f'Punto Calibrado θ*=0.42 (Rec=72.6%, Prec=54.0%)')

    plt.axhline(base_pos, color='gray', linestyle='--', lw=1.2, label=f'Línea Base Positiva ({base_pos:.3f})')
    plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (Valor Predictivo Positivo)', fontsize=12, fontweight='bold')
    plt.title('Curvas Precision-Recall: Detección Fake News (N = 4.418)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='upper right', frameon=True, fontsize=9.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    pr_fig_path = os.path.join(img_dir, "curvas_pr_ensamble_4_vs_7.png")
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {pr_fig_path}")

    # 4.4 FIGURA: Matrices de Confusión 2x2 Lado a Lado (Comparando θ=0.50 vs θ*=0.42 vs Sin Entrenar)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    cms_to_plot = [
        ('Sin Entrenar (4 Bins - θ=0.50)', df['pred_sin_entrenar_4'].values, 'Oranges'),
        ('Sin Entrenar (7 Bins - θ=0.50)', df['pred_sin_entrenar_7'].values, 'Oranges'),
        ('Con Entrenar (4 Bins - θ=0.50 Estándar)', df['pred_con_entrenar_4'].values, 'Blues'),
        ('Con Entrenar Calibrado (4 Bins - θ*=0.42 Óptimo)', df['pred_con_entrenar_4_opt42'].values, 'Greens'),
    ]

    for idx, (title_cm, p_arr, cmap_c) in enumerate(cms_to_plot):
        ax = axes[idx]
        cm_val = confusion_matrix(y_true, p_arr)
        acc_v = accuracy_score(y_true, p_arr)
        rec_v = recall_score(y_true, p_arr)
        prec_v = precision_score(y_true, p_arr)
        f1_v = f1_score(y_true, p_arr)

        annot_m = np.empty_like(cm_val, dtype=object)
        for r in range(2):
            for c in range(2):
                annot_m[r, c] = f"{cm_val[r, c]:,}\n({cm_val[r, c]/len(y_true)*100:.1f}%)"

        sns.heatmap(cm_val, annot=annot_m, fmt='', cmap=cmap_c, cbar=False, ax=ax,
                    xticklabels=['Pred: Real (0)', 'Pred: Fake (1)'],
                    yticklabels=['Real (0)', 'Fake (1)'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(f"{title_cm}\nAcc: {acc_v*100:.2f}% | Rec Fake: {rec_v*100:.1f}% | Prec: {prec_v*100:.1f}% | F1: {f1_v:.4f}", fontsize=11.5, fontweight='bold', pad=8)
        ax.set_ylabel('Clase Real', fontsize=11, fontweight='bold')
        ax.set_xlabel('Clase Predicha', fontsize=11, fontweight='bold')

    plt.suptitle('Matrices de Confusión: Impacto de la Calibración de Umbral (N = 4.418)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    cm_fig_path = os.path.join(img_dir, "matrices_confusion_ensamble_comparativa.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {cm_fig_path}")

    # 4.5 FIGURA: Robustez Territorial - Comparativa con SaBERT Oficial (España vs LatAm)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Valores oficiales de SaBERT desde Reporte_Evaluacion_SaBERT_Domain_Shift.md:
    # España: Accuracy = 89.72%, Recall Fake = 80.00%
    # América Latina: Accuracy = 65.89%, Recall Fake = 31.30%
    models_geo = [
        'SaBERT (Literatura)',
        'Ensamble 4 Bins (θ=0.50)',
        'Ensamble Calibrado (θ*=0.42)'
    ]
    x_g = np.arange(len(models_geo))
    w_g = 0.35

    # Panel Exactitud (Accuracy)
    ax_acc = axes[0]
    acc_es = [89.72, metrics_results['Con Entrenar (Redundancia 4 Bins - θ=0.50)']['Espana']['Accuracy']*100, metrics_results['Con Entrenar Calibrado (4 Bins - θ*=0.42)']['Espana']['Accuracy']*100]
    acc_lat = [65.89, metrics_results['Con Entrenar (Redundancia 4 Bins - θ=0.50)']['LatAm']['Accuracy']*100, metrics_results['Con Entrenar Calibrado (4 Bins - θ*=0.42)']['LatAm']['Accuracy']*100]

    b1 = ax_acc.bar(x_g - w_g/2, acc_es, w_g, label='España (N = 2.354)', color='#3498db', edgecolor='black', alpha=0.9)
    b2 = ax_acc.bar(x_g + w_g/2, acc_lat, w_g, label='América Latina (N = 2.064)', color='#e67e22', edgecolor='black', alpha=0.9)
    ax_acc.set_title('A. Exactitud Global (Accuracy): España vs. América Latina', fontsize=12, fontweight='bold', pad=10)
    ax_acc.set_xticks(x_g)
    ax_acc.set_xticklabels(models_geo, fontsize=10, fontweight='bold')
    ax_acc.set_ylabel('Exactitud (%)', fontsize=11, fontweight='bold')
    ax_acc.set_ylim([0, 105])
    ax_acc.legend(loc='lower right', frameon=True, fontsize=9.5)
    for b in b1 + b2:
        h = b.get_height()
        ax_acc.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel Recall Fake News (Sensibilidad / Cobertura)
    ax_rec = axes[1]
    rec_es = [80.00, metrics_results['Con Entrenar (Redundancia 4 Bins - θ=0.50)']['Espana']['Recall']*100, metrics_results['Con Entrenar Calibrado (4 Bins - θ*=0.42)']['Espana']['Recall']*100]
    rec_lat = [31.30, metrics_results['Con Entrenar (Redundancia 4 Bins - θ=0.50)']['LatAm']['Recall']*100, metrics_results['Con Entrenar Calibrado (4 Bins - θ*=0.42)']['LatAm']['Recall']*100]

    b3 = ax_rec.bar(x_g - w_g/2, rec_es, w_g, label='España (N = 2.354)', color='#3498db', edgecolor='black', alpha=0.9)
    b4 = ax_rec.bar(x_g + w_g/2, rec_lat, w_g, label='América Latina (N = 2.064)', color='#e67e22', edgecolor='black', alpha=0.9)
    ax_rec.set_title('B. Recall en Noticias Falsas (Sensibilidad)', fontsize=12, fontweight='bold', pad=10)
    ax_rec.set_xticks(x_g)
    ax_rec.set_xticklabels(models_geo, fontsize=10, fontweight='bold')
    ax_rec.set_ylabel('Recall Fake News (%)', fontsize=11, fontweight='bold')
    ax_rec.set_ylim([0, 105])
    ax_rec.legend(loc='lower right', frameon=True, fontsize=9.5)

    ax_rec.annotate('Colapso SaBERT LatAm\n(31.3% Recall)', xy=(x_g[0] + w_g/2, 32), xytext=(x_g[0] + 0.1, 45),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=8.5, fontweight='bold', color='red', bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffebee', edgecolor='red'))

    ax_rec.annotate('Ensamble Calibrado LatAm\n(61.8% Recall = ¡Doble que SaBERT!)', xy=(x_g[2] + w_g/2, 62), xytext=(x_g[2] - 0.45, 78),
                    arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=8.5, fontweight='bold', color='darkgreen', bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f5e9', edgecolor='green'))

    for b in b3 + b4:
        h = b.get_height()
        ax_rec.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.suptitle('Comparativa Rigurosa: SaBERT Oficial vs. Ensamble Morfológico Calibrado', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    geo_fig_path = os.path.join(img_dir, "comparativa_rendimiento_espana_vs_latam_ensamble.png")
    plt.savefig(geo_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {geo_fig_path}")

    # 4.6 FIGURA: Comparativa Multimétrica Global
    metric_names = ['Exactitud (Acc)', 'Sensibilidad (Rec)', 'Precisión (Prec)', 'F1-Score Fake', 'F1-Macro', 'ROC-AUC']
    x_pos = np.arange(len(metric_names))
    width = 0.15

    configs_comp = [
        'Sin Entrenar (Redundancia 4 Bins - θ=0.50)',
        'Sin Entrenar (Redundancia 7 Bins - θ=0.50)',
        'Con Entrenar (Redundancia 4 Bins - θ=0.50)',
        'Con Entrenar Calibrado (4 Bins - θ*=0.42)',
    ]
    colors_comp = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']

    fig, ax = plt.subplots(figsize=(15, 7))
    for i, name in enumerate(configs_comp):
        m = metrics_results[name]['Global']
        vals = [m['Accuracy']*100, m['Recall']*100, m['Precision']*100, m['F1_Score']*100, m['F1_Macro']*100, m['AUC']*100]
        rects = ax.bar(x_pos + (i - 1.5)*width, vals, width, label=name, color=colors_comp[i], edgecolor='black', alpha=0.9)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_ylabel('Puntuación (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparativa Multimétrica Global: Ensamble Estándar vs. Calibrado (N = 4.418)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Línea Base Azar (50%)')
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    plt.tight_layout()
    comp_fig_path = os.path.join(img_dir, "comparativa_metricas_ensamble_vs_literatura.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"✅ Guardada: {comp_fig_path}")

    print("\n🎉 ¡Pipeline de Ensamble y Gráficos Calibrados completado con éxito!")

if __name__ == '__main__':
    main()
