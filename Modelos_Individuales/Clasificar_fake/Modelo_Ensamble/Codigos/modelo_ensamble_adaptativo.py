import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

def run_adaptive_ensemble():
    features_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_2604.csv"
    if not os.path.exists(features_csv):
        print(f"❌ No se encontró el archivo de features: {features_csv}")
        return

    print(f"⏳ Cargando características morfológicas BETO SBERT desde: {features_csv}")
    df = pd.read_csv(features_csv)
    print(f"✅ Registros cargados: {len(df)}")

    # 1. Definir la Función de Enrutamiento y Gating Morfológico (Modelo Basado en Reglas Expertas sobre BETO SBERT)
    def compute_morphological_gating(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        p_top2 = row['P_top2']
        sigma = row['sigma_sens']
        dr = row['dilution_ratio']
        delta_p = row['delta_p_gatillo']

        max_sim = row['max_intra_similarity_clean']
        s2_sim = row.get('second_max_intra_similarity_clean', max_sim * 0.85)
        mean_sim = row['mean_intra_similarity_clean']
        red_density = row.get('strict_redundancy_density_p95', row['redundancy_density'])
        shannon_bin = row['shannon_bin_4']

        # --- PERFIL SENSACIONALISMO (BETO JJNeila) ---
        if p_mean >= 0.65 and sigma <= 0.18 and p_full >= 0.70:
            sens_mode = "S_DIST" # Distribuido / Saturado
        elif p_max >= 0.70 and p_mean < 0.50 and p_full >= 0.50 and delta_p >= 0.10:
            sens_mode = "S_GAT"  # Efecto Gatillo Activo
        elif p_max >= 0.70 and p_full < 0.40:
            sens_mode = "S_DIL"  # Dilución Contextual (Amortiguado)
        elif p_max < 0.50 and p_full < 0.35:
            sens_mode = "S_SOB"  # Sobriedad Uniforme
        else:
            sens_mode = "S_MIX"  # Mixto / Moderado

        # --- PERFIL REDUNDANCIA (BETO SBERT: 0.3740, 0.5340, 0.6079, 0.7287 y MexGen S_(2)) ---
        if max_sim >= 0.7287 or (max_sim >= 0.5996 and s2_sim >= 0.5996):
            red_mode = "R_BUC"   # Hiper-Redundancia Circular o Red Multi-Bucle Persistente (MexGen)
        elif max_sim >= 0.5996 and s2_sim < 0.5996:
            red_mode = "R_PUN"   # Redundancia Focalizada de Bucle Único
        elif 0.3740 < max_sim < 0.5996:
            red_mode = "R_EST"   # Cohesión Profesional Estándar en Banda de Shannon
        else:
            red_mode = "R_DES"   # Desarticulado / Fragmentación (<= 0.3740)

        # --- MATRIZ DE DECISIÓN Y SCORE DE MANIPULACIÓN ---
        # Priors por Macro-Estrato de BETO SBERT (Bin 1: 0.552, Bin 2: 0.480, Bin 3: 0.544, Bin 4: 0.630)
        shannon_priors_4 = {
            1: 0.552, # Bin 1: Fragmentación (<= 0.3740)
            2: 0.480, # Bin 2: Cohesión periodística estándar (0.3740 - 0.6079)
            3: 0.544, # Bin 3: Reiteración moderada (0.6079 - 0.7287)
            4: 0.630  # Bin 4: Hiper-redundancia circular (> 0.7287)
        }
        p_red_risk = shannon_priors_4.get(shannon_bin, 0.50)
        if red_mode == "R_BUC":
            p_red_risk = max(p_red_risk, 0.65)

        if sens_mode == "S_DIST" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.55 * p_full + 0.45 * p_red_risk
            decision_note = "Desinformación Saturada en Bucle"
        elif sens_mode == "S_DIST" and red_mode == "R_EST":
            score = 0.70 * p_full + 0.30 * p_red_risk
            decision_note = "Sensacionalismo Distribuido Coherente"
        elif sens_mode == "S_GAT" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.60 * p_full + 0.40 * p_red_risk
            decision_note = "Clickbait con Bucle Argumental"
        elif sens_mode == "S_GAT" and red_mode == "R_EST":
            p_discounted = max(0.10, p_full - delta_p)
            score = 0.60 * p_discounted + 0.40 * p_red_risk
            decision_note = "Clickbait Comercial Amortiguado"
        elif sens_mode == "S_DIL" and red_mode in ["R_EST", "R_PUN"]:
            score = min(p_full, 0.25)
            decision_note = "Periodismo Serio con Dilución Contextual"
        elif sens_mode == "S_SOB" and red_mode == "R_BUC":
            score = 0.20 * p_full + 0.80 * p_red_risk
            decision_note = "Desinformación Sofisticada Encubierta (Multi-Bucle)"
        elif sens_mode == "S_SOB" and red_mode == "R_EST":
            score = 0.50 * p_full + 0.50 * p_red_risk
            decision_note = "Texto Neutro y Cohesión Profesional"
        else:
            score = 0.48 * p_full + 0.52 * p_red_risk
            decision_note = "Patrón Morfológico Mixto"

        return sens_mode, red_mode, float(np.clip(score, 0.0, 1.0)), decision_note

    print("⏳ Aplicando motor de inferencia morfológico adaptativo (BETO SBERT) a todas las noticias...")
    gating_results = [compute_morphological_gating(row) for _, row in df.iterrows()]
    df['sens_mode'] = [r[0] for r in gating_results]
    df['red_mode'] = [r[1] for r in gating_results]
    df['score_ensamble_reglas'] = [r[2] for r in gating_results]
    df['diagnostico_estilometrico'] = [r[3] for r in gating_results]

    y_true = df['label_num'].values
    y_prob_rules = df['score_ensamble_reglas'].values
    y_pred_rules = (y_prob_rules >= 0.50).astype(int)

    acc_rules = accuracy_score(y_true, y_pred_rules)
    prec_rules = precision_score(y_true, y_pred_rules, zero_division=0)
    rec_rules = recall_score(y_true, y_pred_rules, zero_division=0)
    f1_rules = f1_score(y_true, y_pred_rules, zero_division=0)
    auc_rules = roc_auc_score(y_true, y_prob_rules)
    ap_rules = average_precision_score(y_true, y_prob_rules)
    cm_rules = confusion_matrix(y_true, y_pred_rules).tolist()

    print("\n" + "="*80)
    print(" 📊 RESULTADOS DEL ENSAMBLE BASADO EN REGLAS MORFOLÓGICAS ADAPTATIVAS (BETO SBERT)")
    print("="*80)
    print(f"Accuracy:  {acc_rules:.4f} ({acc_rules*100:.2f}%)")
    print(f"Precision: {prec_rules:.4f}")
    print(f"Recall:    {rec_rules:.4f}")
    print(f"F1-Score:  {f1_rules:.4f}")
    print(f"ROC-AUC:   {auc_rules:.4f}")
    print(f"PR-AUC:    {ap_rules:.4f}")
    print(f"Confusion Matrix: {cm_rules}")

    # 2. Meta-Clasificador Supervisado Calibrado con Discretización Estricta In-Fold (5-Fold Stratified CV)
    feature_cols = [
        'P_full', 'P_mean', 'P_max', 'P_top2', 'sigma_sens', 'dilution_ratio', 'delta_p_gatillo',
        'max_intra_similarity_clean', 'second_max_intra_similarity_clean', 'top3_mean_intra_similarity_clean',
        'mean_intra_similarity_clean', 'p90_intra_similarity_clean', 'var_intra_similarity_clean',
        'topic_continuity_density', 'paraphrase_density_p90', 'strict_redundancy_density_p95', 'hyper_redundancy_density',
        'mexgen_persistence_gap', 'delta_mean_sim_morf',
        'conteo_palabras_text', 'num_sentences', 'score_ensamble_reglas'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X_cont = df[feature_cols].values
    max_sim_arr = df['max_intra_similarity_clean'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_meta = np.zeros(len(df))
    y_pred_meta = np.zeros(len(df), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cont, y_true), 1):
        # Discretización estrictamente In-Fold (Cuartiles aprendidos solo en train_idx + umbrales de dominio no supervisados)
        q_cuts = np.percentile(max_sim_arr[train_idx], [25, 50, 75])
        bins_q_tr = np.digitize(max_sim_arr[train_idx], q_cuts)
        bins_q_val = np.digitize(max_sim_arr[val_idx], q_cuts)

        ohe_q_tr = np.eye(4)[bins_q_tr]
        ohe_q_val = np.eye(4)[bins_q_val]

        dom_cuts = [0.3434, 0.5340, 0.5996, 0.7287]
        bins_d_tr = np.digitize(max_sim_arr[train_idx], dom_cuts)
        bins_d_val = np.digitize(max_sim_arr[val_idx], dom_cuts)

        ohe_d_tr = np.eye(5)[bins_d_tr]
        ohe_d_val = np.eye(5)[bins_d_val]

        X_train = np.hstack([X_cont[train_idx], ohe_q_tr, ohe_d_tr])
        X_val = np.hstack([X_cont[val_idx], ohe_q_val, ohe_d_val])
        y_train = y_true[train_idx]

        meta_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.06, max_depth=3, subsample=0.85, random_state=42 + fold)
        meta_clf.fit(X_train, y_train)
        probs_val = meta_clf.predict_proba(X_val)[:, 1]
        preds_val = (probs_val >= 0.50).astype(int)

        y_prob_meta[val_idx] = probs_val
        y_pred_meta[val_idx] = preds_val

    df['score_ensamble_calibrado'] = y_prob_meta
    df['pred_ensamble_calibrado'] = y_pred_meta

    acc_meta = accuracy_score(y_true, y_pred_meta)
    prec_meta = precision_score(y_true, y_pred_meta, zero_division=0)
    rec_meta = recall_score(y_true, y_pred_meta, zero_division=0)
    f1_meta = f1_score(y_true, y_pred_meta, zero_division=0)
    auc_meta = roc_auc_score(y_true, y_prob_meta)
    ap_meta = average_precision_score(y_true, y_prob_meta)
    cm_meta = confusion_matrix(y_true, y_pred_meta).tolist()

    print("\n" + "="*80)
    print(" 🏆 RESULTADOS DEL META-CLASIFICADOR CALIBRADO IN-FOLD (5-FOLD STRATIFIED CV)")
    print("="*80)
    print(f"Accuracy:  {acc_meta:.4f} ({acc_meta*100:.2f}%)")
    print(f"Precision: {prec_meta:.4f}")
    print(f"Recall:    {rec_meta:.4f}")
    print(f"F1-Score:  {f1_meta:.4f}")
    print(f"ROC-AUC:   {auc_meta:.4f}")
    print(f"PR-AUC:    {ap_meta:.4f}")
    print(f"Confusion Matrix: {cm_meta}")

    beto_narrativaai_metrics = {
        "Nombre": "BETO Fake News (Narrativaai)",
        "ID": "Narrativaai/fake-news-detection-spanish",
        "Accuracy": 0.6237,
        "Precision": 0.5852,
        "Recall": 0.7609,
        "F1-Score": 0.6616,
        "AUC": 0.6926,
        "AP": 0.6595,
        "CM": [[666, 679], [301, 958]]
    }

    print("\n" + "="*80)
    print(" ⚖️ TABLA COMPARATIVA: ENSAMBLE ADAPTATIVO PROPIO (BETO SBERT) VS. BETO NARRATIVAAI")
    print("="*80)
    print(f"{'Métrica':<15} | {'BETO Narrativaai':<18} | {'Ensamble Reglas':<18} | {'Ensamble Calibrado':<18}")
    print("-" * 75)
    print(f"{'Accuracy':<15} | {beto_narrativaai_metrics['Accuracy']:<18.4f} | {acc_rules:<18.4f} | {acc_meta:<18.4f}")
    print(f"{'Precision':<15} | {beto_narrativaai_metrics['Precision']:<18.4f} | {prec_rules:<18.4f} | {prec_meta:<18.4f}")
    print(f"{'Recall':<15} | {beto_narrativaai_metrics['Recall']:<18.4f} | {rec_rules:<18.4f} | {rec_meta:<18.4f}")
    print(f"{'F1-Score':<15} | {beto_narrativaai_metrics['F1-Score']:<18.4f} | {f1_rules:<18.4f} | {f1_meta:<18.4f}")
    print(f"{'ROC-AUC':<15} | {beto_narrativaai_metrics['AUC']:<18.4f} | {auc_rules:<18.4f} | {auc_meta:<18.4f}")
    print(f"{'PR-AUC (AP)':<15} | {beto_narrativaai_metrics['AP']:<18.4f} | {ap_rules:<18.4f} | {ap_meta:<18.4f}")

    out_preds_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/resultados_clasificacion_2604.csv"
    export_cols = [c for c in ['class', 'label_num', 'Text', 'Fuente', 'sens_mode', 'red_mode', 'diagnostico_estilometrico',
        'P_full', 'P_max', 'P_mean', 'delta_p_gatillo', 'max_intra_similarity_clean', 'second_max_intra_similarity_clean',
        'shannon_bin_4', 'shannon_bin_7', 'score_ensamble_reglas', 'score_ensamble_calibrado', 'pred_ensamble_calibrado'] if c in df.columns]
    df[export_cols].to_csv(out_preds_csv, index=False)
    print(f"\n💾 Predicciones detalladas guardadas en: {out_preds_csv}")

    out_json = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_ensamble_adaptativo.json"
    metrics_summary = {
        "metadata": {
            "dataset": "Noticias_entre_70_y_370_palabras (1).xlsx",
            "modelo_redundancia": "hiiamsid/sentence_similarity_spanish_es (BETO SBERT, d=768)",
            "total_muestras": len(df),
            "clase_reales_0": int((y_true == 0).sum()),
            "clase_falsas_1": int((y_true == 1).sum()),
            "esquema_validacion": "5-Fold Stratified Cross-Validation (Discretización Estricta In-Fold)",
            "fecha": "2026-10-20"
        },
        "BETO_Narrativaai": beto_narrativaai_metrics,
        "Ensamble_Reglas_Morfologicas": {
            "Accuracy": round(float(acc_rules), 4),
            "Precision": round(float(prec_rules), 4),
            "Recall": round(float(rec_rules), 4),
            "F1-Score": round(float(f1_rules), 4),
            "ROC-AUC": round(float(auc_rules), 4),
            "AP": round(float(ap_rules), 4),
            "CM": cm_rules
        },
        "Ensamble_Calibrado_CV": {
            "Accuracy": round(float(acc_meta), 4),
            "Precision": round(float(prec_meta), 4),
            "Recall": round(float(rec_meta), 4),
            "F1-Score": round(float(f1_meta), 4),
            "ROC-AUC": round(float(auc_meta), 4),
            "AP": round(float(ap_meta), 4),
            "CM": cm_meta
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
    print(f"💾 Métricas JSON guardadas en: {out_json}")

if __name__ == "__main__":
    run_adaptive_ensemble()
