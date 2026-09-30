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

    print(f"⏳ Cargando características morfológicas desde: {features_csv}")
    df = pd.read_csv(features_csv)
    print(f"✅ Registros cargados: {len(df)}")

    # 1. Definir la Función de Enrutamiento y Gating Morfológico (Modelo Basado en Reglas Expertas)
    def compute_morphological_gating(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        p_top2 = row['P_top2']
        sigma = row['sigma_sens']
        dr = row['dilution_ratio']
        delta_p = row['delta_p_gatillo']
        
        max_sim = row['max_intra_similarity_clean']
        mean_sim = row['mean_intra_similarity_clean']
        red_density = row['redundancy_density']
        shannon_bin = row['shannon_bin_7']
        
        # --- PERFIL SENSACIONALISMO ---
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

        # --- PERFIL REDUNDANCIA ---
        if max_sim > 0.8077 and red_density >= 0.15:
            red_mode = "R_BUC"   # Hiper-Redundancia Circular en Bucle
        elif max_sim > 0.8077 and red_density < 0.15:
            red_mode = "R_PUN"   # Redundancia Puntual
        elif 0.6177 < max_sim <= 0.8077:
            red_mode = "R_EST"   # Cohesión Profesional Estándar
        elif max_sim <= 0.5924:
            red_mode = "R_DES"   # Desarticulado / Baja Cohesión
        else:
            red_mode = "R_MED"   # Transición

        # --- MATRIZ DE DECISIÓN Y SCORE DE MANIPULACIÓN ---
        # Tasa empírica por bin de Shannon como a priori de riesgo
        shannon_priors = {
            1: 0.511, # Bin 1: Desarticulado
            2: 0.642, # Bin 2: Transición alta sospecha
            3: 0.353, # Bin 3: Cohesión óptima (mínima tasa fake)
            4: 0.495, # Bin 4: Periodismo estándar
            5: 0.673, # Bin 5: Redundancia elevada
            6: 0.792, # Bin 6: Hiper-redundancia extrema
            7: 0.533  # Bin 7: Casi-duplicación
        }
        p_red_risk = shannon_priors.get(shannon_bin, 0.50)

        # Regla 1: Saturación + Hiper-Redundancia (Desinformación Masiva / IA)
        if sens_mode == "S_DIST" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.50 * p_full + 0.50 * p_red_risk
            decision_note = "Desinformación Saturada en Bucle"

        # Regla 2: Saturación + Cohesión Estándar (Propaganda Formal)
        elif sens_mode == "S_DIST" and red_mode == "R_EST":
            score = 0.70 * p_full + 0.30 * p_red_risk
            decision_note = "Sensacionalismo Distribuido Coherente"

        # Regla 3: Efecto Gatillo + Hiper-Redundancia (Clickbait Tóxico)
        elif sens_mode == "S_GAT" and red_mode in ["R_BUC", "R_PUN"]:
            score = 0.60 * p_full + 0.40 * p_red_risk
            decision_note = "Clickbait con Bucle Argumental"

        # Regla 4: Efecto Gatillo + Cohesión Estándar (Clickbait Comercial Inocuo)
        elif sens_mode == "S_GAT" and red_mode == "R_EST":
            # Ablación virtual: descontamos el impacto del gatillo
            p_discounted = max(0.10, p_full - delta_p)
            score = 0.60 * p_discounted + 0.40 * p_red_risk
            decision_note = "Clickbait Comercial Amortiguado"

        # Regla 5: Dilución Contextual (Periodismo Serio de Impacto - Caso AMA_11)
        elif sens_mode == "S_DIL" and red_mode in ["R_EST", "R_MED"]:
            # El cuerpo amortigua la alarma
            score = min(p_full, 0.25)
            decision_note = "Periodismo Serio con Dilución Contextual"

        # Regla 6: Sobriedad + Hiper-Redundancia (Desinformación Sofisticada / Astroturfing)
        elif sens_mode == "S_SOB" and red_mode in ["R_BUC", "R_PUN"]:
            # Redundancia toma la rectoría de la sospecha
            score = 0.20 * p_full + 0.80 * p_red_risk
            decision_note = "Desinformación Sofisticada Encubierta"

        # Regla 7: Sobriedad + Cohesión Estándar (Noticia Neutral / Confiable)
        elif sens_mode == "S_SOB" and red_mode == "R_EST":
            score = 0.50 * p_full + 0.50 * p_red_risk
            decision_note = "Texto Neutro y Cohesión Profesional"

        # Regla 8: Caso General Mixto
        else:
            score = 0.45 * p_full + 0.55 * p_red_risk
            decision_note = "Patrón Morfológico Mixto"

        return sens_mode, red_mode, float(np.clip(score, 0.0, 1.0)), decision_note

    print("⏳ Aplicando motor de inferencia morfológico adaptativo a todas las noticias...")
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
    print(" 📊 RESULTADOS DEL ENSAMBLE BASADO EN REGLAS MORFOLÓGICAS ADAPTATIVAS")
    print("="*80)
    print(f"Accuracy:  {acc_rules:.4f} ({acc_rules*100:.2f}%)")
    print(f"Precision: {prec_rules:.4f}")
    print(f"Recall:    {rec_rules:.4f}")
    print(f"F1-Score:  {f1_rules:.4f}")
    print(f"ROC-AUC:   {auc_rules:.4f}")
    print(f"PR-AUC:    {ap_rules:.4f}")
    print(f"Confusion Matrix: {cm_rules}")

    # 2. Meta-Clasificador Supervisado Calibrado (Validación Cruzada Estratificada 5 Folds)
    feature_cols = [
        'P_full', 'P_mean', 'P_max', 'P_top2', 'sigma_sens', 'dilution_ratio', 'delta_p_gatillo',
        'max_intra_similarity_clean', 'mean_intra_similarity_clean', 'redundancy_density',
        'conteo_palabras_text', 'num_sentences', 'score_ensamble_reglas'
    ]
    # Añadir One-Hot de Bins de Shannon
    bin_ohe = pd.get_dummies(df['shannon_bin_7'], prefix='shannon_bin').astype(float)
    X = pd.concat([df[feature_cols], bin_ohe], axis=1).values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_meta = np.zeros(len(df))
    y_pred_meta = np.zeros(len(df), dtype=int)

    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    # meta_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true), 1):
        X_train, y_train = X[train_idx], y_true[train_idx]
        X_val, y_val = X[val_idx], y_true[val_idx]
        
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
    print(" 🏆 RESULTADOS DEL META-CLASIFICADOR CALIBRADO (5-FOLD STRATIFIED CV)")
    print("="*80)
    print(f"Accuracy:  {acc_meta:.4f} ({acc_meta*100:.2f}%)")
    print(f"Precision: {prec_meta:.4f}")
    print(f"Recall:    {rec_meta:.4f}")
    print(f"F1-Score:  {f1_meta:.4f}")
    print(f"ROC-AUC:   {auc_meta:.4f}")
    print(f"PR-AUC:    {ap_meta:.4f}")
    print(f"Confusion Matrix: {cm_meta}")

    # Datos oficiales de BETO Narrativaai para comparación
    beto_narrativaai_metrics = {
        "Nombre": "BETO Fake News (Narrativaai)",
        "ID": "Narrativaai/fake-news-detection-spanish",
        "Accuracy": 0.3763,
        "Precision": 0.3630,
        "Recall": 0.3720,
        "F1-Score": 0.3629,
        "AUC": 0.3074,
        "AP": 0.3812,
        "CM": [[301, 958], [666, 679]]
    }

    print("\n" + "="*80)
    print(" ⚖️ TABLA COMPARATIVA: ENSAMBLE ADAPTATIVO PROPIO VS. BETO NARRATIVAAI")
    print("="*80)
    print(f"{'Métrica':<15} | {'BETO Narrativaai':<18} | {'Ensamble Reglas':<18} | {'Ensamble Calibrado':<18}")
    print("-" * 75)
    print(f"{'Accuracy':<15} | {beto_narrativaai_metrics['Accuracy']:<18.4f} | {acc_rules:<18.4f} | {acc_meta:<18.4f}")
    print(f"{'Precision':<15} | {beto_narrativaai_metrics['Precision']:<18.4f} | {prec_rules:<18.4f} | {prec_meta:<18.4f}")
    print(f"{'Recall':<15} | {beto_narrativaai_metrics['Recall']:<18.4f} | {rec_rules:<18.4f} | {rec_meta:<18.4f}")
    print(f"{'F1-Score':<15} | {beto_narrativaai_metrics['F1-Score']:<18.4f} | {f1_rules:<18.4f} | {f1_meta:<18.4f}")
    print(f"{'ROC-AUC':<15} | {beto_narrativaai_metrics['AUC']:<18.4f} | {auc_rules:<18.4f} | {auc_meta:<18.4f}")
    print(f"{'PR-AUC (AP)':<15} | {beto_narrativaai_metrics['AP']:<18.4f} | {ap_rules:<18.4f} | {ap_meta:<18.4f}")

    # Guardar CSV con predicciones completas
    out_preds_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/resultados_clasificacion_2604.csv"
    df[['class', 'label_num', 'Text', 'Fuente', 'sens_mode', 'red_mode', 'diagnostico_estilometrico', 
        'P_full', 'P_max', 'P_mean', 'delta_p_gatillo', 'max_intra_similarity_clean', 'shannon_bin_7',
        'score_ensamble_reglas', 'score_ensamble_calibrado', 'pred_ensamble_calibrado']].to_csv(out_preds_csv, index=False)
    print(f"\n💾 Predicciones detalladas guardadas en: {out_preds_csv}")

    # Guardar JSON de métricas
    out_json = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_ensamble_adaptativo.json"
    metrics_summary = {
        "metadata": {
            "dataset": "Noticias_entre_70_y_370_palabras (1).xlsx",
            "total_muestras": len(df),
            "clase_reales_0": int((y_true == 0).sum()),
            "clase_falsas_1": int((y_true == 1).sum()),
            "esquema_validacion": "5-Fold Stratified Cross-Validation",
            "fecha": "2026-09-27"
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
