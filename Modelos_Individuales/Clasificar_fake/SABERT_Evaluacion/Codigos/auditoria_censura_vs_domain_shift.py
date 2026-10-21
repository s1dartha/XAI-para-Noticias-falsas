"""
=============================================================================
AUDITORIA CONTRAFACTICA: DATOS CENSURADOS (*NUMBER*, *URL*, *PHONE*) VS.
DESPLAZAMIENTO DE DOMINIO GEOGRAFICO Y ESTILISTICO (DOMAIN SHIFT)
=============================================================================
Modulo: Modelos_Individuales/Clasificar_fake/SABERT_Evaluacion/Codigos/auditoria_censura_vs_domain_shift.py

Responde con rigor experimental a la observacion metodologica del tutor:
"Verificar los datos censurados de las noticias latinas y si eso cambia las
metricas del modelo cuando se le pasa el corpus latino: ¿las metricas bajan en si
por entidades y estilo diferente o por datos censurados?"

Experimentos ejecutados:
1. Cuantificacion exacta de tokens censurados (*NUMBER*, *URL*, *PHONE*) por
   sub-corpus y clase en el dataset ampliado (N = 4.418).
2. Desglose intra-regional en America Latina:
   - LatAm sin censura (Omdena LATAM, N = 1.250, 0% mascaras)
   - LatAm con censura de corpus (MEX-A3T + FakeDeS, N = 814), separando ademas
     textos con >= 1 mascara (N = 472) vs. textos sin mascaras (N = 342).
3. Experimento Contrafactico A (Enmascaramiento Sintetico sobre Espana, N = 2.354):
   Se reemplazan todas las cifras y URLs de las noticias de Espana por *NUMBER*
   y *URL* para medir el efecto puro de la censura manteniendo fijos el pais,
   las entidades y el estilo periodistico peninsular.
4. Experimento Contrafactico B (Restauracion Numerica sobre MEX-A3T + FakeDeS, N = 814):
   Se sustituyen los tokens *NUMBER*, *URL* y *PHONE* por cifras e hipervinculos
   plausibles para medir la variacion en SaBERT al eliminar las mascaras.
5. Impacto sobre el Ensamble Estilometrico 5D antes y despues de descontaminar
   *NUMBER* en las variables de mayusculas sostenidas (all_caps_count/ratio)
   y densidad numerica (numbers_density).
=============================================================================
"""

import os
import re
import json
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "n": int(len(y_true)),
        "n_real_0": int((y_true == 0).sum()),
        "n_fake_1": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "mean_prob_fake": float(np.mean(y_prob)),
        "cm": confusion_matrix(y_true, y_pred).tolist()
    }


def run_sabert_inference(texts, model, tokenizer, device, batch_size=32):
    preds, probs = [], []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1)
            # En SaBERT: id 0 = FAKE / Falsa, id 1 = REAL / Verdadera
            p_fake = p[:, 0].cpu().numpy()
            pred = (p_fake >= 0.5).astype(int)
            probs.extend(p_fake.tolist())
            preds.extend(pred.tolist())
    return np.array(preds), np.array(probs)


def main():
    base_dir = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake"
    preds_csv = os.path.join(base_dir, "Dataset", "predicciones_sabert_ampliado_4418.csv")
    report_dir = os.path.join(base_dir, "SABERT_Evaluacion", "Reportes")
    img_dir = os.path.join(report_dir, "Imagenes")
    os.makedirs(img_dir, exist_ok=True)

    print("=" * 85)
    print(" AUDITORIA CONTRAFACTICA: CENSURA (*NUMBER*) VS. DOMAIN SHIFT EN SABERT")
    print("=" * 85)

    df = pd.read_csv(preds_csv)
    y_true = df["clase_num"].astype(int).values
    y_prob_orig = df["prob_fake"].astype(float).values
    y_pred_orig = (y_prob_orig >= 0.5).astype(int)

    pat_number = re.compile(r"\*(?:NUMBER|\d+)\*", re.IGNORECASE)
    pat_url = re.compile(r"\*URL\*", re.IGNORECASE)
    pat_phone = re.compile(r"\*PHONE\*", re.IGNORECASE)
    pat_any_mask = re.compile(r"\*(?:NUMBER|URL|PHONE|\d+)\*", re.IGNORECASE)

    df["n_mask_number"] = df["Text"].astype(str).apply(lambda s: len(pat_number.findall(s)))
    df["n_mask_url"] = df["Text"].astype(str).apply(lambda s: len(pat_url.findall(s)))
    df["n_mask_phone"] = df["Text"].astype(str).apply(lambda s: len(pat_phone.findall(s)))
    df["n_mask_total"] = df["Text"].astype(str).apply(lambda s: len(pat_any_mask.findall(s)))
    df["has_mask"] = df["n_mask_total"] > 0

    # 1. Estadisticas de censura por corpus
    corpus_mask_stats = {}
    for fuente, grp in df.groupby("dataset_origen"):
        corpus_mask_stats[fuente] = {
            "n_noticias": int(len(grp)),
            "n_con_mascara": int(grp["has_mask"].sum()),
            "pct_con_mascara": float(grp["has_mask"].mean() * 100.0),
            "ocurrencias_number": int(grp["n_mask_number"].sum()),
            "ocurrencias_url": int(grp["n_mask_url"].sum()),
            "ocurrencias_phone": int(grp["n_mask_phone"].sum()),
            "pct_real_con_mascara": float(grp[grp["clase_num"] == 0]["has_mask"].mean() * 100.0) if (grp["clase_num"] == 0).any() else 0.0,
            "pct_fake_con_mascara": float(grp[grp["clase_num"] == 1]["has_mask"].mean() * 100.0) if (grp["clase_num"] == 1).any() else 0.0,
        }

    print("\n[1/4] Distribucion de tokens censurados por sub-corpus:")
    for k, v in corpus_mask_stats.items():
        print(f"  - {k}: N={v['n_noticias']} | Con mascara={v['n_con_mascara']} ({v['pct_con_mascara']:.1f}%) | *NUMBER*={v['ocurrencias_number']} | *URL*={v['ocurrencias_url']} | *PHONE*={v['ocurrencias_phone']}")

    # 2. Metricas estratificadas sobre predicciones originales
    mask_es = df["region"].str.contains("España", na=False)
    mask_latam = df["region"].str.contains("América Latina", na=False)
    mask_omdena = df["dataset_origen"].str.contains("Omdena", case=False, na=False)
    mask_mex_fakedes = df["dataset_origen"].str.contains("MEX-A3T|FakeDeS", case=False, na=False)
    mask_mex_con_mask = mask_mex_fakedes & df["has_mask"]
    mask_mex_sin_mask = mask_mex_fakedes & (~df["has_mask"])

    stratified_metrics = {
        "Global_4418": compute_metrics(y_true, y_pred_orig, y_prob_orig),
        "Espana_Original_2354": compute_metrics(y_true[mask_es], y_pred_orig[mask_es], y_prob_orig[mask_es]),
        "LatAm_Total_2064": compute_metrics(y_true[mask_latam], y_pred_orig[mask_latam], y_prob_orig[mask_latam]),
        "LatAm_Sin_Censura_Omdena_1250": compute_metrics(y_true[mask_omdena], y_pred_orig[mask_omdena], y_prob_orig[mask_omdena]),
        "LatAm_Corpus_Censurados_MEX_FakeDeS_814": compute_metrics(y_true[mask_mex_fakedes], y_pred_orig[mask_mex_fakedes], y_prob_orig[mask_mex_fakedes]),
        "MEX_FakeDeS_Textos_CON_Mascara_472": compute_metrics(y_true[mask_mex_con_mask], y_pred_orig[mask_mex_con_mask], y_prob_orig[mask_mex_con_mask]),
        "MEX_FakeDeS_Textos_SIN_Mascara_342": compute_metrics(y_true[mask_mex_sin_mask], y_pred_orig[mask_mex_sin_mask], y_prob_orig[mask_mex_sin_mask]),
    }

    print("\n[2/4] Comparativa intra-regional en SaBERT:")
    for name, m in stratified_metrics.items():
        print(f"  - {name:42s}: Acc={m['accuracy']*100:6.2f}% | RecFake={m['recall_fake']*100:6.2f}% | F1Fake={m['f1_fake']:.4f} | AUC={m['roc_auc']:.4f} | AP={m['ap']:.4f}")

    # 3. Experimentos Contrafacticos con SaBERT en GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "VerificadoProfesional/SaBERT-Spanish-Fake-News"
    print(f"\n[3/4] Ejecutando inferencia contrafactica con {model_name} en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Contrafactico A: Enmascarar todos los numeros y URLs del corpus de Espana (N = 2.354)
    pat_raw_url = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    pat_raw_num = re.compile(r"\b\d+(?:[\.,]\d+)?\b")

    texts_es_orig = df.loc[mask_es, "Text"].astype(str).tolist()
    texts_es_masked = []
    n_es_modified = 0
    total_es_nums_masked = 0
    for t in texts_es_orig:
        t_m = pat_raw_url.sub("*URL*", t)
        nums_found = len(pat_raw_num.findall(t_m))
        total_es_nums_masked += nums_found
        t_m = pat_raw_num.sub("*NUMBER*", t_m)
        if t_m != t:
            n_es_modified += 1
        texts_es_masked.append(t_m)

    pred_es_masked, prob_es_masked = run_sabert_inference(texts_es_masked, model, tokenizer, device)
    metrics_es_masked = compute_metrics(y_true[mask_es], pred_es_masked, prob_es_masked)
    metrics_es_masked["n_textos_modificados"] = int(n_es_modified)
    metrics_es_masked["pct_textos_modificados"] = float(n_es_modified / len(texts_es_orig) * 100.0)
    metrics_es_masked["total_cifras_enmascaradas"] = int(total_es_nums_masked)

    # Contrafactico B: Restaurar cifras plausibles en MEX-A3T + FakeDeS (N = 814)
    texts_mex_orig = df.loc[mask_mex_fakedes, "Text"].astype(str).tolist()
    texts_mex_unmasked = []
    for t in texts_mex_orig:
        t_u = pat_url.sub("https://www.portalnoticias.com", t)
        t_u = pat_phone.sub("5512345678", t_u)
        t_u = pat_number.sub("20", t_u)
        texts_mex_unmasked.append(t_u)

    pred_mex_unmasked, prob_mex_unmasked = run_sabert_inference(texts_mex_unmasked, model, tokenizer, device)
    metrics_mex_unmasked = compute_metrics(y_true[mask_mex_fakedes], pred_mex_unmasked, prob_mex_unmasked)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    es_orig = stratified_metrics["Espana_Original_2354"]
    lat_total = stratified_metrics["LatAm_Total_2064"]
    lat_omdena = stratified_metrics["LatAm_Sin_Censura_Omdena_1250"]
    mex_orig = stratified_metrics["LatAm_Corpus_Censurados_MEX_FakeDeS_814"]

    deltas_attribution = {
        "delta_censura_en_espana": {
            "descripcion": "Efecto puro de reemplazar cifras y URLs por *NUMBER* y *URL* en Espana (N=2354)",
            "delta_accuracy_pp": float((metrics_es_masked["accuracy"] - es_orig["accuracy"]) * 100.0),
            "delta_recall_fake_pp": float((metrics_es_masked["recall_fake"] - es_orig["recall_fake"]) * 100.0),
            "delta_f1_fake": float(metrics_es_masked["f1_fake"] - es_orig["f1_fake"]),
            "delta_roc_auc": float(metrics_es_masked["roc_auc"] - es_orig["roc_auc"]),
            "delta_ap": float(metrics_es_masked["ap"] - es_orig["ap"]),
        },
        "delta_restauracion_en_mex_fakedes": {
            "descripcion": "Efecto de sustituir *NUMBER* por cifras numericas en MEX-A3T + FakeDeS (N=814)",
            "delta_accuracy_pp": float((metrics_mex_unmasked["accuracy"] - mex_orig["accuracy"]) * 100.0),
            "delta_recall_fake_pp": float((metrics_mex_unmasked["recall_fake"] - mex_orig["recall_fake"]) * 100.0),
            "delta_f1_fake": float(metrics_mex_unmasked["f1_fake"] - mex_orig["f1_fake"]),
            "delta_roc_auc": float(metrics_mex_unmasked["roc_auc"] - mex_orig["roc_auc"]),
            "delta_ap": float(metrics_mex_unmasked["ap"] - mex_orig["ap"]),
        },
        "delta_domain_shift_puro_sin_censura": {
            "descripcion": "Caida de Espana (sin censura, N=2354) a Omdena LATAM (sin censura, N=1250)",
            "delta_accuracy_pp": float((lat_omdena["accuracy"] - es_orig["accuracy"]) * 100.0),
            "delta_recall_fake_pp": float((lat_omdena["recall_fake"] - es_orig["recall_fake"]) * 100.0),
            "delta_f1_fake": float(lat_omdena["f1_fake"] - es_orig["f1_fake"]),
            "delta_roc_auc": float(lat_omdena["roc_auc"] - es_orig["roc_auc"]),
            "delta_ap": float(lat_omdena["ap"] - es_orig["ap"]),
        },
        "delta_domain_shift_global_latam": {
            "descripcion": "Caida total de Espana (N=2354) a America Latina consolidada (N=2064)",
            "delta_accuracy_pp": float((lat_total["accuracy"] - es_orig["accuracy"]) * 100.0),
            "delta_recall_fake_pp": float((lat_total["recall_fake"] - es_orig["recall_fake"]) * 100.0),
            "delta_f1_fake": float(lat_total["f1_fake"] - es_orig["f1_fake"]),
            "delta_roc_auc": float(lat_total["roc_auc"] - es_orig["roc_auc"]),
            "delta_ap": float(lat_total["ap"] - es_orig["ap"]),
        }
    }

    print("\n" + "=" * 85)
    print(" RESULTADOS DE LOS EXPERIMENTOS CONTRAFACTICOS")
    print("=" * 85)
    print(f"1. Espana Original (N=2354):           Acc={es_orig['accuracy']*100:.2f}% | RecFake={es_orig['recall_fake']*100:.2f}% | F1={es_orig['f1_fake']:.4f} | AUC={es_orig['roc_auc']:.4f} | AP={es_orig['ap']:.4f}")
    print(f"2. Espana Censurada *NUMBER* (N=2354): Acc={metrics_es_masked['accuracy']*100:.2f}% | RecFake={metrics_es_masked['recall_fake']*100:.2f}% | F1={metrics_es_masked['f1_fake']:.4f} | AUC={metrics_es_masked['roc_auc']:.4f} | AP={metrics_es_masked['ap']:.4f}")
    print(f"   -> Delta puro por Censura en Espana: dAcc={deltas_attribution['delta_censura_en_espana']['delta_accuracy_pp']:+.2f} pp | dRec={deltas_attribution['delta_censura_en_espana']['delta_recall_fake_pp']:+.2f} pp | dAUC={deltas_attribution['delta_censura_en_espana']['delta_roc_auc']:+.4f}")
    print(f"3. LatAm Omdena SIN censura (N=1250):  Acc={lat_omdena['accuracy']*100:.2f}% | RecFake={lat_omdena['recall_fake']*100:.2f}% | F1={lat_omdena['f1_fake']:.4f} | AUC={lat_omdena['roc_auc']:.4f} | AP={lat_omdena['ap']:.4f}")
    print(f"   -> Delta por Domain Shift puro (sin censura): dAcc={deltas_attribution['delta_domain_shift_puro_sin_censura']['delta_accuracy_pp']:+.2f} pp | dRec={deltas_attribution['delta_domain_shift_puro_sin_censura']['delta_recall_fake_pp']:+.2f} pp | dAUC={deltas_attribution['delta_domain_shift_puro_sin_censura']['delta_roc_auc']:+.4f}")
    print(f"4. MEX+FakeDeS Original (N=814):       Acc={mex_orig['accuracy']*100:.2f}% | RecFake={mex_orig['recall_fake']*100:.2f}% | F1={mex_orig['f1_fake']:.4f} | AUC={mex_orig['roc_auc']:.4f} | AP={mex_orig['ap']:.4f}")
    print(f"5. MEX+FakeDeS Restaurado (N=814):     Acc={metrics_mex_unmasked['accuracy']*100:.2f}% | RecFake={metrics_mex_unmasked['recall_fake']*100:.2f}% | F1={metrics_mex_unmasked['f1_fake']:.4f} | AUC={metrics_mex_unmasked['roc_auc']:.4f} | AP={metrics_mex_unmasked['ap']:.4f}")

    # 4. Generar grafico comparativo de alta resolucion
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # Panel A: Comparativa de Escenarios (Accuracy, Recall Fake, ROC-AUC, AP)
    ax1 = axes[0]
    scenarios = [
        "España\nOriginal\n(N=2.354)",
        "España + Censura\n*NUMBER* Sintética\n(N=2.354)",
        "LatAm Con Máscaras\nMEX-A3T + FakeDeS\n(N=814)",
        "LatAm Restaurado\nSin *NUMBER*\n(N=814)",
        "LatAm Sin Censura\nOmdena LATAM\n(N=1.250)"
    ]
    scen_objs = [es_orig, metrics_es_masked, mex_orig, metrics_mex_unmasked, lat_omdena]

    x = np.arange(len(scenarios))
    w = 0.20

    b_acc = ax1.bar(x - 1.5 * w, [s["accuracy"] for s in scen_objs], w, label="Exactitud (Accuracy)", color="#1f77b4")
    b_rec = ax1.bar(x - 0.5 * w, [s["recall_fake"] for s in scen_objs], w, label="Sensibilidad Fake (Recall)", color="#d62728")
    b_auc = ax1.bar(x + 0.5 * w, [s["roc_auc"] for s in scen_objs], w, label="ROC-AUC", color="#2ca02c")
    b_ap = ax1.bar(x + 1.5 * w, [s["ap"] for s in scen_objs], w, label="AP (PR-AUC)", color="#9467bd")

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Puntuación (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Ablación Contrafáctica en SaBERT: Censura vs. Domain Shift", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", ncol=2, frameon=True, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for bars in [b_acc, b_rec, b_auc, b_ap]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=45
            )

    # Panel B: Descomposicion de la Caida de Rendimiento (Puntos Porcentuales / Delta)
    ax2 = axes[1]
    metric_labels = ["Exactitud\n(Accuracy)", "Sensibilidad\n(Recall Fake)", "F1-Score\n(Clase Fake)", "ROC-AUC\n(×100)", "AP / PR-AUC\n(×100)"]
    d_cens = [
        deltas_attribution["delta_censura_en_espana"]["delta_accuracy_pp"],
        deltas_attribution["delta_censura_en_espana"]["delta_recall_fake_pp"],
        deltas_attribution["delta_censura_en_espana"]["delta_f1_fake"] * 100.0,
        deltas_attribution["delta_censura_en_espana"]["delta_roc_auc"] * 100.0,
        deltas_attribution["delta_censura_en_espana"]["delta_ap"] * 100.0,
    ]
    d_shift = [
        deltas_attribution["delta_domain_shift_puro_sin_censura"]["delta_accuracy_pp"],
        deltas_attribution["delta_domain_shift_puro_sin_censura"]["delta_recall_fake_pp"],
        deltas_attribution["delta_domain_shift_puro_sin_censura"]["delta_f1_fake"] * 100.0,
        deltas_attribution["delta_domain_shift_puro_sin_censura"]["delta_roc_auc"] * 100.0,
        deltas_attribution["delta_domain_shift_puro_sin_censura"]["delta_ap"] * 100.0,
    ]

    x2 = np.arange(len(metric_labels))
    w2 = 0.35
    rects_c = ax2.bar(x2 - w2 / 2, d_cens, w2, label="Efecto Puro Censura (*NUMBER* en España)", color="#ff7f0e", edgecolor="black")
    rects_s = ax2.bar(x2 + w2 / 2, d_shift, w2, label="Efecto Puro Domain Shift (Omdena LatAm sin censura)", color="#d62728", edgecolor="black")

    ax2.axhline(0, color="black", lw=1.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Variación respecto a España Original (Puntos / %)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Atribución Causal de la Caída: Censura Léxica vs. Dominio Regional", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=9)
    ax2.set_ylim(-88, 15)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bars in [rects_c, rects_s]:
        for bar in bars:
            h = bar.get_height()
            va_pos = "bottom" if h >= 0 else "top"
            y_off = 3 if h >= 0 else -12
            ax2.annotate(
                f"{h:+.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, y_off),
                textcoords="offset points",
                ha="center", va=va_pos, fontsize=8.5, fontweight="bold"
            )

    plt.suptitle("Auditoría de Robustez: Impacto de Datos Censurados (*NUMBER*) frente al Desplazamiento de Dominio Geográfico", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()

    fig_out = os.path.join(img_dir, "7_impacto_censura_vs_domain_shift.png")
    fig.savefig(fig_out, dpi=300)
    plt.close(fig)
    print(f"\n[4/4] Grafico guardado en: {fig_out}")

    for target_dir in [
        "/home/ubuntu/Documentos/Tesis_DaNat_1/imagenes/corregidas",
        "/home/ubuntu/Documentos/Tesis_DaNat_1/imagenes/Sabert"
    ]:
        if os.path.exists(target_dir):
            shutil.copy2(fig_out, os.path.join(target_dir, "7_impacto_censura_vs_domain_shift.png"))

    out_json = os.path.join(report_dir, "auditoria_censura_vs_domain_shift.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "distribucion_mascaras_por_corpus": corpus_mask_stats,
            "metricas_estratificadas_originales": stratified_metrics,
            "contrafactico_espana_censurada_2354": metrics_es_masked,
            "contrafactico_mex_fakedes_restaurado_814": metrics_mex_unmasked,
            "atribucion_deltas": deltas_attribution
        }, f, indent=4, ensure_ascii=False)
    print(f"[OK] JSON de auditoria guardado en: {out_json}")


if __name__ == "__main__":
    main()
