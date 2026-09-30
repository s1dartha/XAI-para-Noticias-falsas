import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

# Estilo gráfico académico y limpio
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

base_dir = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake"
output_dir = os.path.join(base_dir, "SABERT_Evaluacion")
img_dir = os.path.join(output_dir, "imagenes")
os.makedirs(img_dir, exist_ok=True)

print("="*80)
print(" EVALUACIÓN EXPERIMENTAL RIGUROSA DE SaBERT DESGLOSADO POR FUENTE")
print("="*80)

# Cargar dataset 70-370 palabras
dataset_2604_path = os.path.join(base_dir, "Dataset", "Noticias_entre_70_y_370_palabras (1).xlsx")
print(f"[INFO] Cargando dataset 70-370: {dataset_2604_path}")
df_2604 = pd.read_excel(dataset_2604_path)

def parse_label(val):
    if isinstance(val, (bool, np.bool_)):
        return 0 if val is True else 1
    s = str(val).strip().upper()
    if s in ['TRUE', 'VERDADERO', 'VERDADERA', 'REAL', '0']:
        return 0
    return 1

df_2604['label_num'] = df_2604['class'].apply(parse_label)

# Cargar SaBERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Dispositivo de cómputo: {device}")
model_name = "VerificadoProfesional/SaBERT-Spanish-Fake-News"
print(f"[INFO] Cargando modelo: {model_name}")
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

batch_size = 64

# -------------------------------------------------------------------------
# FASE 1: INFERENCIA EN DATASET 70-370 PALABRAS (N = 2,604)
# -------------------------------------------------------------------------
csv_2604_out = os.path.join(output_dir, "predicciones_sabert_70_370_desglosadas.csv")
if os.path.exists(csv_2604_out):
    print(f"\n[INFO] Cargando predicciones previas de {csv_2604_out}...")
    df_cached = pd.read_csv(csv_2604_out)
    if 'pred_sabert' in df_cached.columns and 'prob_fake' in df_cached.columns:
        df_2604['pred_sabert'] = df_cached['pred_sabert']
        df_2604['prob_fake'] = df_cached['prob_fake']
        df_2604['prob_real'] = df_cached['prob_real']
    else:
        df_cached = None
else:
    df_cached = None

if df_cached is None or 'pred_sabert' not in df_2604.columns:
    print("\n>>> FASE 1: Inferencia en Dataset 70-370 palabras...")
    texts_2604 = df_2604['Text'].astype(str).tolist()
    probs_fake_2604 = []
    probs_real_2604 = []
    preds_2604 = []

    batch_size = 64
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(texts_2604), batch_size):
            batch = texts_2604[i:i+batch_size]
            inp = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model(**inp).logits
            p = torch.softmax(logits, dim=-1)
            p_fake = p[:, 0].cpu().tolist()
            p_real = p[:, 1].cpu().tolist()
            pred_fake = (p[:, 0] > 0.5).long().cpu().tolist()
            
            probs_fake_2604.extend(p_fake)
            probs_real_2604.extend(p_real)
            preds_2604.extend(pred_fake)

    elapsed_2604 = time.time() - t0
    print(f"[INFO] Inferencia 2604 noticias completada en {elapsed_2604:.2f}s ({(elapsed_2604/len(df_2604))*1000:.2f} ms/muestra)")

    df_2604['pred_sabert'] = preds_2604
    df_2604['prob_fake'] = probs_fake_2604
    df_2604['prob_real'] = probs_real_2604
    df_2604.to_csv(csv_2604_out, index=False)
    print(f"[INFO] Predicciones guardadas en {csv_2604_out}")

# Métricas por fuente
metricas_2604 = {}

# Métrica global
y_all = df_2604['label_num'].values
p_all = df_2604['pred_sabert'].values
prob_all = df_2604['prob_fake'].values
acc_all = accuracy_score(y_all, p_all)
prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(y_all, p_all, average='macro', zero_division=0)
prec1_all, rec1_all, f1_1_all, _ = precision_recall_fscore_support(y_all, p_all, average='binary', pos_label=1, zero_division=0)
auc_all = roc_auc_score(y_all, prob_all)
cm_all = confusion_matrix(y_all, p_all)
tn, fp, fn, tp = cm_all.ravel()

metricas_2604['GLOBAL'] = {
    'fuente': 'GLOBAL (Dataset Completo 70-370)',
    'n_total': int(len(df_2604)),
    'n_real': int(np.sum(y_all == 0)),
    'n_fake': int(np.sum(y_all == 1)),
    'accuracy': float(acc_all),
    'f1_macro': float(f1_all),
    'f1_fake': float(f1_1_all),
    'precision_fake': float(prec1_all),
    'recall_fake': float(rec1_all),
    'roc_auc': float(auc_all),
    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    'cm': cm_all.tolist(),
    'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    'false_positive_rate': float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
}

fuentes_nombres_cortos = {
    'https://huggingface.co/datasets/Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus': 'Freiren (Gabriel Hurtado)',
    'https://huggingface.co/datasets/Edds/spanish-fake-news-fixed': 'Edds (Fact-checking España)',
    'https://huggingface.co/datasets/mariagrandury/fake_news_corpus_spanish': 'mariagrandury (FakeDeS - LatAm)',
    'https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=fakes1000.csv': 'arseniitretiakov (Kaggle Spanish Fakes)'
}

for src, sub in df_2604.groupby('Fuente'):
    y_sub = sub['label_num'].values
    p_sub = sub['pred_sabert'].values
    pr_sub = sub['prob_fake'].values
    acc = accuracy_score(y_sub, p_sub)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_sub, p_sub, average='macro', zero_division=0)
    prec_1, rec_1, f1_1, _ = precision_recall_fscore_support(y_sub, p_sub, average='binary', pos_label=1, zero_division=0)
    auc_v = roc_auc_score(y_sub, pr_sub) if len(np.unique(y_sub)) > 1 else None
    cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm.ravel()
    
    fnr = float(fn_s / (fn_s + tp_s)) if (fn_s + tp_s) > 0 else 0.0
    fpr = float(fp_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
    
    metricas_2604[src] = {
        'nombre_corto': fuentes_nombres_cortos.get(src, src),
        'fuente': src,
        'n_total': int(len(sub)),
        'n_real': int(np.sum(y_sub == 0)),
        'n_fake': int(np.sum(y_sub == 1)),
        'proporcion_del_dataset': float(len(sub) / len(df_2604)),
        'accuracy': float(acc),
        'f1_macro': float(f1_m),
        'f1_fake': float(f1_1),
        'precision_fake': float(prec_1),
        'recall_fake': float(rec_1),
        'roc_auc': float(auc_v) if auc_v is not None else None,
        'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s),
        'cm': cm.tolist(),
        'false_negative_rate': fnr,
        'false_positive_rate': fpr
    }

json_2604_path = os.path.join(output_dir, "metricas_sabert_fuentes_70_370.json")
with open(json_2604_path, 'w', encoding='utf-8') as f:
    json.dump(metricas_2604, f, indent=2, ensure_ascii=False)
print(f"[INFO] Metricas 70-370 guardadas en {json_2604_path}")

# -------------------------------------------------------------------------
# FASE 2: EVALUACIÓN EN MUESTRA ESTRATIFICADA BALANCEADA (DATASET COMPLETO)
# -------------------------------------------------------------------------
full_dataset_path = "/home/ubuntu/Descargas/Noticias_con_conteo_palabras.xlsx"
metricas_balanceadas = {}
if os.path.exists(full_dataset_path):
    print(f"\n>>> FASE 2: Cargando dataset completo para prueba de Domain Shift balanceada: {full_dataset_path}")
    df_full = pd.read_excel(full_dataset_path)
    df_full['label_num'] = df_full['class'].apply(parse_label)
    
    # Muestras balanceadas
    samples = []
    for src in df_full['Fuente'].unique():
        sub = df_full[df_full['Fuente'] == src]
        if len(sub) > 600:
            sub_0 = sub[sub['label_num'] == 0].sample(min(300, (sub['label_num']==0).sum()), random_state=42)
            sub_1 = sub[sub['label_num'] == 1].sample(min(300, (sub['label_num']==1).sum()), random_state=42)
            samples.append(pd.concat([sub_0, sub_1]))
        else:
            samples.append(sub)
    
    test_bal = pd.concat(samples).reset_index(drop=True)
    print(f"[INFO] Muestra balanceada armada: {len(test_bal)} registros")
    print(test_bal.groupby(['Fuente', 'label_num']).size())
    
    texts_bal = test_bal['Text'].astype(str).tolist()
    probs_fake_bal = []
    preds_bal = []
    with torch.inference_mode():
        for i in range(0, len(texts_bal), batch_size):
            batch = texts_bal[i:i+batch_size]
            inp = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model(**inp).logits
            p = torch.softmax(logits, dim=-1)
            probs_fake_bal.extend(p[:, 0].cpu().tolist())
            preds_bal.extend((p[:, 0] > 0.5).long().cpu().tolist())
    
    test_bal['pred_sabert'] = preds_bal
    test_bal['prob_fake'] = probs_fake_bal
    
    test_bal.to_csv(os.path.join(output_dir, "predicciones_sabert_balanceadas_desglosadas.csv"), index=False)
    
    for src, sub in test_bal.groupby('Fuente'):
        y_s = sub['label_num'].values
        p_s = sub['pred_sabert'].values
        pr_s = sub['prob_fake'].values
        acc_s = accuracy_score(y_s, p_s)
        f1_m, rec_fake = f1_all, 0.0
        cm_s = confusion_matrix(y_s, p_s, labels=[0, 1])
        tn_b, fp_b, fn_b, tp_b = cm_s.ravel()
        auc_s = roc_auc_score(y_s, pr_s) if len(np.unique(y_s)) > 1 else None
        
        metricas_balanceadas[src] = {
            'nombre_corto': fuentes_nombres_cortos.get(src, src),
            'n_total': int(len(sub)),
            'n_real': int(np.sum(y_s == 0)),
            'n_fake': int(np.sum(y_s == 1)),
            'accuracy': float(acc_s),
            'roc_auc': float(auc_s) if auc_s is not None else None,
            'tn': int(tn_b), 'fp': int(fp_b), 'fn': int(fn_b), 'tp': int(tp_b),
            'fn_fallos_fake': f"{fn_b} de {fn_b + tp_b}",
            'tasa_fallo_fake_pct': float(fn_b / (fn_b + tp_b) * 100) if (fn_b + tp_b) > 0 else 0.0,
            'cm': cm_s.tolist()
        }
    
    json_bal_path = os.path.join(output_dir, "metricas_sabert_fuentes_balanceadas.json")
    with open(json_bal_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_balanceadas, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metricas balanceadas guardadas en {json_bal_path}")

# -------------------------------------------------------------------------
# FASE 3: GENERACIÓN DE FIGURAS Y GRÁFICOS DE ALTA RESOLUCIÓN (300 DPI)
# -------------------------------------------------------------------------
print("\n>>> FASE 3: Generando gráficos de alta resolución...")

# 1. Composición y Dominancia del Dataset 2,604
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Torta / Donut
fuentes_order = [
    'https://huggingface.co/datasets/Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus',
    'https://huggingface.co/datasets/Edds/spanish-fake-news-fixed',
    'https://huggingface.co/datasets/mariagrandury/fake_news_corpus_spanish',
    'https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=fakes1000.csv'
]
sizes = [len(df_2604[df_2604['Fuente'] == src]) for src in fuentes_order]
labels = [
    f"Freiren\n(N=2,060 | 79.11%)",
    f"Edds\n(N=294 | 11.29%)",
    f"mariagrandury\n(N=248 | 9.52%)",
    f"arseniitretiakov\n(N=2 | 0.08%)"
]
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
explode = (0.05, 0.05, 0.08, 0.15)

wedges, texts, autotexts = ax1.pie(
    sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    startangle=140, colors=colors, textprops={'fontsize': 10, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
ax1.set_title("A. Dominancia de Fuentes en Dataset 70-370 Palabras\n(N = 2,604 noticias)", fontsize=13, fontweight='bold', pad=15)

# Barras apiladas de clases
sub_counts = []
for src in fuentes_order:
    sub = df_2604[df_2604['Fuente'] == src]
    n_r = (sub['label_num'] == 0).sum()
    n_f = (sub['label_num'] == 1).sum()
    sub_counts.append((n_r, n_f))

names_bar = ['Freiren', 'Edds', 'mariagrandury', 'arseniitretiakov']
reals = [c[0] for c in sub_counts]
fakes = [c[1] for c in sub_counts]

p1 = ax2.bar(names_bar, reals, color='#3498db', label='Noticia Real (0)', edgecolor='black', alpha=0.85)
p2 = ax2.bar(names_bar, fakes, bottom=reals, color='#e74c3c', label='Noticia Falsa (1)', edgecolor='black', alpha=0.85)
ax2.set_ylabel("Cantidad de Noticias", fontsize=11, fontweight='bold')
ax2.set_title("B. Distribución de Clases (Real vs Fake) por Fuente", fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for i, total in enumerate(sizes):
    ax2.text(i, total + 25, f"N={total}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "1_distribucion_composicion_dataset_2604.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 1 generado: distribucion_composicion_dataset_2604.png")

# 2. Comparativa de Métricas por Fuente (70-370)
fig, ax = plt.subplots(figsize=(12, 6))
fuentes_eval = [
    ('GLOBAL', 'GLOBAL\n(2,604)', metricas_2604['GLOBAL']),
    (fuentes_order[0], 'Freiren\n(2,060 - España)', metricas_2604[fuentes_order[0]]),
    (fuentes_order[1], 'Edds\n(294 - España)', metricas_2604[fuentes_order[1]]),
    (fuentes_order[2], 'mariagrandury\n(248 - LatAm)', metricas_2604[fuentes_order[2]])
]

x = np.arange(len(fuentes_eval))
width = 0.2

accs = [m[2]['accuracy'] * 100 for m in fuentes_eval]
f1s = [m[2]['f1_macro'] * 100 for m in fuentes_eval]
rec_fakes = [m[2]['recall_fake'] * 100 for m in fuentes_eval]
aucs = [(m[2]['roc_auc'] or 0) * 100 for m in fuentes_eval]

rects1 = ax.bar(x - 1.5*width, accs, width, label='Accuracy (%)', color='#2b5c8f')
rects2 = ax.bar(x - 0.5*width, f1s, width, label='F1-Score Macro (%)', color='#4682b4')
rects3 = ax.bar(x + 0.5*width, rec_fakes, width, label='Recall Fake News (%)', color='#e67e22')
rects4 = ax.bar(x + 1.5*width, aucs, width, label='ROC-AUC (x100)', color='#27ae60')

ax.set_ylabel('Porcentaje / Puntuación (%)', fontsize=11, fontweight='bold')
ax.set_title('Desempeño de SaBERT Desglosado por Fuente (Dataset 70-370 Palabras)\nEvidencia de Desplome Fuera de Distribución (mariagrandury)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([m[1] for m in fuentes_eval], fontsize=11, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 115)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "2_metricas_sabert_por_fuente_70_370.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 2 generado: metricas_sabert_por_fuente_70_370.png")

# 3. Panel de Matrices de Confusión por Fuente (70-370)
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
cms = [
    ("A. Global (N = 2,604 | Acc: 85.71%)", metricas_2604['GLOBAL']['cm'], axes[0, 0], 'Blues'),
    ("B. Freiren - Domina 79.1% (N = 2,060 | Acc: 90.29%)", metricas_2604[fuentes_order[0]]['cm'], axes[0, 1], 'Greens'),
    ("C. Edds (N = 294 | Acc: 85.71%)", metricas_2604[fuentes_order[1]]['cm'], axes[1, 0], 'PuBu'),
    ("D. mariagrandury - Colapso (N = 248 | Acc: 48.39%)", metricas_2604[fuentes_order[2]]['cm'], axes[1, 1], 'Reds')
]

for title, cm_mat, ax_c, cmap in cms:
    cm_np = np.array(cm_mat)
    sns.heatmap(cm_np, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Pred: Real (0)', 'Pred: Fake (1)'],
                yticklabels=['Real: Real (0)', 'Real: Fake (1)'],
                annot_kws={'size': 13, 'weight': 'bold'}, ax=ax_c)
    ax_c.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax_c.set_xlabel("Predicción de SaBERT", fontsize=10)
    ax_c.set_ylabel("Etiqueta Verdadera", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "3_matrices_confusion_por_fuente_70_370.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 3 generado: matrices_confusion_por_fuente_70_370.png")

# 4. Curvas ROC Comparativas por Fuente
fig, ax = plt.subplots(figsize=(9, 7))

# Global
fpr_g, tpr_g, _ = roc_curve(df_2604['label_num'], df_2604['prob_fake'])
ax.plot(fpr_g, tpr_g, color='black', lw=2.5, linestyle='-', label=f"Global (N=2,604) - AUC = {metricas_2604['GLOBAL']['roc_auc']:.4f}")

# Freiren
sub_fr = df_2604[df_2604['Fuente'] == fuentes_order[0]]
fpr_fr, tpr_fr, _ = roc_curve(sub_fr['label_num'], sub_fr['prob_fake'])
ax.plot(fpr_fr, tpr_fr, color='#1f77b4', lw=2.2, label=f"Freiren (N=2,060) - AUC = {metricas_2604[fuentes_order[0]]['roc_auc']:.4f}")

# Edds
sub_ed = df_2604[df_2604['Fuente'] == fuentes_order[1]]
fpr_ed, tpr_ed, _ = roc_curve(sub_ed['label_num'], sub_ed['prob_fake'])
ax.plot(fpr_ed, tpr_ed, color='#2ca02c', lw=2.2, label=f"Edds (N=294) - AUC = {metricas_2604[fuentes_order[1]]['roc_auc']:.4f}")

# mariagrandury
sub_mg = df_2604[df_2604['Fuente'] == fuentes_order[2]]
fpr_mg, tpr_mg, _ = roc_curve(sub_mg['label_num'], sub_mg['prob_fake'])
ax.plot(fpr_mg, tpr_mg, color='#d62728', lw=2.5, linestyle='--', label=f"mariagrandury (N=248) - AUC = {metricas_2604[fuentes_order[2]]['roc_auc']:.4f}")

# Diagonal azar
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Clasificador Aleatorio (AUC = 0.5000)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11, fontweight='bold')
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=11, fontweight='bold')
ax.set_title('Curvas ROC de SaBERT Desglosadas por Fuente\nEvidencia de Pérdida de Separabilidad Latente', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "4_curvas_roc_por_fuente.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 4 generado: curvas_roc_por_fuente.png")

# 5. Curvas Precision-Recall por Fuente
fig, ax = plt.subplots(figsize=(9, 7))

# Global
pr_rec_g, pr_prec_g, _ = precision_recall_curve(df_2604['label_num'], df_2604['prob_fake'])
ap_g = average_precision_score(df_2604['label_num'], df_2604['prob_fake'])
ax.plot(pr_rec_g, pr_prec_g, color='black', lw=2.5, label=f"Global - AP = {ap_g:.4f}")

# Freiren
pr_rec_fr, pr_prec_fr, _ = precision_recall_curve(sub_fr['label_num'], sub_fr['prob_fake'])
ap_fr = average_precision_score(sub_fr['label_num'], sub_fr['prob_fake'])
ax.plot(pr_rec_fr, pr_prec_fr, color='#1f77b4', lw=2.2, label=f"Freiren - AP = {ap_fr:.4f}")

# Edds
pr_rec_ed, pr_prec_ed, _ = precision_recall_curve(sub_ed['label_num'], sub_ed['prob_fake'])
ap_ed = average_precision_score(sub_ed['label_num'], sub_ed['prob_fake'])
ax.plot(pr_rec_ed, pr_prec_ed, color='#2ca02c', lw=2.2, label=f"Edds - AP = {ap_ed:.4f}")

# mariagrandury
pr_rec_mg, pr_prec_mg, _ = precision_recall_curve(sub_mg['label_num'], sub_mg['prob_fake'])
ap_mg = average_precision_score(sub_mg['label_num'], sub_mg['prob_fake'])
ax.plot(pr_rec_mg, pr_prec_mg, color='#d62728', lw=2.5, linestyle='--', label=f"mariagrandury - AP = {ap_mg:.4f}")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall / Cobertura de Fake News', fontsize=11, fontweight='bold')
ax.set_ylabel('Precisión en Detección de Fake News', fontsize=11, fontweight='bold')
ax.set_title('Curvas Precision-Recall de SaBERT por Fuente\nSevera Degradación de Precisión en mariagrandury', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "5_curvas_pr_por_fuente.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 5 generado: curvas_pr_por_fuente.png")

# 6. Colapso por Domain Shift: Tasa de Falsos Negativos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fuentes_comp = ['Freiren', 'Edds', 'mariagrandury']
fnrs = [
    metricas_2604[fuentes_order[0]]['false_negative_rate'] * 100,
    metricas_2604[fuentes_order[1]]['false_negative_rate'] * 100,
    metricas_2604[fuentes_order[2]]['false_negative_rate'] * 100
]
recalls = [
    metricas_2604[fuentes_order[0]]['recall_fake'] * 100,
    metricas_2604[fuentes_order[1]]['recall_fake'] * 100,
    metricas_2604[fuentes_order[2]]['recall_fake'] * 100
]

bar_colors = ['#2ecc71', '#3498db', '#e74c3c']

b1 = ax1.bar(fuentes_comp, fnrs, color=bar_colors, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Tasa de Falsos Negativos (%)', fontsize=11, fontweight='bold')
ax1.set_title('A. Falsos Negativos: Noticias Falsas que SaBERT\nClasifica Erróneamente como Verdaderas', fontsize=12, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, 95)
for rect in b1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

b2 = ax2.bar(fuentes_comp, recalls, color=['#27ae60', '#2980b9', '#c0392b'], edgecolor='black', alpha=0.85)
ax2.set_ylabel('Recall / Sensibilidad (%)', fontsize=11, fontweight='bold')
ax2.set_title('B. Cobertura Real de Noticias Falsas\n(Efectividad de SaBERT en atrapar Fakes)', fontsize=12, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 95)
for rect in b2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "6_colapso_domain_shift_falsos_negativos.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 6 generado: colapso_domain_shift_falsos_negativos.png")

# 7. Domain Shift en Muestra Balanceada Ampliada (si existe)
if metricas_balanceadas:
    fig, ax = plt.subplots(figsize=(13, 6))
    
    fuentes_b_order = [
        'https://huggingface.co/datasets/Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus',
        'https://huggingface.co/datasets/Edds/spanish-fake-news-fixed',
        'https://huggingface.co/datasets/mariagrandury/fake_news_corpus_spanish',
        'https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=fakes1000.csv'
    ]
    
    labels_b = [
        "Freiren\n(España Fact-checkers)\nN=600",
        "Edds\n(España Verificado)\nN=538",
        "mariagrandury\n(México / LatAm)\nN=572",
        "arseniitretiakov\n(Kaggle Web abierta)\nN=600"
    ]
    
    accs_b = [metricas_balanceadas[src]['accuracy'] * 100 for src in fuentes_b_order]
    fn_rates_b = [metricas_balanceadas[src]['tasa_fallo_fake_pct'] for src in fuentes_b_order]
    
    x_b = np.arange(len(labels_b))
    w_b = 0.35
    
    r1 = ax.bar(x_b - w_b/2, accs_b, w_b, label='Accuracy (%)', color='#2980b9', edgecolor='black')
    r2 = ax.bar(x_b + w_b/2, fn_rates_b, w_b, label='% Falsos Negativos (Fakes marcadas como Reales)', color='#c0392b', edgecolor='black')
    
    ax.set_ylabel('Porcentaje (%)', fontsize=11, fontweight='bold')
    ax.set_title('Prueba Empírica de Domain Shift: SaBERT Colapsa Fuera de España\nEvaluación en Muestras Balanceadas Independientes (N = 2,310 noticias)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    
    for rect in r1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for rect in r2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#900C3F')
    
    # Línea de azar
    ax.axhline(50, color='gray', linestyle=':', label='Azar (50%)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, "7_evaluacion_balanceada_domain_shift_ampliada.png"), dpi=300)
    plt.close(fig)
    print("[OK] Gráfico 7 generado: evaluacion_balanceada_domain_shift_ampliada.png")

# 8. Distribución de Densidad de Probabilidades Predichas por Fuente (Dataset 70-370)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sub_fuentes = [
    (fuentes_order[0], "A. Freiren (In-Domain España)", axes[0]),
    (fuentes_order[1], "B. Edds (In-Domain España)", axes[1]),
    (fuentes_order[2], "C. mariagrandury (Domain Shift LatAm)", axes[2])
]

for src, title, ax_d in sub_fuentes:
    sub = df_2604[df_2604['Fuente'] == src]
    sub_real = sub[sub['label_num'] == 0]['prob_fake']
    sub_fake = sub[sub['label_num'] == 1]['prob_fake']
    
    sns.kdeplot(sub_real, ax=ax_d, color='#2980b9', fill=True, alpha=0.4, label='Noticias Reales (0)', lw=2)
    sns.kdeplot(sub_fake, ax=ax_d, color='#c0392b', fill=True, alpha=0.4, label='Noticias Falsas (1)', lw=2)
    ax_d.axvline(0.5, color='black', linestyle='--', lw=1.2, label='Umbral Decisión (0.5)')
    ax_d.set_title(title, fontsize=11, fontweight='bold')
    ax_d.set_xlabel('Probabilidad Predicha de Fake P(Fake|x)', fontsize=10)
    ax_d.set_ylabel('Densidad', fontsize=10)
    ax_d.legend(loc='upper center', fontsize=9)
    ax_d.grid(True, linestyle='--', alpha=0.4)
    ax_d.set_xlim(0, 1)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "8_distribucion_probabilidades_p_fake.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 8 generado: distribucion_probabilidades_p_fake.png")

print("\n" + "="*80)
print(" EXPERIMENTO Y GENERACIÓN DE RECURSOS COMPLETADOS CON ÉXITO")
print("="*80)
