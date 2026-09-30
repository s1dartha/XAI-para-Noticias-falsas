import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

script_dir = os.path.dirname(os.path.abspath(__file__))
sabert_root = os.path.abspath(os.path.join(script_dir, ".."))
clasificar_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
dataset_dir = os.path.join(clasificar_root, "Dataset")
reportes_dir = os.path.join(sabert_root, "Reportes")
img_dir = os.path.join(reportes_dir, "Imagenes")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(reportes_dir, exist_ok=True)

print("="*80)
print(" REORGANIZACIÓN DEL DATASET Y RE-EVALUACIÓN ESPAÑA VS AMÉRICA LATINA (N = 4,418)")
print("="*80)

# -------------------------------------------------------------------------
# 1. REORGANIZACIÓN DEL DATASET: 2 CLASES Y ENFOQUE ESPAÑA VS LATAM
# -------------------------------------------------------------------------
path_dataset_in = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.xlsx")
df = pd.read_excel(path_dataset_in)
print(f"[INFO] Dataset original cargado con {len(df)} filas.")

# 1. Definir región estricta binaria: España vs América Latina
if 'pais' in df.columns:
    df['region'] = df['pais'].apply(lambda x: 'España' if 'España' in str(x) else 'América Latina')
elif 'region' not in df.columns:
    df['region'] = 'América Latina'

# 2. Definir exactamente 2 columnas de clase:
# - categoria: 'VERDADERA' / 'FALSA' (texto)
# - clase_num: 0 / 1 (numérico: 0 = VERDADERA, 1 = FALSA)
df['categoria'] = df['categoria'].astype(str).str.strip().str.upper()
df['clase_num'] = df['categoria'].apply(lambda c: 1 if c == 'FALSA' else 0).astype(int)

# 3. Columnas finales estrictas (8 columnas)
columnas_finales = [
    'Text',
    'conteo_palabras_text',
    'region',
    'categoria',
    'clase_num',
    'Fuente',
    'subfuente_medio',
    'dataset_origen'
]

df_reorg = df[columnas_finales].copy()

# Guardar en Dataset/
path_clean_xlsx = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.xlsx")
path_clean_csv = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv")
df_reorg.to_excel(path_clean_xlsx, index=False)
df_reorg.to_csv(path_clean_csv, index=False)
print(f"[OK] Guardado en Dataset/: {path_clean_xlsx} y .csv")

# -------------------------------------------------------------------------
# 2. ACTUALIZACIÓN DE PREDICCIONES CACHEADAS
# -------------------------------------------------------------------------
pred_cache_file = os.path.join(dataset_dir, "predicciones_sabert_ampliado_4418.csv")
df_cached = pd.read_csv(pred_cache_file)
df_reorg['pred_sabert'] = df_cached['pred_sabert'].astype(int)
df_reorg['prob_fake'] = df_cached['prob_fake'].astype(float)
df_reorg['prob_real'] = df_cached['prob_real'].astype(float)

# Guardar predicciones actualizadas
df_reorg.to_csv(pred_cache_file, index=False)
print(f"[OK] Predicciones actualizadas guardadas en: {pred_cache_file}")

# -------------------------------------------------------------------------
# 3. CÁLCULO DE MÉTRICAS: ESPAÑA VS AMÉRICA LATINA Y GLOBAL
# -------------------------------------------------------------------------
def compute_metrics(sub_df, name):
    y = sub_df['clase_num'].values
    p = sub_df['pred_sabert'].values
    pr = sub_df['prob_fake'].values
    acc = accuracy_score(y, p)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    prec_1, rec_1, f1_1, _ = precision_recall_fscore_support(y, p, average='binary', pos_label=1, zero_division=0)
    prec_0, rec_0, f1_0, _ = precision_recall_fscore_support(y, p, average='binary', pos_label=0, zero_division=0)
    auc_val = roc_auc_score(y, pr) if len(np.unique(y)) > 1 else None
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        'nombre': name,
        'n_total': int(len(sub_df)),
        'n_real': int(np.sum(y == 0)),
        'n_fake': int(np.sum(y == 1)),
        'balance_real_pct': float(np.sum(y == 0) / len(sub_df) * 100),
        'balance_fake_pct': float(np.sum(y == 1) / len(sub_df) * 100),
        'accuracy': float(acc),
        'f1_macro': float(f1_m),
        'f1_fake': float(f1_1),
        'precision_fake': float(prec_1),
        'recall_fake': float(rec_1),
        'f1_real': float(f1_0),
        'precision_real': float(prec_0),
        'recall_real': float(rec_0),
        'roc_auc': float(auc_val) if auc_val is not None else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cm': cm.tolist(),
        'false_negative_rate': fnr,
        'false_positive_rate': fpr
    }

sub_esp = df_reorg[df_reorg['region'] == 'España']
sub_lat = df_reorg[df_reorg['region'] == 'América Latina']

metricas_dict = {
    'GLOBAL': compute_metrics(df_reorg, "Global Ampliado (N = 4,418)"),
    'REGIONES': {
        'España': compute_metrics(sub_esp, "España (En Dominio)"),
        'América Latina': compute_metrics(sub_lat, "América Latina (Domain Shift)")
    },
    'FUENTES': {},
    'DATASETS': {}
}

# Desglose secundario por fuentes y datasets para trazabilidad técnica
for src, sub in df_reorg.groupby('Fuente'):
    metricas_dict['FUENTES'][src] = compute_metrics(sub, f"Fuente: {src}")

for dset, sub in df_reorg.groupby('dataset_origen'):
    metricas_dict['DATASETS'][dset] = compute_metrics(sub, f"Dataset: {dset}")

json_path = os.path.join(reportes_dir, "metricas_sabert_dataset_ampliado_4418.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(metricas_dict, f, indent=2, ensure_ascii=False)
print(f"[OK] Métricas consolidadas guardadas en {json_path}")

# -------------------------------------------------------------------------
# 4. GENERACIÓN DE GRÁFICOS: ENFOQUE ESTRICTO ESPAÑA VS AMÉRICA LATINA
# -------------------------------------------------------------------------
print("\n>>> Generando suite completa de gráficos analíticos (España vs América Latina)...")

# ---------------------------------------------------------
# Gráfico 1: Composición Macro-Regional y Balance de Clases
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Torta España vs América Latina
reg_counts = df_reorg['region'].value_counts()
ax1.pie(
    [reg_counts['España'], reg_counts['América Latina']],
    labels=[f"España\n(N = 2.354 | 53,28%)",
            f"América Latina\n(N = 2.064 | 46,72%)"],
    autopct='%1.1f%%',
    colors=['#2b5c8f', '#d35400'],
    startangle=140,
    explode=(0.04, 0.04),
    textprops={'fontsize': 12, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax1.set_title("A. Distribución Macro-Regional del Corpus Ampliado\n(N = 4.418 noticias sin web abierta)", fontsize=13, fontweight='bold', pad=15)

# Panel B: Barras de Balance de Clases por Región y Total
labels_bar = ['España\n(N = 2.354)', 'América Latina\n(N = 2.064)', 'GLOBAL Total\n(N = 4.418)']
reales_bar = [metricas_dict['REGIONES']['España']['n_real'],
              metricas_dict['REGIONES']['América Latina']['n_real'],
              metricas_dict['GLOBAL']['n_real']]
fakes_bar = [metricas_dict['REGIONES']['España']['n_fake'],
             metricas_dict['REGIONES']['América Latina']['n_fake'],
             metricas_dict['GLOBAL']['n_fake']]

x_b = np.arange(len(labels_bar))
width_b = 0.35

rects1 = ax2.bar(x_b - width_b/2, reales_bar, width_b, label='Verdadera (Clase 0)', color='#3498db', edgecolor='black', alpha=0.9)
rects2 = ax2.bar(x_b + width_b/2, fakes_bar, width_b, label='Falsa (Clase 1)', color='#e74c3c', edgecolor='black', alpha=0.9)

ax2.set_ylabel("Cantidad de Noticias", fontsize=11, fontweight='bold')
ax2.set_title("B. Balance Paritario de Clases: España vs. América Latina", fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_b)
ax2.set_xticklabels(labels_bar, fontsize=11, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
    h = rect.get_height()
    ax2.annotate(f"{h}\n({h/sum(reales_bar[:2] if rect.get_x()<1.5 else [metricas_dict['GLOBAL']['n_total']])*100:.1f}%)" if rect.get_x()>=1.5 else f"{h}\n({h/(metricas_dict['REGIONES']['España' if rect.get_x()<0 else 'América Latina']['n_total'])*100:.1f}%)",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    total_reg = metricas_dict['GLOBAL']['n_total'] if rect.get_x()>=1.5 else metricas_dict['REGIONES']['España' if rect.get_x()<1.0 else 'América Latina']['n_total']
    ax2.annotate(f"{h}\n({h/total_reg*100:.1f}%)",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')

ax2.set_ylim(0, 2700)
plt.tight_layout()
fig.savefig(os.path.join(img_dir, "1_distribucion_composicion_dataset_4418.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 1: 1_distribucion_composicion_dataset_4418.png")

# ---------------------------------------------------------
# Gráfico 2: Desempeño Comparativo: España vs América Latina
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
grupos_eval = [
    ('España\n(N = 2.354)', metricas_dict['REGIONES']['España']),
    ('América Latina\n(N = 2.064)', metricas_dict['REGIONES']['América Latina']),
    ('GLOBAL Ampliado\n(N = 4.418)', metricas_dict['GLOBAL'])
]

x = np.arange(len(grupos_eval))
width = 0.18

acc_v = [g[1]['accuracy'] * 100 for g in grupos_eval]
f1_v = [g[1]['f1_macro'] * 100 for g in grupos_eval]
rec_fake_v = [g[1]['recall_fake'] * 100 for g in grupos_eval]
auc_v = [g[1]['roc_auc'] * 100 for g in grupos_eval]

r1 = ax.bar(x - 1.5*width, acc_v, width, label='Accuracy (%)', color='#2b5c8f', edgecolor='black')
r2 = ax.bar(x - 0.5*width, f1_v, width, label='F1-Score Macro (%)', color='#4682b4', edgecolor='black')
r3 = ax.bar(x + 0.5*width, rec_fake_v, width, label='Recall Fake News (%)', color='#e67e22', edgecolor='black')
r4 = ax.bar(x + 1.5*width, auc_v, width, label='ROC-AUC (x100)', color='#27ae60', edgecolor='black')

ax.set_ylabel('Porcentaje / Puntuación (%)', fontsize=11, fontweight='bold')
ax.set_title('Desempeño de SaBERT: España (En Dominio) vs América Latina (Domain Shift)\nCaída Drástica en Cobertura de Fake News fuera de España (-48,7% Recall)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in grupos_eval], fontsize=11, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 115)

for rect in r1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in r3:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B03A2E')
for rect in r4:
    h = rect.get_height()
    ax.annotate(f'{h/100:.3f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#196f3d')

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "2_metricas_sabert_espana_vs_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 2: 2_metricas_sabert_espana_vs_latam.png")

# ---------------------------------------------------------
# Gráfico 3: Panel de Matrices de Confusión: España vs América Latina vs Global
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
cms_panel = [
    ("A. ESPAÑA (N = 2.354 | Acc: 89,72%)", metricas_dict['REGIONES']['España']['cm'], axes[0], 'Greens'),
    ("B. AMÉRICA LATINA (N = 2.064 | Acc: 65,89%)", metricas_dict['REGIONES']['América Latina']['cm'], axes[1], 'Reds'),
    ("C. GLOBAL (N = 4.418 | Acc: 78,59%)", metricas_dict['GLOBAL']['cm'], axes[2], 'Blues')
]

for title, cm_mat, ax_p, cmap in cms_panel:
    cm_np = np.array(cm_mat)
    sns.heatmap(cm_np, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Pred: Verdadera (0)', 'Pred: Falsa (1)'],
                yticklabels=['Real: Verdadera (0)', 'Real: Falsa (1)'],
                annot_kws={'size': 13, 'weight': 'bold'}, ax=ax_p)
    ax_p.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax_p.set_xlabel("Predicción de SaBERT", fontsize=10, fontweight='bold')
    ax_p.set_ylabel("Etiqueta Real", fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "3_matrices_confusion_espana_vs_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 3: 3_matrices_confusion_espana_vs_latam.png")

# ---------------------------------------------------------
# Gráfico 4: Curvas ROC Comparativas: España vs América Latina vs Global
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

# Global
fpr_g, tpr_g, _ = roc_curve(df_reorg['clase_num'], df_reorg['prob_fake'])
ax.plot(fpr_g, tpr_g, color='black', lw=2.5, label=f"Global Ampliado (N = 4.418) - AUC = {metricas_dict['GLOBAL']['roc_auc']:.4f}")

# España
fpr_es, tpr_es, _ = roc_curve(sub_esp['clase_num'], sub_esp['prob_fake'])
ax.plot(fpr_es, tpr_es, color='#27ae60', lw=2.8, label=f"España (N = 2.354) - AUC = {metricas_dict['REGIONES']['España']['roc_auc']:.4f}")

# América Latina
fpr_lat, tpr_lat, _ = roc_curve(sub_lat['clase_num'], sub_lat['prob_fake'])
ax.plot(fpr_lat, tpr_lat, color='#e74c3c', lw=2.8, linestyle='--', label=f"América Latina (N = 2.064) - AUC = {metricas_dict['REGIONES']['América Latina']['roc_auc']:.4f}")

# Azar
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Clasificador Aleatorio (AUC = 0.5000)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11, fontweight='bold')
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=11, fontweight='bold')
ax.set_title('Curvas ROC de SaBERT: Colapso de Separabilidad en América Latina\n(Brecha de AUC: 0,9536 en España vs 0,6800 en LatAm)', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10.5)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "4_curvas_roc_espana_vs_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 4: 4_curvas_roc_espana_vs_latam.png")

# ---------------------------------------------------------
# Gráfico 5: Curvas Precision-Recall Comparativas
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

pr_rec_g, pr_prec_g, _ = precision_recall_curve(df_reorg['clase_num'], df_reorg['prob_fake'])
ap_g = average_precision_score(df_reorg['clase_num'], df_reorg['prob_fake'])
ax.plot(pr_rec_g, pr_prec_g, color='black', lw=2.5, label=f"Global Ampliado - AP = {ap_g:.4f}")

pr_rec_es, pr_prec_es, _ = precision_recall_curve(sub_esp['clase_num'], sub_esp['prob_fake'])
ap_es = average_precision_score(sub_esp['clase_num'], sub_esp['prob_fake'])
ax.plot(pr_rec_es, pr_prec_es, color='#27ae60', lw=2.8, label=f"España - AP = {ap_es:.4f}")

pr_rec_lat, pr_prec_lat, _ = precision_recall_curve(sub_lat['clase_num'], sub_lat['prob_fake'])
ap_lat = average_precision_score(sub_lat['clase_num'], sub_lat['prob_fake'])
ax.plot(pr_rec_lat, pr_prec_lat, color='#e74c3c', lw=2.8, linestyle='--', label=f"América Latina - AP = {ap_lat:.4f}")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall / Cobertura de Fake News', fontsize=11, fontweight='bold')
ax.set_ylabel('Precisión en Detección de Fake News', fontsize=11, fontweight='bold')
ax.set_title('Curvas Precision-Recall: Degradación Vertical de SaBERT en LatAm\n(Average Precision cae de 0,9490 a 0,6558)', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower left', fontsize=10.5)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "5_curvas_pr_espana_vs_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 5: 5_curvas_pr_espana_vs_latam.png")

# ---------------------------------------------------------
# Gráfico 6: Falsos Negativos: España vs América Latina
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
comp_regiones = ['España (N = 2.354)', 'América Latina (N = 2.064)', 'GLOBAL (N = 4.418)']
fnr_values = [
    metricas_dict['REGIONES']['España']['false_negative_rate'] * 100,
    metricas_dict['REGIONES']['América Latina']['false_negative_rate'] * 100,
    metricas_dict['GLOBAL']['false_negative_rate'] * 100
]
colores_fnr = ['#27ae60', '#c0392b', '#7f8c8d']

bars = ax.bar(comp_regiones, fnr_values, color=colores_fnr, edgecolor='black', alpha=0.85, width=0.5)
ax.set_ylabel('Tasa de Falsos Negativos (%)', fontsize=11, fontweight='bold')
ax.set_title('Tasa de Falsos Negativos de SaBERT: El Colapso por Domain Shift\n(% de Noticias Falsas que el Modelo Deja Pasar como Verdaderas)', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 100)

for rect in bars:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 2.0, f"{h:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

# Subtítulo explicativo interno
ax.text(0, 40, "Solo 20 de cada 100\nfakes se escapan", ha='center', fontsize=10, style='italic', color='#145a32', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60'))
ax.text(1, 40, "¡69 de cada 100\nfakes se escapan!", ha='center', fontsize=10, style='italic', color='#78281f', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#c0392b'))

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "6_colapso_domain_shift_falsos_negativos_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 6: 6_colapso_domain_shift_falsos_negativos_latam.png")

# ---------------------------------------------------------
# Gráfico 8: Densidad de Probabilidades P(Fake): España vs América Latina
# ---------------------------------------------------------
fig, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(15, 5.5))

# España
sns.kdeplot(sub_esp[sub_esp['clase_num'] == 0]['prob_fake'], ax=ax_d1, color='#2980b9', fill=True, alpha=0.4, label='Verdaderas (Clase 0)', lw=2)
sns.kdeplot(sub_esp[sub_esp['clase_num'] == 1]['prob_fake'], ax=ax_d1, color='#c0392b', fill=True, alpha=0.4, label='Falsas (Clase 1)', lw=2)
ax_d1.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Umbral Decisión (0.5)')
ax_d1.set_title("A. España (En Dominio): Bimodalidad y Alta Separabilidad\n(Las clases se concentran en extremos opuestos)", fontsize=11, fontweight='bold')
ax_d1.set_xlabel('Probabilidad Predicha P(Fake|x)', fontsize=10, fontweight='bold')
ax_d1.set_ylabel('Densidad', fontsize=10, fontweight='bold')
ax_d1.legend(loc='upper center', fontsize=9.5)
ax_d1.grid(True, linestyle='--', alpha=0.4)
ax_d1.set_xlim(0, 1)

# América Latina
sns.kdeplot(sub_lat[sub_lat['clase_num'] == 0]['prob_fake'], ax=ax_d2, color='#2980b9', fill=True, alpha=0.4, label='Verdaderas (Clase 0)', lw=2)
sns.kdeplot(sub_lat[sub_lat['clase_num'] == 1]['prob_fake'], ax=ax_d2, color='#c0392b', fill=True, alpha=0.4, label='Falsas (Clase 1)', lw=2)
ax_d2.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Umbral Decisión (0.5)')
ax_d2.set_title("B. América Latina (Domain Shift): Colapso hacia la Izquierda\n(Las noticias falsas colapsan a la izquierda y se confunden con reales)", fontsize=11, fontweight='bold')
ax_d2.set_xlabel('Probabilidad Predicha P(Fake|x)', fontsize=10, fontweight='bold')
ax_d2.set_ylabel('Densidad', fontsize=10, fontweight='bold')
ax_d2.legend(loc='upper right', fontsize=9.5)
ax_d2.grid(True, linestyle='--', alpha=0.4)
ax_d2.set_xlim(0, 1)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "8_distribucion_probabilidades_espana_vs_latam.png"), dpi=300)
plt.close(fig)
print("[OK] Gráfico 8: 8_distribucion_probabilidades_espana_vs_latam.png")

print("\n" + "="*80)
print(" PROCESO COMPLETADO EXITOSAMENTE")
print("="*80)
