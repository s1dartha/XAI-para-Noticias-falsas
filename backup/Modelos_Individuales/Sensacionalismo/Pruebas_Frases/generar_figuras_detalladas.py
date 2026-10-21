#!/usr/bin/env python3
"""
Generación de Figuras Científicas Detalladas y Educativas
para el Reporte de Sensacionalismo: Frases vs Documento.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(SENS_ROOT, "..", ".."))
REPORT_DIR = os.path.join(SENS_ROOT, "reportes")
IMAGE_DIR = os.path.join(REPORT_DIR, "imagenes", "sensacionalismo_frases")
CSV_PATH = os.path.join(REPORT_DIR, "datos_sensacionalismo_frases.csv")
JSON_PATH = os.path.join(REPORT_DIR, "metricas_sensacionalismo_frases.json")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Estilo visual limpio y publicable
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

df_results = pd.read_csv(CSV_PATH)
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

stat_summary = metrics['estadisticas_ajuste']
ablation_summary = metrics['ablacion_causal']

# Cargar también los textos de frases de los 4 casos para la Figura 7
candidate_ama = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_Amarillismo.csv"),
    os.path.join(SENS_ROOT, "dataset_Amarillismo.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_Amarillismo.csv")
]
df_ama_raw = pd.read_csv(next((p for p in candidate_ama if os.path.exists(p)), candidate_ama[0]))

candidate_ia = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_IA_sintetico_70.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_IA_sintetico_70.csv")
]
df_ia_raw = pd.read_csv(next((p for p in candidate_ia if os.path.exists(p)), candidate_ia[0]))

def split_into_phrases(text):
    raw_pieces = re.split(r'[,;:.!?¡¿\n—–\"\'«»()]+', str(text))
    return [p.strip() for p in raw_pieces if len(p.strip().split()) >= 2 and len(p.strip()) >= 6]

print("[FIG 1] Generando Figura 1: Noticia Completa vs 4 Reglas de Agregación con distinción de Dataset...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12.5))
configs_f1 = [
    (axes[0, 0], 'p_mean', 'Promedio Simple (Mean)'),
    (axes[0, 1], 'p_max', 'Máximo (Top-1 Frase)'),
    (axes[1, 0], 'p_top2', 'Promedio Top-2 Frases'),
    (axes[1, 1], 'p_first', 'Titular / Primera Frase')
]

for ax, col_name, title_suffix in configs_f1:
    m_key = col_name.replace('p_', '')
    # Separar por dataset
    sub_ama = df_results[df_results['fuente'].str.contains('Amarillismo')]
    sub_ia = df_results[df_results['fuente'].str.contains('IA')]
    
    # Scatter por dataset
    ax.scatter(sub_ama[col_name], sub_ama['p_full'], color='#1f77b4', alpha=0.55, s=35, label='Prensa Digital Real (202)', edgecolors='none')
    ax.scatter(sub_ia[col_name], sub_ia['p_full'], color='#ff7f0e', alpha=0.85, s=45, marker='^', label='Noticias Sintéticas IA (70)', edgecolors='black', linewidths=0.5)
    
    # Regresión global
    sns.regplot(data=df_results, x=col_name, y='p_full', ax=ax, scatter=False, color='#2ca02c', line_kws={'linewidth': 2.0, 'label': 'Ajuste Lineal Global'})
    # Línea ideal y = x
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.2, label='Ideal y = x')
    
    r_glob = stat_summary['global'][m_key]['pearson_r']
    r2_glob = stat_summary['global'][m_key]['r2']
    r_ama = stat_summary['Amarillismo Real (Prensa)'][m_key]['pearson_r']
    r_ia = stat_summary['IA Sintético (70 reg.)'][m_key]['pearson_r']
    
    ax.set_title(f"Noticia Completa vs. {title_suffix}\nGlobal: r={r_glob:.2f}, R²={r2_glob:.2f} | Prensa Real: r={r_ama:.2f} | IA: r={r_ia:.2f}", fontsize=10.5, fontweight='bold')
    ax.set_xlabel(f"Sensacionalismo de Frases: {title_suffix}", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Sensacionalismo Noticia Completa P(Full)", fontsize=10, fontweight='semibold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=8.5, frameon=True)

fig.suptitle("FIGURA 1: Dispersión y Regresión entre Noticia Completa y 4 Reglas de Agregación de Frases\nContraste de Datasets: Prensa Digital Real (202 noticias) vs. Noticias Sintéticas IA (70 noticias)", fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(IMAGE_DIR, "fig1_dispersion_regresion_pfull_vs_agregaciones.png")
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig1_path)


print("[FIG 2] Generando Figura 2: Benchmark Métricas de Ajuste...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
models_keys = ['mean', 'max', 'top2', 'first', 'w_mean']
models_labels = ['Promedio\nSimple', 'Máximo\n(Top-1)', 'Top-2\nPromedio', 'Titular /\n1ra Frase', 'Promedio\nPonderado']

mae_vals = [stat_summary['global'][k]['mae'] for k in models_keys]
rmse_vals = [stat_summary['global'][k]['rmse'] for k in models_keys]
r2_vals = [stat_summary['global'][k]['r2'] for k in models_keys]
acc_vals = [stat_summary['global'][k]['agreement_acc'] * 100 for k in models_keys]

x_pos = np.arange(len(models_labels))
width = 0.35

# Subplot 1: Errores MAE y RMSE
b1 = ax1.bar(x_pos - width/2, mae_vals, width, label='MAE (Error Absoluto Medio) ↓ Menor es mejor', color='#e6550d', alpha=0.9, edgecolor='#a63603')
b2 = ax1.bar(x_pos + width/2, rmse_vals, width, label='RMSE (Raíz Error Cuadrático) ↓ Menor es mejor', color='#3182bd', alpha=0.9, edgecolor='#08519c')

for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_ylabel("Magnitud del Error [0, 1]", fontsize=11, fontweight='bold')
ax1.set_title("Errores de Ajuste Numérico Frente a Noticia Completa\n(Corpus 272 Noticias: 202 Prensa Real + 70 IA)", fontsize=11, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_labels, fontsize=9.5, fontweight='semibold')
ax1.set_ylim(0, max(rmse_vals)*1.25)
ax1.legend(loc='upper right', frameon=True, fontsize=9)

# Subplot 2: R2 y Concordancia
b3 = ax2.bar(x_pos - width/2, r2_vals, width, label='R² (Bondad de Ajuste / Explicación Varianza) ↑', color='#756bb1', alpha=0.9, edgecolor='#54278f')
b4 = ax2.bar(x_pos + width/2, [a/100 for a in acc_vals], width, label='Concordancia Clasificatoria (% Match con P_full) ↑', color='#31a354', alpha=0.9, edgecolor='#006d2c')

for b, raw_acc in zip(b4, acc_vals):
    ax2.annotate(f'{raw_acc:.1f}%', xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b, raw_r2 in zip(b3, r2_vals):
    ax2.annotate(f'{raw_r2:.2f}', xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3 if raw_r2 >= 0 else -12), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel("Puntuación Relativa", fontsize=11, fontweight='bold')
ax2.set_title("Capacidad Predictiva y Preservación de la Decisión Binaria\n(Mayor Valor = Mejor Reproducción de P_full)", fontsize=11, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_labels, fontsize=9.5, fontweight='semibold')
ax2.set_ylim(min(r2_vals)*1.15, 1.25)
ax2.legend(loc='lower right', frameon=True, fontsize=9)

fig.suptitle("FIGURA 2: Benchmark Cuantitativo de 5 Reglas de Agregación de Frases vs. Noticia Completa", fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2_path = os.path.join(IMAGE_DIR, "fig2_comparativa_metricas_ajuste_mae_rmse_r2.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig2_path)


print("[FIG 3] Generando Figura 3: Distribución de Residuos por Dataset...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

sub_ama = df_results[df_results['fuente'].str.contains('Amarillismo')]
sub_ia = df_results[df_results['fuente'].str.contains('IA')]

res_mean_ama = sub_ama['p_full'] - sub_ama['p_mean']
res_mean_ia = sub_ia['p_full'] - sub_ia['p_mean']

sns.kdeplot(res_mean_ama, ax=ax1, color='#1f77b4', linewidth=2.2, label=f'Prensa Real (Sesgo medio = {res_mean_ama.mean():+.3f})', fill=True, alpha=0.3)
sns.kdeplot(res_mean_ia, ax=ax1, color='#ff7f0e', linewidth=2.2, label=f'IA Sintético (Sesgo medio = {res_mean_ia.mean():+.3f})', fill=True, alpha=0.3)
ax1.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Residuo Cero (Ajuste Perfecto)')
ax1.set_title("Residuo: P(Completa) - P(Promedio Frases)\nPrensa Real muestra sesgo masivo a la izquierda (Sobreestimación del Promedio)", fontsize=10.5, fontweight='bold')
ax1.set_xlabel("Diferencia P(Full) - P(Mean)", fontsize=10)
ax1.set_ylabel("Densidad Estimada", fontsize=10)
ax1.legend(frameon=True, fontsize=8.5, loc='upper left')

res_max_ama = sub_ama['p_full'] - sub_ama['p_max']
res_max_ia = sub_ia['p_full'] - sub_ia['p_max']

sns.kdeplot(res_max_ama, ax=ax2, color='#1f77b4', linewidth=2.2, label=f'Prensa Real (Sesgo medio = {res_max_ama.mean():+.3f})', fill=True, alpha=0.3)
sns.kdeplot(res_max_ia, ax=ax2, color='#ff7f0e', linewidth=2.2, label=f'IA Sintético (Sesgo medio = {res_max_ia.mean():+.3f})', fill=True, alpha=0.3)
ax2.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Residuo Cero (Ajuste Perfecto)')
ax2.set_title("Residuo: P(Completa) - P(Máximo Frase)\nPrensa Real muestra Dilución Contextual masiva (Frase Pico no contagia al todo)", fontsize=10.5, fontweight='bold')
ax2.set_xlabel("Diferencia P(Full) - P(Max)", fontsize=10)
ax2.set_ylabel("Densidad Estimada", fontsize=10)
ax2.legend(frameon=True, fontsize=8.5, loc='upper left')

fig.suptitle("FIGURA 3: Distribución Empírica de Residuos por Dataset (Prensa Digital Real vs. IA Sintético)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig3_path = os.path.join(IMAGE_DIR, "fig3_distribucion_residuos_sesgo_gatillo.png")
fig.savefig(fig3_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig3_path)


print("[FIG 4] Generando Figura 4: Ablación Causal con Top-1, Top-2 y Top-3...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

abl_ops = ['Quitar Top-1\nFrase', 'Quitar Top-2\nFrases', 'Quitar Top-3\nFrases', 'Quitar Frase\nAleatoria (Ctrl)']
drops_ama = [
    ablation_summary['Amarillismo Real (Prensa)']['drop_top1_mean'],
    ablation_summary['Amarillismo Real (Prensa)']['drop_top2_mean'],
    ablation_summary['Amarillismo Real (Prensa)']['drop_top3_mean'],
    ablation_summary['Amarillismo Real (Prensa)']['drop_rand_mean']
]
drops_ia = [
    ablation_summary['IA Sintético (70 reg.)']['drop_top1_mean'],
    ablation_summary['IA Sintético (70 reg.)']['drop_top2_mean'],
    ablation_summary['IA Sintético (70 reg.)']['drop_top3_mean'],
    ablation_summary['IA Sintético (70 reg.)']['drop_rand_mean']
]

x_abl = np.arange(len(abl_ops))
w_abl = 0.35

b_a1 = ax1.bar(x_abl - w_abl/2, drops_ama, w_abl, label='Prensa Real (202 noticias)', color='#1f77b4', alpha=0.9, edgecolor='#08519c')
b_a2 = ax1.bar(x_abl + w_abl/2, drops_ia, w_abl, label='IA Sintético (70 noticias)', color='#ff7f0e', alpha=0.9, edgecolor='#a63603')

for b in b_a1:
    h = b.get_height()
    ax1.annotate(f'{h:+.3f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3 if h>=0 else -12), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b in b_a2:
    h = b.get_height()
    ax1.annotate(f'{h:+.3f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3 if h>=0 else -12), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylabel("Variación Promedio ΔP = P(Full) - P(Ablado)", fontsize=10.5, fontweight='bold')
ax1.set_title("Impacto Causal en la Certidumbre de la Noticia\n(Valores negativos indican reducción en la certidumbre sensacionalista)", fontsize=11, fontweight='bold')
ax1.set_xticks(x_abl)
ax1.set_xticklabels(abl_ops, fontsize=9.5, fontweight='semibold')
ax1.set_ylim(min(drops_ama)*1.35, 0.1)
ax1.legend(loc='lower right', frameon=True, fontsize=9)

# Subplot 2: Tasa de Desclasificación (% Flips de Sensacionalista a Sobria)
flips_ama = [
    ablation_summary['Amarillismo Real (Prensa)']['flips_top1_pct'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_top2_pct'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_top3_pct'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_rand_pct']
]
flips_counts_ama = [
    ablation_summary['Amarillismo Real (Prensa)']['flips_top1_count'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_top2_count'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_top3_count'],
    ablation_summary['Amarillismo Real (Prensa)']['flips_rand_count']
]

bars_flip = ax2.bar(x_abl, flips_ama, color=['#d73027', '#e6550d', '#fdae61', '#4575b4'], width=0.5, alpha=0.9, edgecolor='#333333')
for b, cnt in zip(bars_flip, flips_counts_ama):
    h = b.get_height()
    ax2.annotate(f'{h:.1f}%\n({cnt} / 40)', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel("% Noticias Sensacionalistas Desclasificadas", fontsize=10.5, fontweight='bold')
ax2.set_title("Tasa de Inversión Predictiva en Prensa Real (Sensacionalista → Sobria)\n(En IA Sintético fue 0.0% en todas las intervenciones)", fontsize=11, fontweight='bold')
ax2.set_xticks(x_abl)
ax2.set_xticklabels(abl_ops, fontsize=9.5, fontweight='semibold')
ax2.set_ylim(0, max(flips_ama)*1.35)

fig.suptitle("FIGURA 4: Protocolo de Ablación Causal de Frases: Quitar Top-1, Top-2 y Top-3 Frases Clave", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig4_path = os.path.join(IMAGE_DIR, "fig4_ablacion_causal_frases_drop_flips.png")
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig4_path)


print("[FIG 5] Generando Figura 5: Perfil Posicional del Sensacionalismo con AMBOS Datasets...")
phrase_rows = []
for _, row in df_results.iterrows():
    if 'AMA' in row['doc_id']:
        idx = int(row['doc_id'].replace('AMA_', ''))
        r = df_ama_raw.iloc[idx]
        tit = str(r['Titular']).strip()
        cop = str(r.get('Copete/Resumen', '')).strip() if str(r.get('Copete/Resumen', '')) != 'nan' else ''
        cue = str(r['Cuerpo']).strip() if str(r['Cuerpo']) != 'nan' else ''
        full = f"{tit}. {cop} {cue}".strip()[:1200]
        phrases = split_into_phrases(full)
    else:
        idx = int(row['doc_id'].replace('IA_', ''))
        r = df_ia_raw.iloc[idx]
        texto = str(r['Texto']).strip()
        phrases = split_into_phrases(texto)
        if len(phrases) < 2:
            phrases = [p.strip() for p in re.split(r'[,;:]+', texto) if len(p.strip().split()) >= 2] or [texto]
            
    k = len(phrases)
    for p_idx, p_text in enumerate(phrases):
        rel_pos = p_idx / max(1, k - 1)
        phrase_rows.append({
            'doc_id': row['doc_id'],
            'fuente': row['fuente'],
            'doc_label': row['clase_real'],
            'clase_nombre': row['clase_nombre'],
            'phrase_idx': p_idx,
            'phrase_text': p_text,
            'rel_pos': rel_pos,
            'p_full': row['p_full'],
            'num_frases': k
        })

df_all_phr = pd.DataFrame(phrase_rows)
df_all_phr['quintil'] = pd.qcut(df_all_phr['rel_pos'], q=5, labels=['0-20%\n(Titular/Lead)', '20-40%\n(Copete)', '40-60%\n(Cuerpo)', '60-80%\n(Desarrollo)', '80-100%\n(Cierre)'])

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("JJNeila/bert-spanish-sensationalism-oss")
model = AutoModelForSequenceClassification.from_pretrained("JJNeila/bert-spanish-sensationalism-oss").to(DEVICE)
model.eval()

phr_texts = df_all_phr['phrase_text'].tolist()
batch_size = 64
p_phr_all = []
for i in range(0, len(phr_texts), batch_size):
    b = phr_texts[i:i+batch_size]
    inp = tokenizer(b, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        p_phr_all.extend(torch.softmax(model(**inp).logits, dim=-1)[:, 1].cpu().numpy())

df_all_phr['p_phrase'] = p_phr_all

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Prensa Digital Real
df_ama_phr = df_all_phr[df_all_phr['fuente'].str.contains('Amarillismo')].copy()
sns.lineplot(
    data=df_ama_phr,
    x='quintil',
    y='p_phrase',
    hue='clase_nombre',
    palette={'Amarillista': '#d73027', 'No Amarillista': '#2b5c8f'},
    marker='o',
    markersize=8,
    linewidth=2.5,
    errorbar=('ci', 95),
    ax=ax1
)
ax1.set_title("PANEL A: PRENSA DIGITAL REAL (202 Noticias - Corpus Amarillismo)\nEfecto Carga Frontal Marcado: Titular Alarmista (0.88) que desciende en el cuerpo", fontsize=11, fontweight='bold')
ax1.set_xlabel("Ubicación de la Frase en la Noticia (Quintiles Posicionales)", fontsize=10, fontweight='semibold')
ax1.set_ylabel("Sensacionalismo Promedio P(Sens | Frase)", fontsize=10, fontweight='semibold')
ax1.set_ylim(0.15, 0.98)
ax1.legend(title='Clase Real Periodística', frameon=True, fontsize=9.5)

# Panel 2: IA Sintético
df_ia_phr = df_all_phr[df_all_phr['fuente'].str.contains('IA')].copy()
# Segmentar IA en 3 posiciones: Inicio, Medio, Fin
df_ia_phr['pos_segmento'] = pd.cut(df_ia_phr['rel_pos'], bins=[-0.01, 0.33, 0.66, 1.01], labels=['Apertura\n(0-33%)', 'Medio\n(34-66%)', 'Cierre\n(67-100%)'])
sns.lineplot(
    data=df_ia_phr,
    x='pos_segmento',
    y='p_phrase',
    hue='clase_nombre',
    palette={'Sensacionalista': '#d73027', 'No sensacionalista': '#2b5c8f'},
    marker='s',
    markersize=8,
    linewidth=2.5,
    errorbar=('ci', 95),
    ax=ax2
)
ax2.set_title("PANEL B: NOTICIAS SINTÉTICAS IA (70 Noticias - Corpus IA 70)\nHomogeneidad Artificial: Curvas Planas (Saturación a ~0.99 vs. Sobriedad a ~0.03)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Ubicación de la Frase en el Texto Sintético", fontsize=10, fontweight='semibold')
ax2.set_ylabel("Sensacionalismo Promedio P(Sens | Frase)", fontsize=10, fontweight='semibold')
ax2.set_ylim(0.0, 1.05)
ax2.legend(title='Clase Real Sintética', frameon=True, fontsize=9.5)

fig.suptitle("FIGURA 5: Evolución de la Probabilidad de Sensacionalismo según la Ubicación de la Frase en el Documento (IC 95%)\nComparativa Directa entre Prensa Digital Real y Noticias Sintéticas Generadas por IA", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig5_path = os.path.join(IMAGE_DIR, "fig5_perfil_posicional_sensacionalismo.png")
fig.savefig(fig5_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig5_path)


print("[FIG 6] Generando Figura 6: Contraste Estructural IA Sintético vs Prensa Real...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: IA Sintético
sns.scatterplot(data=sub_ia, x='p_mean', y='p_full', hue='clase_nombre', palette={'Sensacionalista': '#d73027', 'No sensacionalista': '#2b5c8f'}, s=65, alpha=0.85, ax=ax1)
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Ideal y = x')
r_ia = stat_summary['IA Sintético (70 reg.)']['mean']['pearson_r']
r2_ia = stat_summary['IA Sintético (70 reg.)']['mean']['r2']
ax1.set_title(f"PANEL A: NOTICIAS SINTÉTICAS IA (70 reg.)\nComportamiento Homogéneo Lineal (r = {r_ia:.3f} | R² = {r2_ia:.3f})", fontsize=11, fontweight='bold')
ax1.set_xlabel("Promedio de Frases P(Mean)", fontsize=10, fontweight='semibold')
ax1.set_ylabel("Noticia Completa P(Full)", fontsize=10, fontweight='semibold')
ax1.legend(loc='lower right', frameon=True, fontsize=9.5)

# Panel 2: Prensa Real
sns.scatterplot(data=sub_ama, x='p_mean', y='p_full', hue='clase_nombre', palette={'Amarillista': '#d73027', 'No Amarillista': '#2b5c8f'}, s=55, alpha=0.75, ax=ax2)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Ideal y = x')
r_ama = stat_summary['Amarillismo Real (Prensa)']['mean']['pearson_r']
r2_ama = stat_summary['Amarillismo Real (Prensa)']['mean']['r2']
ax2.set_title(f"PANEL B: PRENSA DIGITAL REAL (202 reg.)\nComportamiento No Lineal Disociado (r = {r_ama:.3f} | R² = {r2_ama:.3f})", fontsize=11, fontweight='bold')
ax2.set_xlabel("Promedio de Frases P(Mean)", fontsize=10, fontweight='semibold')
ax2.set_ylabel("Noticia Completa P(Full)", fontsize=10, fontweight='semibold')
ax2.legend(loc='lower right', frameon=True, fontsize=9.5)

fig.suptitle("FIGURA 6: Contraste Estructural: Homogeneidad Sintética vs. Amortiguación Contextual en Prensa Real", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig6_path = os.path.join(IMAGE_DIR, "fig6_contraste_ia_sintetico_vs_prensa_real.png")
fig.savefig(fig6_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig6_path)


print("[FIG 7] Generando Figura 7: Estudio de Casos Paradigmáticos de AMBOS Corpus...")
# Casos representativos:
# 1. Prensa Real Sensacionalista: AMA_2 (Asteroide Apophis)
# 2. Prensa Real Sobria: AMA_11 (Nueva criatura marina)
# 3. IA Sintético Sensacionalista: IA_0 (Fin de la humanidad)
# 4. IA Sintético Sobria: IA_1 (Modelo de lenguaje automatiza tareas)

fig, axes = plt.subplots(2, 2, figsize=(18, 11))

cases_data = [
    (axes[0, 0], 'AMA_2', 'CASO 1: EFECTO GATILLO (PRENSA REAL - dataset_Amarillismo.csv)',
     'Clase Real: Amarillista | P(Full)=0.552 | Top-1 desclasifica el artículo'),
    (axes[0, 1], 'AMA_11', 'CASO 2: EFECTO DILUCIÓN (PRENSA REAL - dataset_Amarillismo.csv)',
     'Clase Real: No Amarillista | P(Full)=0.107 | Cuerpo científico neutraliza titular'),
    (axes[1, 0], 'IA_0', 'CASO 3: SATURACIÓN LÉXICA (IA SINTÉTICO - dataset_IA_sintetico_70.csv)',
     'Clase Real: Sensacionalista | P(Full)=0.996 | Todas las frases son alarmistas'),
    (axes[1, 1], 'IA_1', 'CASO 4: SOBRIEDAD UNIFORME (IA SINTÉTICO - dataset_IA_sintetico_70.csv)',
     'Clase Real: No sensacionalista | P(Full)=0.041 | Tono técnico sin estridencias')
]

for ax, doc_id, c_title, c_subtitle in cases_data:
    sub_doc = df_all_phr[df_all_phr['doc_id'] == doc_id].sort_values('phrase_idx')
    p_full_val = df_results[df_results['doc_id'] == doc_id]['p_full'].values[0]
    p_mean_val = sub_doc['p_phrase'].mean()
    
    # Textos de frases
    if 'AMA' in doc_id:
        idx = int(doc_id.replace('AMA_', ''))
        r = df_ama_raw.iloc[idx]
        full = f"{r['Titular']}. {r.get('Copete/Resumen', '')} {r['Cuerpo']}".strip()[:1200]
        phr_list = split_into_phrases(full)
    else:
        idx = int(doc_id.replace('IA_', ''))
        r = df_ia_raw.iloc[idx]
        t = str(r['Texto']).strip()
        phr_list = split_into_phrases(t)
        if len(phr_list) < 2:
            phr_list = [p.strip() for p in re.split(r'[,;:]+', t) if len(p.strip().split()) >= 2] or [t]
            
    # Truncar etiquetas para visualización limpia
    labels = [f"F{i+1}: {p[:36]}..." for i, p in enumerate(phr_list)]
    y_pos = np.arange(len(labels))
    probs = sub_doc['p_phrase'].values
    
    colors = ['#d73027' if p >= 0.5 else '#2b5c8f' for p in probs]
    ax.barh(y_pos, probs, color=colors, height=0.65, alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(p_full_val, color='red' if p_full_val >= 0.5 else 'blue', linestyle='-', linewidth=2.5, label=f'P(Completa) = {p_full_val:.3f}')
    ax.axvline(p_mean_val, color='orange', linestyle=':', linewidth=2.0, label=f'Promedio Frases = {p_mean_val:.3f}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_title(f"{c_title}\n{c_subtitle}", fontsize=9.5, fontweight='bold')
    ax.set_xlabel("P(Sens | Frase)", fontsize=9, fontweight='semibold')
    ax.legend(loc='lower right', frameon=True, fontsize=8.5)

fig.suptitle("FIGURA 7: Estudio Anatómico de Casos Ejemplares: Comparativa Frase a Frase por Corpus y Clase Real", fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig7_path = os.path.join(IMAGE_DIR, "fig7_casos_estudio_ejemplares.png")
fig.savefig(fig7_path, dpi=300)
plt.close(fig)
print("  -> Guardada:", fig7_path)

print("[SUCCESS] Todas las figuras fueron generadas y exportadas con éxito.")
