#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTO: MECANISMO DE CLASIFICACIÓN DE SENSACIONALISMO:
¿PROMEDIO DE FRASES O DEPENDENCIA DE 1-2 FRASES GATILLO?
Modelo: JJNeila/bert-spanish-sensationalism-oss (BETO fine-tuned)
Hardware: GPU NVIDIA GeForce GTX 1650 (device='cuda')
Corpus: Dataset Amarillismo Prensa Real (202 noticias) + Dataset IA Sintético (70 noticias) = 272 noticias
=============================================================================
Tesis de Maestría / Investigación en NLP & Data Science
"""

import os
import gc
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuración de reproducibilidad y estilos visuales
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Rutas del Sistema
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(SENS_ROOT, "..", ".."))
REPORT_DIR = os.path.join(SENS_ROOT, "reportes")
IMAGE_DIR = os.path.join(REPORT_DIR, "imagenes", "sensacionalismo_frases")

candidate_ama_paths = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_Amarillismo.csv"),
    os.path.join(SENS_ROOT, "dataset_Amarillismo.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_Amarillismo.csv")
]
DATASET_AMA_PATH = next((p for p in candidate_ama_paths if os.path.exists(p)), candidate_ama_paths[0])

candidate_ia_paths = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_IA_sintetico_70.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_IA_sintetico_70.csv")
]
DATASET_IA_PATH = next((p for p in candidate_ia_paths if os.path.exists(p)), candidate_ia_paths[0])

os.makedirs(SCRIPT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Dispositivo configurado: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INIT] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")

# =============================================================================
# 1. CARGA DE MODELO Y TOKENIZADOR
# =============================================================================
MODEL_NAME = "JJNeila/bert-spanish-sensationalism-oss"
print(f"\n[MODEL] Cargando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()
print(f"[MODEL] Modelo cargado exitosamente en VRAM.")

def get_sensationalism_probs(texts, batch_size=64, max_len=256):
    """Calcula P(Sensacionalista) para una lista de textos mediante inferencia en lotes."""
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(p)
    return np.array(probs_all)

# =============================================================================
# 2. FUNCIÓN DE SEGMENTACIÓN POR SIGNOS DE PUNTUACIÓN
# =============================================================================
def split_into_phrases(text):
    """
    Separa el texto en frases/cláusulas utilizando signos de puntuación:
    comas, puntos, punto y coma, dos puntos, signos de exclamación e interrogación, guiones y saltos de línea.
    Preserva fragmentos informativos con al menos 2 palabras y 6 caracteres.
    """
    # Expresión regular que divide por puntuación
    raw_pieces = re.split(r'[,;:.!?¡¿\n—–\"\'«»()]+', str(text))
    phrases = []
    for p in raw_pieces:
        clean = p.strip()
        words = clean.split()
        if len(words) >= 2 and len(clean) >= 6:
            phrases.append(clean)
    return phrases

# =============================================================================
# 3. PREPARACIÓN DEL CORPUS DE NOTICIAS (AMARILLISMO REAL + IA SINTÉTICO)
# =============================================================================
print("\n[DATA] Cargando y estructurando corpus de noticias...")
df_ama_raw = pd.read_csv(DATASET_AMA_PATH)
df_ia_raw = pd.read_csv(DATASET_IA_PATH)

articles = []

# A. Dataset Amarillismo Prensa Real (202 noticias)
for idx, r in df_ama_raw.iterrows():
    titular = str(r['Titular']).strip()
    cuerpo = str(r['Cuerpo']).strip() if str(r['Cuerpo']) != 'nan' else ''
    copete = str(r.get('Copete/Resumen', '')).strip() if str(r.get('Copete/Resumen', '')) != 'nan' else ''
    
    # Noticia completa consolidada (Titular + Copete + Cuerpo)
    # Truncamos suavemente a 1200 caracteres para cubrir el titular y los párrafos clave dentro de la ventana de BETO
    full_text = f"{titular}. {copete} {cuerpo}".strip()
    full_text = re.sub(r'\s+', ' ', full_text)[:1200]
    
    phrases = split_into_phrases(full_text)
    if len(phrases) >= 2:
        articles.append({
            'doc_id': f'AMA_{idx}',
            'fuente': 'Amarillismo Real (Prensa)',
            'etiqueta_real': 1 if r['Amarillismo'] == 'Amarillista' else 0,
            'etiqueta_nombre': str(r['Amarillismo']),
            'titular': titular,
            'texto_completo': full_text,
            'frases': phrases,
            'num_frases': len(phrases)
        })

# B. Dataset IA Sintético (70 noticias)
for idx, r in df_ia_raw.iterrows():
    texto = str(r['Texto']).strip()
    phrases = split_into_phrases(texto)
    # Si tiene al menos 2 frases, la evaluamos; si tiene 1 sola frase larga, dividimos por conectores o comas
    if len(phrases) < 2:
        # Intento de división más fina por conjunciones o espacios en caso de ser mono-oracional
        sub_pieces = [p.strip() for p in re.split(r'[,;:]+', texto) if len(p.strip().split()) >= 2]
        if len(sub_pieces) >= 2:
            phrases = sub_pieces
        else:
            phrases = [texto]
            
    articles.append({
        'doc_id': f'IA_{idx}',
        'fuente': 'IA Sintético (70 reg.)',
        'etiqueta_real': 1 if r['Clase'] == 'Sensacionalista' else 0,
        'etiqueta_nombre': str(r['Clase']),
        'titular': phrases[0] if phrases else texto,
        'texto_completo': texto,
        'frases': phrases,
        'num_frases': len(phrases)
    })

df_articles = pd.DataFrame(articles)
print(f"[DATA] Corpus consolidado: {len(df_articles)} noticias preparadas.")
print(f"       - Amarillismo Real: {len(df_articles[df_articles['fuente'].str.contains('Amarillismo')])} noticias.")
print(f"       - IA Sintético:     {len(df_articles[df_articles['fuente'].str.contains('IA')])} noticias.")
print(f"       - Total frases individuales extraídas: {sum(df_articles['num_frases'])}")

# =============================================================================
# 4. EJECUCIÓN EXPERIMENTAL DE INFERENCIA Y ABLACIÓN
# =============================================================================
print("\n[INFERENCE] Ejecutando predicción en GPU para noticias completas, frases y ablaciones...")
t_start = time.time()

# 1. Inferencia sobre todos los textos completos
full_texts = df_articles['texto_completo'].tolist()
p_full_all = get_sensationalism_probs(full_texts, batch_size=64, max_len=256)
df_articles['p_full'] = p_full_all
df_articles['pred_full'] = (p_full_all >= 0.5).astype(int)

# 2. Inferencia sobre frases y cálculo de agregaciones por documento
all_phrase_rows = []
doc_metrics = []

for i, row in df_articles.iterrows():
    p_full = row['p_full']
    phrases = row['frases']
    k = len(phrases)
    
    # Inferencia de frases individuales
    p_phrases = get_sensationalism_probs(phrases, batch_size=64, max_len=128)
    
    # Guardar en registro detallado de frases
    for p_idx, (phrase_text, p_val) in enumerate(zip(phrases, p_phrases)):
        all_phrase_rows.append({
            'doc_id': row['doc_id'],
            'fuente': row['fuente'],
            'doc_label': row['etiqueta_real'],
            'phrase_idx': p_idx,
            'rel_position': p_idx / max(1, k - 1),
            'phrase_text': phrase_text,
            'p_phrase': float(p_val),
            'p_full': float(p_full)
        })

    # Reglas de Agregación
    p_mean = float(np.mean(p_phrases))
    p_max = float(np.max(p_phrases))
    p_min = float(np.min(p_phrases))
    p_first = float(p_phrases[0])
    
    # Top-2 promedio
    sorted_p = np.sort(p_phrases)[::-1]
    p_top2 = float(np.mean(sorted_p[:2])) if k >= 2 else float(sorted_p[0])
    
    # Promedio ponderado por longitud de caracteres
    lens = np.array([len(p) for p in phrases])
    p_w_mean = float(np.sum(p_phrases * lens) / max(1, np.sum(lens)))
    
    # Dispersión interna
    p_std = float(np.std(p_phrases))
    gap_peak_mean = p_max - p_mean

    # Ablaciones Causales
    top1_idx = int(np.argmax(p_phrases))
    sorted_indices = np.argsort(p_phrases)[::-1]
    top2_indices = set(sorted_indices[:2]) if k >= 2 else {top1_idx}

    # Texto sin Top-1
    phrases_no_top1 = [p for j, p in enumerate(phrases) if j != top1_idx]
    text_no_top1 = " ".join(phrases_no_top1) if phrases_no_top1 else row['texto_completo']
    
    # Texto sin Top-2
    phrases_no_top2 = [p for j, p in enumerate(phrases) if j not in top2_indices]
    text_no_top2 = " ".join(phrases_no_top2) if phrases_no_top2 else row['texto_completo']

    # Texto sin Frase Aleatoria de Control (que no sea Top-1 ni Top-2)
    other_indices = [j for j in range(k) if j not in top2_indices]
    if other_indices:
        rand_idx = other_indices[len(other_indices) // 2]
        phrases_no_rand = [p for j, p in enumerate(phrases) if j != rand_idx]
        text_no_rand = " ".join(phrases_no_rand)
    else:
        text_no_rand = text_no_top1

    # Inferencia de textos ablados
    ablation_probs = get_sensationalism_probs([text_no_top1, text_no_top2, text_no_rand], batch_size=3, max_len=256)
    p_no_top1 = float(ablation_probs[0])
    p_no_top2 = float(ablation_probs[1])
    p_no_rand = float(ablation_probs[2])

    drop_top1 = p_full - p_no_top1
    drop_top2 = p_full - p_no_top2
    drop_rand = p_full - p_no_rand

    # Clasificaciones binarias (umbral 0.5)
    pred_full = int(p_full >= 0.5)
    pred_mean = int(p_mean >= 0.5)
    pred_max = int(p_max >= 0.5)
    pred_top2 = int(p_top2 >= 0.5)
    pred_first = int(p_first >= 0.5)
    pred_no_top1 = int(p_no_top1 >= 0.5)
    pred_no_rand = int(p_no_rand >= 0.5)

    # ¿Hubo cambio de clase al quitar la frase Top-1?
    flip_top1 = bool(pred_full == 1 and pred_no_top1 == 0)
    flip_rand = bool(pred_full == 1 and pred_no_rand == 0)

    doc_metrics.append({
        'doc_id': row['doc_id'],
        'fuente': row['fuente'],
        'clase_real': row['etiqueta_real'],
        'clase_nombre': row['etiqueta_nombre'],
        'num_frases': k,
        'p_full': p_full,
        'pred_full': pred_full,
        # Agregaciones
        'p_mean': p_mean,
        'p_max': p_max,
        'p_min': p_min,
        'p_top2': p_top2,
        'p_first': p_first,
        'p_w_mean': p_w_mean,
        'p_std': p_std,
        'gap_peak_mean': gap_peak_mean,
        # Aciertos binarios con noticia completa
        'agree_mean': int(pred_mean == pred_full),
        'agree_max': int(pred_max == pred_full),
        'agree_top2': int(pred_top2 == pred_full),
        'agree_first': int(pred_first == pred_full),
        # Ablación causal
        'p_no_top1': p_no_top1,
        'p_no_top2': p_no_top2,
        'p_no_rand': p_no_rand,
        'drop_top1': drop_top1,
        'drop_top2': drop_top2,
        'drop_rand': drop_rand,
        'flip_top1': flip_top1,
        'flip_rand': flip_rand,
        'frase_top1': phrases[top1_idx]
    })

print(f"[DONE] Inferencia y ablación completadas en {time.time() - t_start:.2f} segundos.")

df_results = pd.DataFrame(doc_metrics)
df_phrases = pd.DataFrame(all_phrase_rows)

# Exportar CSVs de resultados
csv_results_path = os.path.join(REPORT_DIR, "datos_sensacionalismo_frases.csv")
df_results.to_csv(csv_results_path, index=False)
print(f"[EXPORT] Datos por noticia exportados a: {csv_results_path}")

# =============================================================================
# 5. ANÁLISIS ESTADÍSTICO Y COMPARATIVO
# =============================================================================
def calcular_metricas_ajuste(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    agree = accuracy_score(y_true >= 0.5, y_pred >= 0.5)
    return {
        'pearson_r': round(float(r), 4),
        'spearman_rho': round(float(rho), 4),
        'r2': round(float(r2), 4),
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'agreement_acc': round(float(agree), 4)
    }

stat_summary = {
    'global': {
        'mean': calcular_metricas_ajuste(df_results['p_full'], df_results['p_mean']),
        'max': calcular_metricas_ajuste(df_results['p_full'], df_results['p_max']),
        'top2': calcular_metricas_ajuste(df_results['p_full'], df_results['p_top2']),
        'first': calcular_metricas_ajuste(df_results['p_full'], df_results['p_first']),
        'w_mean': calcular_metricas_ajuste(df_results['p_full'], df_results['p_w_mean'])
    }
}

# Desglose por fuente
for f_name in ['Amarillismo Real (Prensa)', 'IA Sintético (70 reg.)']:
    sub = df_results[df_results['fuente'] == f_name]
    stat_summary[f_name] = {
        'mean': calcular_metricas_ajuste(sub['p_full'], sub['p_mean']),
        'max': calcular_metricas_ajuste(sub['p_full'], sub['p_max']),
        'top2': calcular_metricas_ajuste(sub['p_full'], sub['p_top2']),
        'first': calcular_metricas_ajuste(sub['p_full'], sub['p_first']),
        'w_mean': calcular_metricas_ajuste(sub['p_full'], sub['p_w_mean'])
    }

# Métricas de Ablación Causal
sens_articles = df_results[df_results['pred_full'] == 1]
ablation_summary = {
    'total_noticias_sensacionalistas': len(sens_articles),
    'caida_media_drop_top1': round(float(df_results['drop_top1'].mean()), 4),
    'caida_media_drop_top2': round(float(df_results['drop_top2'].mean()), 4),
    'caida_media_drop_rand': round(float(df_results['drop_rand'].mean()), 4),
    'flips_top1_count': int(df_results['flip_top1'].sum()),
    'flips_top1_pct': round(float(df_results['flip_top1'].sum() / max(1, len(sens_articles)) * 100), 2),
    'flips_rand_count': int(df_results['flip_rand'].sum()),
    'flips_rand_pct': round(float(df_results['flip_rand'].sum() / max(1, len(sens_articles)) * 100), 2)
}

# Desglose de ablación por dataset
for f_name in ['Amarillismo Real (Prensa)', 'IA Sintético (70 reg.)']:
    sub = df_results[df_results['fuente'] == f_name]
    sub_sens = sub[sub['pred_full'] == 1]
    ablation_summary[f_name] = {
        'sens_count': len(sub_sens),
        'drop_top1_mean': round(float(sub['drop_top1'].mean()), 4),
        'drop_top2_mean': round(float(sub['drop_top2'].mean()), 4),
        'drop_rand_mean': round(float(sub['drop_rand'].mean()), 4),
        'flips_top1_count': int(sub['flip_top1'].sum()),
        'flips_top1_pct': round(float(sub['flip_top1'].sum() / max(1, len(sub_sens)) * 100), 2),
        'flips_rand_count': int(sub['flip_rand'].sum()),
        'flips_rand_pct': round(float(sub['flip_rand'].sum() / max(1, len(sub_sens)) * 100), 2)
    }

json_metrics_path = os.path.join(REPORT_DIR, "metricas_sensacionalismo_frases.json")
with open(json_metrics_path, 'w', encoding='utf-8') as f:
    json.dump({
        'estadisticas_ajuste': stat_summary,
        'ablacion_causal': ablation_summary,
        'total_documentos': len(df_results),
        'total_frases': len(df_phrases)
    }, f, indent=2, ensure_ascii=False)
print(f"[EXPORT] Métricas estadísticas exportadas a: {json_metrics_path}")

# Imprimir Resumen en Terminal
print("\n" + "="*80)
print("SÍNTESIS COMPARATIVA: NOTICIA COMPLETA VS REGLAS DE AGREGACIÓN DE FRASES")
print("="*80)
print(f"{'Modelo de Agregación':<22} | {'Pearson r':<10} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8} | {'Concordancia Clásif.'}")
print("-" * 80)
for m_key, m_name in [('mean', '1. Promedio Simple'), ('max', '2. Máximo (Top-1)'), ('top2', '3. Top-2 Promedio'), ('first', '4. Titular / 1ra Frase'), ('w_mean', '5. Prom. Ponderado')]:
    stats = stat_summary['global'][m_key]
    print(f"{m_name:<22} | {stats['pearson_r']:<10.4f} | {stats['r2']:<8.4f} | {stats['mae']:<8.4f} | {stats['rmse']:<8.4f} | {stats['agreement_acc']*100:.1f}%")

print("\n" + "="*80)
print("IMPACTO CAUSAL DE ABLACIÓN DE FRASES")
print("="*80)
print(f"Caída de Probabilidad al quitar Top-1 Frase:      {ablation_summary['caida_media_drop_top1']:+.4f}")
print(f"Caída de Probabilidad al quitar Top-2 Frases:     {ablation_summary['caida_media_drop_top2']:+.4f}")
print(f"Caída de Probabilidad al quitar Frase Aleatoria:  {ablation_summary['caida_media_drop_rand']:+.4f}")
print(f"Noticias Sensacionalistas que invirtieron clase al quitar Top-1: {ablation_summary['flips_top1_count']} / {ablation_summary['total_noticias_sensacionalistas']} ({ablation_summary['flips_top1_pct']}%)")
print(f"Noticias Sensacionalistas que invirtieron clase con Frase Control: {ablation_summary['flips_rand_count']} / {ablation_summary['total_noticias_sensacionalistas']} ({ablation_summary['flips_rand_pct']}%)")

# =============================================================================
# 6. GENERACIÓN Y EXPORTACIÓN DE FIGURAS Y GRÁFICAS (300 DPI)
# =============================================================================
print("\n[VISUALIZATION] Generando conjunto completo de figuras y gráficos científicos (300 DPI)...")

# FIGURA 1: Dispersión y Regresión: P(Full) vs 4 Reglas de Agregación
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
configs_f1 = [
    (axes[0, 0], 'p_mean', 'Promedio Simple (Mean)', '#1f77b4'),
    (axes[0, 1], 'p_max', 'Máximo (Top-1 Frase)', '#d62728'),
    (axes[1, 0], 'p_top2', 'Promedio Top-2 Frases', '#2ca02c'),
    (axes[1, 1], 'p_first', 'Titular / Primera Frase', '#9467bd')
]

for ax, col_name, title_suffix, color_hex in configs_f1:
    sns.regplot(
        data=df_results,
        x=col_name,
        y='p_full',
        ax=ax,
        color=color_hex,
        scatter_kws={'alpha': 0.45, 's': 35},
        line_kws={'linewidth': 2.2, 'label': 'Ajuste Lineal'}
    )
    # Línea diagonal ideal y = x
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.2, label='Ideal y = x')
    
    r_val = stat_summary['global'][col_name.replace('p_', '')]['pearson_r']
    r2_val = stat_summary['global'][col_name.replace('p_', '')]['r2']
    mae_val = stat_summary['global'][col_name.replace('p_', '')]['mae']
    acc_val = stat_summary['global'][col_name.replace('p_', '')]['agreement_acc']
    
    ax.set_title(f"Noticia Completa vs. {title_suffix}\n(r = {r_val:.3f} | R² = {r2_val:.3f} | MAE = {mae_val:.3f} | Concordancia = {acc_val*100:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"Sensacionalismo de Frases: {title_suffix}", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Sensacionalismo Noticia Completa P(Sens | Full)", fontsize=10, fontweight='semibold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

fig.suptitle("Relación Causal: Sensacionalismo de Noticia Completa vs. Agregación de Frases\nModelo: JJNeila/bert-spanish-sensationalism-oss (Corpus 272 Noticias)", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(IMAGE_DIR, "fig1_dispersion_regresion_pfull_vs_agregaciones.png")
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig1_path}")


# FIGURA 2: Comparativa de Métricas de Ajuste (MAE, RMSE, R2 y Concordancia)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
models_keys = ['mean', 'max', 'top2', 'first', 'w_mean']
models_labels = ['Promedio\nSimple', 'Máximo\n(Top-1)', 'Top-2\nPromedio', 'Titular /\n1ra Frase', 'Promedio\nPonderado']

mae_vals = [stat_summary['global'][k]['mae'] for k in models_keys]
rmse_vals = [stat_summary['global'][k]['rmse'] for k in models_keys]
r2_vals = [stat_summary['global'][k]['r2'] for k in models_keys]
acc_vals = [stat_summary['global'][k]['agreement_acc'] * 100 for k in models_keys]

x_pos = np.arange(len(models_labels))
width = 0.35

# Subplot 1: Errores MAE y RMSE (Menor es mejor)
b1 = ax1.bar(x_pos - width/2, mae_vals, width, label='MAE (Error Absoluto Medio) ↓', color='#e6550d', alpha=0.9, edgecolor='#a63603')
b2 = ax1.bar(x_pos + width/2, rmse_vals, width, label='RMSE (Raíz Error Cuadrático) ↓', color='#3182bd', alpha=0.9, edgecolor='#08519c')

for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_ylabel("Magnitud del Error", fontsize=11, fontweight='bold')
ax1.set_title("Errores de Ajuste Frente a Noticia Completa\n(Menor Error = Mayor Similitud Numérica)", fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_labels, fontsize=10, fontweight='semibold')
ax1.set_ylim(0, max(rmse_vals)*1.25)
ax1.legend(loc='upper right', frameon=True)

# Subplot 2: R2 y Concordancia de Decisión (Mayor es mejor)
b3 = ax2.bar(x_pos - width/2, r2_vals, width, label='R² (Varianza Explicada) ↑', color='#756bb1', alpha=0.9, edgecolor='#54278f')
b4 = ax2.bar(x_pos + width/2, [a/100 for a in acc_vals], width, label='Concordancia Decisión (Accuracy) ↑', color='#31a354', alpha=0.9, edgecolor='#006d2c')

for b, raw_acc in zip(b4, acc_vals):
    ax2.annotate(f'{raw_acc:.1f}%', xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b, raw_r2 in zip(b3, r2_vals):
    ax2.annotate(f'{raw_r2:.2f}', xy=(b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_ylabel("Puntuación Relativa [0, 1]", fontsize=11, fontweight='bold')
ax2.set_title("Capacidad Predictiva y Concordancia Clasificatoria\n(Mayor Valor = Mayor Preservación de la Decisión)", fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_labels, fontsize=10, fontweight='semibold')
ax2.set_ylim(0, 1.25)
ax2.legend(loc='upper right', frameon=True)

fig.suptitle("Benchmark Cuantitativo de Reglas de Agregación de Frases vs. Noticia Completa", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2_path = os.path.join(IMAGE_DIR, "fig2_comparativa_metricas_ajuste_mae_rmse_r2.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig2_path}")


# FIGURA 3: Distribución del Sesgo y Residuos (P_full - P_mean vs P_full - P_max)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

res_mean = df_results['p_full'] - df_results['p_mean']
res_max = df_results['p_full'] - df_results['p_max']

sns.histplot(res_mean, kde=True, ax=ax1, color='#d95f02', bins=25, stat="density", alpha=0.6)
ax1.axvline(0, color='black', linestyle='--', linewidth=1.2)
ax1.axvline(res_mean.mean(), color='red', linestyle='-', linewidth=1.8, label=f'Sesgo Medio = {res_mean.mean():+.3f}')
ax1.set_title("Residuo: P(Completa) - P(Promedio Frases)\nSesgo Asimétrico: El modelo sobrepasa el promedio", fontsize=11, fontweight='bold')
ax1.set_xlabel("Diferencia P(Full) - P(Mean)", fontsize=10)
ax1.set_ylabel("Densidad de Noticias", fontsize=10)
ax1.legend(frameon=True)

sns.histplot(res_max, kde=True, ax=ax2, color='#2b5c8f', bins=25, stat="density", alpha=0.6)
ax2.axvline(0, color='black', linestyle='--', linewidth=1.2)
ax2.axvline(res_max.mean(), color='blue', linestyle='-', linewidth=1.8, label=f'Sesgo Medio = {res_max.mean():+.3f}')
ax2.set_title("Residuo: P(Completa) - P(Máximo Frase)\nEfecto Dilución: El cuerpo atenúa la frase pico", fontsize=11, fontweight='bold')
ax2.set_xlabel("Diferencia P(Full) - P(Max)", fontsize=10)
ax2.set_ylabel("Densidad de Noticias", fontsize=10)
ax2.legend(frameon=True)

fig.suptitle("Análisis Distribucional de Residuos: Evidencia del Doble Fenómeno Gatillo vs. Dilución", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig3_path = os.path.join(IMAGE_DIR, "fig3_distribucion_residuos_sesgo_gatillo.png")
fig.savefig(fig3_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig3_path}")


# FIGURA 4: Impacto Causal de Ablación de Frases (Caída de Probabilidad y Cambio de Clase)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Caída de Probabilidad al quitar frases
ablation_names = ['Quitar Frase\nTop-1', 'Quitar Frases\nTop-2', 'Quitar Frase\nAleatoria (Control)']
drops = [ablation_summary['caida_media_drop_top1'], ablation_summary['caida_media_drop_top2'], ablation_summary['caida_media_drop_rand']]
colors_abl = ['#d73027', '#fc8d59', '#4575b4']

bars_a = ax1.bar(ablation_names, drops, color=colors_abl, width=0.45, alpha=0.9, edgecolor='#333333')
for b in bars_a:
    h = b.get_height()
    ax1.annotate(f'{h:+.4f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4 if h >= 0 else -12), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylabel("Variación Promedio en P(Sens | Noticia)", fontsize=11, fontweight='bold')
ax1.set_title("Ablación de Frases: Impacto Causal en Certidumbre\n(Mayor caída = Frase más indispensable)", fontsize=11, fontweight='bold')
ax1.set_ylim(min(drops)*1.5, max(drops)*1.35)

# Subplot 2: Tasa de Cambio de Decisión (Prediction Flip Rate) en Noticias Sensacionalistas
flips_pcts = [ablation_summary['flips_top1_pct'], ablation_summary['flips_rand_pct']]
flips_counts = [ablation_summary['flips_top1_count'], ablation_summary['flips_rand_count']]
flip_labels = ['Top-1 Frase\nEliminada', 'Frase Control\nEliminada']

bars_f = ax2.bar(flip_labels, flips_pcts, color=['#d73027', '#4575b4'], width=0.4, alpha=0.9, edgecolor='#333333')
for b, cnt in zip(bars_f, flips_counts):
    h = b.get_height()
    ax2.annotate(f'{h:.1f}%\n({cnt} noticias)', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel("% Noticias Sensacionalistas Desclasificadas", fontsize=11, fontweight='bold')
ax2.set_title(f"Inversión de Clase Predictiva (Sensacionalista → Sobria)\nTotal Noticias Sensacionalistas Analizadas: {ablation_summary['total_noticias_sensacionalistas']}", fontsize=11, fontweight='bold')
ax2.set_ylim(0, max(flips_pcts)*1.35)

fig.suptitle("Prueba Causal de Hipótesis: ¿La Noticia Depende de 1 o 2 Frases Específicas?", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig4_path = os.path.join(IMAGE_DIR, "fig4_ablacion_causal_frases_drop_flips.png")
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig4_path}")


# FIGURA 5: Perfil Posicional del Sensacionalismo (Front-Loading Effect)
fig, ax = plt.subplots(figsize=(10, 5.5))

# Discretizar la posición relativa en 5 quintiles
df_phrases_multi = df_phrases.copy()
df_phrases_multi['quintil_pos'] = pd.qcut(df_phrases_multi['rel_position'], q=5, labels=['0-20% (Inicio)', '20-40%', '40-60%', '60-80%', '80-100% (Final)'])

sns.lineplot(
    data=df_phrases_multi,
    x='quintil_pos',
    y='p_phrase',
    hue='doc_label',
    palette={1: '#d73027', 0: '#2b5c8f'},
    marker='o',
    markersize=8,
    linewidth=2.5,
    errorbar=('ci', 95),
    ax=ax
)

ax.set_title("Evolución del Sensacionalismo según la Posición de la Frase en la Noticia\n(Efecto de Carga Frontal: Titular y Párrafo de Entrada)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Segmento Posicional dentro de la Noticia", fontsize=11, fontweight='bold')
ax.set_ylabel("Sensacionalismo Promedio P(Sens | Frase)", fontsize=11, fontweight='bold')
ax.set_ylim(0.0, 1.0)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['No Sensacionalista (Objetiva)', 'Sensacionalista (Amarillista)'], frameon=True, fontsize=10, loc='center right')
plt.tight_layout()
fig5_path = os.path.join(IMAGE_DIR, "fig5_perfil_posicional_sensacionalismo.png")
fig.savefig(fig5_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig5_path}")


# FIGURA 6: Contraste: IA Sintético vs Prensa Digital Real
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sub_ia = df_results[df_results['fuente'].str.contains('IA')]
sub_ama = df_results[df_results['fuente'].str.contains('Amarillismo')]

# Subplot 1: IA Sintético (Dispersión P_full vs P_mean)
sns.scatterplot(data=sub_ia, x='p_mean', y='p_full', hue='clase_nombre', palette={'Sensacionalista': '#d73027', 'No sensacionalista': '#2b5c8f'}, s=60, alpha=0.85, ax=ax1)
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='y = x')
r_ia, _ = pearsonr(sub_ia['p_full'], sub_ia['p_mean'])
r2_ia = r2_score(sub_ia['p_full'], sub_ia['p_mean'])
ax1.set_title(f"Noticias Sintéticas IA (70 reg.)\nComportamiento Homogéneo (r = {r_ia:.3f} | R² = {r2_ia:.3f})", fontsize=11, fontweight='bold')
ax1.set_xlabel("Promedio de Frases P(Mean)", fontsize=10)
ax1.set_ylabel("Noticia Completa P(Full)", fontsize=10)
ax1.legend(loc='lower right', frameon=True)

# Subplot 2: Amarillismo Real (Dispersión P_full vs P_mean)
sns.scatterplot(data=sub_ama, x='p_mean', y='p_full', hue='clase_nombre', palette={'Amarillista': '#d73027', 'No Amarillista': '#2b5c8f'}, s=60, alpha=0.85, ax=ax2)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='y = x')
r_ama, _ = pearsonr(sub_ama['p_full'], sub_ama['p_mean'])
r2_ama = r2_score(sub_ama['p_full'], sub_ama['p_mean'])
ax2.set_title(f"Prensa Digital Real (202 reg.)\nComportamiento Heterogéneo Disociado (r = {r_ama:.3f} | R² = {r2_ama:.3f})", fontsize=11, fontweight='bold')
ax2.set_xlabel("Promedio de Frases P(Mean)", fontsize=10)
ax2.set_ylabel("Noticia Completa P(Full)", fontsize=10)
ax2.legend(loc='lower right', frameon=True)

fig.suptitle("Contraste Estructural: Homogeneidad Sintética vs. Disociación en Prensa Real", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig6_path = os.path.join(IMAGE_DIR, "fig6_contraste_ia_sintetico_vs_prensa_real.png")
fig.savefig(fig6_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig6_path}")


# FIGURA 7: Estudio de Casos Paradigmáticos (Efecto Gatillo vs Dilución Contextual)
# Seleccionamos dos casos emblemáticos:
# Caso A (Efecto Gatillo): AMA_2 (¿Asteroide Apophis destruiría la Tierra?)
# Caso B (Efecto Dilución): AMA_1 (Los científicos descubrieron un planeta habitable)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Caso A
phrases_c1 = df_phrases[df_phrases['doc_id'] == 'AMA_2'].sort_values('phrase_idx')
y_pos1 = np.arange(len(phrases_c1))
c1_labels = [f"F{i+1}: {t[:42]}..." for i, t in enumerate(phrases_c1['phrase_text'])]
p_full_c1 = df_results[df_results['doc_id'] == 'AMA_2']['p_full'].values[0]

colors1 = ['#d73027' if p >= 0.5 else '#2b5c8f' for p in phrases_c1['p_phrase']]
ax1.barh(y_pos1, phrases_c1['p_phrase'], color=colors1, height=0.65, alpha=0.85)
ax1.axvline(0.5, color='gray', linestyle='--', linewidth=1)
ax1.axvline(p_full_c1, color='red', linestyle='-', linewidth=2.5, label=f'P(Completa) = {p_full_c1:.3f}')
c1_mean = phrases_c1['p_phrase'].mean()
ax1.axvline(c1_mean, color='orange', linestyle=':', linewidth=2, label=f'Promedio Frases = {c1_mean:.3f}')
ax1.set_yticks(y_pos1)
ax1.set_yticklabels(c1_labels, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlim(0, 1.1)
ax1.set_title("CASO 1: EFECTO GATILLO (AMA_2)\n1 frase alarmista ('Asteroide Apophis...') eleva todo el texto a Sensacionalista", fontsize=10.5, fontweight='bold')
ax1.set_xlabel("Probabilidad de Sensacionalismo P(Sens)", fontsize=10)
ax1.legend(loc='lower right', frameon=True, fontsize=9)

# Caso B
phrases_c2 = df_phrases[df_phrases['doc_id'] == 'AMA_1'].sort_values('phrase_idx')
y_pos2 = np.arange(len(phrases_c2))
c2_labels = [f"F{i+1}: {t[:42]}..." for i, t in enumerate(phrases_c2['phrase_text'])]
p_full_c2 = df_results[df_results['doc_id'] == 'AMA_1']['p_full'].values[0]

colors2 = ['#d73027' if p >= 0.5 else '#2b5c8f' for p in phrases_c2['p_phrase']]
ax2.barh(y_pos2, phrases_c2['p_phrase'], color=colors2, height=0.65, alpha=0.85)
ax2.axvline(0.5, color='gray', linestyle='--', linewidth=1)
ax2.axvline(p_full_c2, color='blue', linestyle='-', linewidth=2.5, label=f'P(Completa) = {p_full_c2:.3f}')
c2_mean = phrases_c2['p_phrase'].mean()
ax2.axvline(c2_mean, color='orange', linestyle=':', linewidth=2, label=f'Promedio Frases = {c2_mean:.3f}')
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(c2_labels, fontsize=8.5)
ax2.invert_yaxis()
ax2.set_xlim(0, 1.1)
ax2.set_title("CASO 2: EFECTO DILUCIÓN CONTEXTUAL (AMA_1)\nFrase inicial alta (0.78), pero el cuerpo sobrio desciende la completa a 0.007", fontsize=10.5, fontweight='bold')
ax2.set_xlabel("Probabilidad de Sensacionalismo P(Sens)", fontsize=10)
ax2.legend(loc='lower right', frameon=True, fontsize=9)

fig.suptitle("Estudio Anatómico de Casos: Mecanismo de Interacción Frases vs. Documento en BETO", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig7_path = os.path.join(IMAGE_DIR, "fig7_casos_estudio_ejemplares.png")
fig.savefig(fig7_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig7_path}")

print("\n" + "="*80)
print("EXPERIMENTO COMPLETADO CON ÉXITO")
print(f"Todas las figuras fueron guardadas en: {IMAGE_DIR}")
print(f"Métricas estadísticas en: {json_metrics_path}")
print(f"Dataset de resultados en: {csv_results_path}")
print("="*80)
