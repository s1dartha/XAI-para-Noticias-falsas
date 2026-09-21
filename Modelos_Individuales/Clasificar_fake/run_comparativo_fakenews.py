import os
import gc
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

print("=== INICIANDO EVALUACIÓN COMPLETA DE DATASET PARA CLASIFICACIÓN DE FAKE NEWS ===")

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.join(script_dir, "reportes")
img_dir = os.path.join(report_dir, "imagenes")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

candidate_dataset_paths = [
    os.path.join(script_dir, "Dataset", "Noticias_entre_70_y_370_palabras (1).xlsx"),
    os.path.join(script_dir, "Noticias_entre_70_y_370_palabras (1).xlsx"),
    "/home/ubuntu/Documentos/Tesis/Datasets/casificar fake/Noticias_entre_70_y_370_palabras (1).xlsx",
    "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras (1).xlsx"
]
dataset_path = next((p for p in candidate_dataset_paths if os.path.exists(p)), candidate_dataset_paths[0])

# 1. Cargar Dataset Completo
df_raw = pd.read_excel(dataset_path)
print(f"Dataset cargado exitosamente. Total de registros: {len(df_raw)}, Columnas: {df_raw.columns.tolist()}")

col_label = 'class' if 'class' in df_raw.columns else ('PRED_LABEL' if 'PRED_LABEL' in df_raw.columns else 'label')
col_text = 'Text' if 'Text' in df_raw.columns else ('Texto' if 'Texto' in df_raw.columns else df_raw.columns[1])

df_raw['label_num'] = df_raw[col_label].map({False: 0, True: 1, 'REAL': 0, 'FAKE': 1, 0: 0, 1: 1}).fillna(0).astype(int)
df_sample = df_raw.copy().reset_index(drop=True)
y_true = df_sample['label_num'].values
texts = df_sample[col_text].astype(str).tolist()

print(f"Total de noticias a evaluar: {len(df_sample)}")
print(f"Distribución de etiquetas (0 = Verdadera, 1 = Falsa): {np.bincount(y_true)}")

results = {}

# ==========================================
# MODELO 1: SaBERT (JJNeila/bert-spanish-sensationalism-oss)
# ==========================================
print("\n" + "="*60)
print(">>> [1/3] EVALUANDO MODELO 1: SaBERT Baseline (JJNeila/bert-spanish-sensationalism-oss)...")
print("="*60)

t0 = time.time()
tokenizer_m1 = AutoTokenizer.from_pretrained("JJNeila/bert-spanish-sensationalism-oss")
model_m1 = AutoModelForSequenceClassification.from_pretrained("JJNeila/bert-spanish-sensationalism-oss")
model_m1.eval()

pred_m1 = []
batch_size = 32
with torch.inference_mode():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer_m1(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        logits = model_m1(**inputs).logits
        p = torch.argmax(logits, dim=-1).tolist()
        pred_m1.extend(p)
        if (i // batch_size) % 10 == 0 or (i + batch_size >= len(texts)):
            print(f"  Procesados {min(i+batch_size, len(texts))}/{len(texts)}...")

time_m1 = time.time() - t0
lat_m1 = (time_m1 / len(texts)) * 1000

print(f"✓ Modelo 1 evaluado en {time_m1:.2f} s ({lat_m1:.2f} ms/muestra)")
print("\n=== REPORTE DE CLASIFICACIÓN: SaBERT ===")
print(classification_report(y_true, pred_m1, target_names=['Verdadera (0)', 'Falsa (1)'], digits=4))

cm_m1 = confusion_matrix(y_true, pred_m1)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusión - SaBERT (JJNeila)')
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m1_sabert_fakenews.png'), dpi=300)
fig.savefig(os.path.join(img_dir, 'cm_m1_fakenews.png'), dpi=300)
plt.close(fig)

acc_m1 = accuracy_score(y_true, pred_m1)
prec_m1, rec_m1, f1_m1, _ = precision_recall_fscore_support(y_true, pred_m1, average="macro", zero_division=0)
results['SaBERT'] = {
    'Accuracy': acc_m1, 'Precision': prec_m1, 'Recall': rec_m1, 'F1-Score': f1_m1,
    'Tiempo Total (s)': time_m1, 'Latencia (ms)': lat_m1, 'CM': cm_m1.tolist()
}

del model_m1, tokenizer_m1
gc.collect()
time.sleep(10)

# ==========================================
# MODELO 2: BETO Fake News (Narrativaai/fake-news-detection-spanish)
# ==========================================
print("\n" + "="*60)
print(">>> [2/3] EVALUANDO MODELO 2: BETO / RoBERTa Fake News (Narrativaai/fake-news-detection-spanish)...")
print("="*60)

t0 = time.time()
tokenizer_m2 = AutoTokenizer.from_pretrained("Narrativaai/fake-news-detection-spanish")
model_m2 = AutoModelForSequenceClassification.from_pretrained("Narrativaai/fake-news-detection-spanish")
model_m2.eval()

# Check label mapping in model config
id2label_m2 = model_m2.config.id2label
print(f"Label mapping M2: {id2label_m2}")

pred_m2 = []
batch_size = 16
with torch.inference_mode():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer_m2(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        logits = model_m2(**inputs).logits
        p = torch.argmax(logits, dim=-1).tolist()
        pred_m2.extend(p)
        if (i // batch_size) % 10 == 0 or (i + batch_size >= len(texts)):
            print(f"  Procesados {min(i+batch_size, len(texts))}/{len(texts)}...")

# If id2label indicates 0=FAKE or 1=FAKE, ensure correct alignment
# Typically in Narrativaai: label 0: REAL, label 1: FAKE or vice versa
# Let's map according to label name if available:
fake_id = 1
for k, v in id2label_m2.items():
    if 'fake' in str(v).lower() or 'fals' in str(v).lower():
        fake_id = int(k)
        break

if fake_id == 0:
    pred_m2 = [1 if p == 0 else 0 for p in pred_m2]

time_m2 = time.time() - t0
lat_m2 = (time_m2 / len(texts)) * 1000

print(f"✓ Modelo 2 evaluado en {time_m2:.2f} s ({lat_m2:.2f} ms/muestra)")
print("\n=== REPORTE DE CLASIFICACIÓN: BETO Fake News ===")
print(classification_report(y_true, pred_m2, target_names=['Verdadera (0)', 'Falsa (1)'], digits=4))

cm_m2 = confusion_matrix(y_true, pred_m2)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Greens', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusión - BETO Fake News (Narrativaai)')
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m2_beto_fakenews.png'), dpi=300)
fig.savefig(os.path.join(img_dir, 'cm_m2_fakenews.png'), dpi=300)
plt.close(fig)

acc_m2 = accuracy_score(y_true, pred_m2)
prec_m2, rec_m2, f1_m2, _ = precision_recall_fscore_support(y_true, pred_m2, average="macro", zero_division=0)
results['BETO'] = {
    'Accuracy': acc_m2, 'Precision': prec_m2, 'Recall': rec_m2, 'F1-Score': f1_m2,
    'Tiempo Total (s)': time_m2, 'Latencia (ms)': lat_m2, 'CM': cm_m2.tolist()
}

del model_m2, tokenizer_m2
gc.collect()
time.sleep(10)

# ==========================================
# MODELO 3: LLM Zero-Shot Fact-Checker (Qwen/Qwen2.5-1.5B-Instruct)
# ==========================================
print("\n" + "="*60)
print(">>> [3/3] EVALUANDO MODELO 3: LLM Zero-Shot Fact-Checker (Qwen/Qwen2.5-1.5B-Instruct)...")
print("="*60)

t0 = time.time()
tokenizer_m3 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model_m3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model_m3.eval()
tokenizer_m3.pad_token = tokenizer_m3.eos_token

prompts = [
    f"Actúa como un experto verificador de datos (fact-checker). Lee la siguiente noticia y responde únicamente con la palabra FALSA o VERDADERA.\nNoticia: {t[:300]}\nRespuesta:"
    for t in texts
]

token_falsa = tokenizer_m3.encode("FALSA", add_special_tokens=False)[0]
token_verdadera = tokenizer_m3.encode("VERDADERA", add_special_tokens=False)[0]

pred_m3 = []
batch_size = 16
with torch.inference_mode():
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i:i+batch_size]
        inputs = tokenizer_m3(batch_p, padding=True, truncation=True, max_length=256, return_tensors='pt')
        outputs = model_m3(**inputs)
        logits = outputs.logits[:, -1, :]
        p_f = logits[:, token_falsa]
        p_v = logits[:, token_verdadera]
        preds = (p_f > p_v).long().tolist()
        pred_m3.extend(preds)
        if (i // batch_size) % 10 == 0 or (i + batch_size >= len(prompts)):
            print(f"  Procesados {min(i+batch_size, len(prompts))}/{len(prompts)}...")

time_m3 = time.time() - t0
lat_m3 = (time_m3 / len(texts)) * 1000

print(f"✓ Modelo 3 evaluado en {time_m3:.2f} s ({lat_m3:.2f} ms/muestra)")
print("\n=== REPORTE DE CLASIFICACIÓN: LLM Zero-Shot Fact-Checker ===")
print(classification_report(y_true, pred_m3, target_names=['Verdadera (0)', 'Falsa (1)'], digits=4))

cm_m3 = confusion_matrix(y_true, pred_m3)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(cm_m3, annot=True, fmt='d', cmap='Purples', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusión - LLM Zero-Shot Fact-Checker (Qwen 1.5B)')
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m3_llm_fakenews.png'), dpi=300)
fig.savefig(os.path.join(img_dir, 'cm_m3_fakenews.png'), dpi=300)
plt.close(fig)

acc_m3 = accuracy_score(y_true, pred_m3)
prec_m3, rec_m3, f1_m3, _ = precision_recall_fscore_support(y_true, pred_m3, average="macro", zero_division=0)
results['LLM'] = {
    'Accuracy': acc_m3, 'Precision': prec_m3, 'Recall': rec_m3, 'F1-Score': f1_m3,
    'Tiempo Total (s)': time_m3, 'Latencia (ms)': lat_m3, 'CM': cm_m3.tolist()
}

del model_m3, tokenizer_m3
gc.collect()

# ==========================================
# TABLA Y GRÁFICOS COMPARATIVOS
# ==========================================
print("\n" + "="*60)
print(">>> GENERANDO TABLA Y GRÁFICOS COMPARATIVOS...")
print("="*60)

tabla_df = pd.DataFrame([
    {
        "Modelo": "1. SaBERT Baseline (JJNeila)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['SaBERT']['Accuracy'], 4),
        "Precision (Macro)": round(results['SaBERT']['Precision'], 4),
        "Recall (Macro)": round(results['SaBERT']['Recall'], 4),
        "F1-Score (Macro)": round(results['SaBERT']['F1-Score'], 4),
        "Tiempo Total (s)": round(results['SaBERT']['Tiempo Total (s)'], 2),
        "Latencia por Muestra (ms)": round(results['SaBERT']['Latencia (ms)'], 2)
    },
    {
        "Modelo": "2. BETO Fake News (Narrativaai)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['BETO']['Accuracy'], 4),
        "Precision (Macro)": round(results['BETO']['Precision'], 4),
        "Recall (Macro)": round(results['BETO']['Recall'], 4),
        "F1-Score (Macro)": round(results['BETO']['F1-Score'], 4),
        "Tiempo Total (s)": round(results['BETO']['Tiempo Total (s)'], 2),
        "Latencia por Muestra (ms)": round(results['BETO']['Latencia (ms)'], 2)
    },
    {
        "Modelo": "3. LLM Zero-Shot Fact-Checker (Qwen)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['LLM']['Accuracy'], 4),
        "Precision (Macro)": round(results['LLM']['Precision'], 4),
        "Recall (Macro)": round(results['LLM']['Recall'], 4),
        "F1-Score (Macro)": round(results['LLM']['F1-Score'], 4),
        "Tiempo Total (s)": round(results['LLM']['Tiempo Total (s)'], 2),
        "Latencia por Muestra (ms)": round(results['LLM']['Latencia (ms)'], 2)
    }
])

print("\n=== TABLA COMPARATIVA GENERAL ===")
print(tabla_df.to_string(index=False))

# Guardar resultados en JSON para reproducibilidad
metricas_json_path = os.path.join(report_dir, "metricas_fakenews.json")
with open(metricas_json_path, "w") as f:
    json.dump(results, f, indent=4)

# Gráfico Comparativo de Métricas
modelos = ["SaBERT (JJNeila)", "BETO (Narrativaai)", "LLM Zero-Shot (Qwen)"]
x = np.arange(len(modelos))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, tabla_df["Accuracy"], width, label="Accuracy", color="#1f77b4")
rects2 = ax.bar(x - 0.5*width, tabla_df["Precision (Macro)"], width, label="Precision (Macro)", color="#2ca02c")
rects3 = ax.bar(x + 0.5*width, tabla_df["Recall (Macro)"], width, label="Recall (Macro)", color="#ff7f0e")
rects4 = ax.bar(x + 1.5*width, tabla_df["F1-Score (Macro)"], width, label="F1-Score (Macro)", color="#d62728")

ax.set_ylabel("Puntuación (0.0 - 1.0)", fontsize=12)
ax.set_title("Comparativa de Rendimiento en Detección de Fake News (Dataset Completo 2.604 Reg.)", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=11)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "comparativa_metricas_fakenews.png"), dpi=300)
plt.close(fig)

# Gráfico Comparativo de Tiempos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(modelos, tabla_df["Tiempo Total (s)"], color=['#1f77b4', '#2ca02c', '#9467bd'])
ax1.set_title("Tiempo Total de Inferencia (s)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Segundos", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(tabla_df["Tiempo Total (s)"]):
    ax1.text(i, v + (v*0.02), f"{v:.1f}s", ha='center', fontweight='bold')

ax2.bar(modelos, tabla_df["Latencia por Muestra (ms)"], color=['#1f77b4', '#2ca02c', '#9467bd'])
ax2.set_title("Latencia por Muestra (ms)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Milisegundos (ms)", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(tabla_df["Latencia por Muestra (ms)"]):
    ax2.text(i, v + (v*0.02), f"{v:.1f}ms", ha='center', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "comparativa_tiempos_fakenews.png"), dpi=300)
plt.close(fig)

print("\n✓ Todos los gráficos y métricas guardados con éxito.")
