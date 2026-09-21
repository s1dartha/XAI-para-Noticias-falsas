import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)

print("="*80, flush=True)
print(" SISTEMA DE EVALUACION EXPERIMENTAL: 4 MODELOS PARA DETECCION DE FAKE NEWS", flush=True)
print(f" Hilos CPU asignados: {torch.get_num_threads()}", flush=True)
print("="*80, flush=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
report_dir = os.path.join(script_dir, "reportes")
img_dir = os.path.join(report_dir, "clasificar_model_4_imagenes")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
print(f"[INFO] Directorio de graficos: {img_dir}", flush=True)

potential_paths = [
    os.path.join(script_dir, "Dataset", "Noticias_entre_70_y_370_palabras (1).xlsx"),
    os.path.join(script_dir, "Noticias_entre_70_y_370_palabras (1).xlsx"),
    os.path.join(project_root, "Datasets", "casificar fake", "Noticias_entre_70_y_370_palabras (1).xlsx"),
    os.path.join(project_root, "Datasets", "Noticias_entre_70_y_370_palabras (1).xlsx"),
    "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras (1).xlsx",
    "Noticias_entre_70_y_370_palabras (1).xlsx"
]

dataset_path = None
for p in potential_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if not dataset_path:
    raise FileNotFoundError("No se encontro el dataset en las rutas esperadas.")

print(f"[INFO] Dataset localizado en: {dataset_path}", flush=True)

df_raw = pd.read_excel(dataset_path)
col_label = 'class' if 'class' in df_raw.columns else ('PRED_LABEL' if 'PRED_LABEL' in df_raw.columns else 'label')
col_text = 'Text' if 'Text' in df_raw.columns else ('Texto' if 'Texto' in df_raw.columns else df_raw.columns[1])

df_raw['label_num'] = df_raw[col_label].map({False: 0, True: 1, 'REAL': 0, 'FAKE': 1, 0: 0, 1: 1}).fillna(0).astype(int)

y_true = df_raw['label_num'].values
texts = df_raw[col_text].astype(str).tolist()
counts = np.bincount(y_true)

print(f"[INFO] Dataset: {len(df_raw)} registros. Clase 0 (Verdadera): {counts[0]} | Clase 1 (Falsa): {counts[1]}", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Dispositivo: {device}", flush=True)

cache_file = os.path.join(script_dir, "predictions_cache.npz")
cached_data = {}
if os.path.exists(cache_file):
    try:
        cached_data = dict(np.load(cache_file, allow_pickle=True))
        print(f"[INFO] Cargada cache previa con modelos: {list(cached_data.keys())}", flush=True)
    except Exception as e:
        print(f"[WARN] No se pudo leer cache previa: {e}", flush=True)

results = {}
predictions_raw = {}

# =========================================================================
# MODELO 1: BETO / RoBERTa Fake News (Narrativaai/fake-news-detection-spanish)
# =========================================================================
m1_name = "Narrativaai/fake-news-detection-spanish"
print("\n" + "="*80, flush=True)
print(f">>> [1/4] EVALUANDO MODELO 1: BETO / RoBERTa Fake News ({m1_name})...", flush=True)
print("="*80, flush=True)

if 'prob_m1' in cached_data and 'pred_m1' in cached_data and 'time_m1' in cached_data:
    print("[INFO] Cargando predicciones de M1 desde cache...", flush=True)
    prob_m1 = cached_data['prob_m1'].tolist()
    pred_m1 = cached_data['pred_m1'].tolist()
    time_m1 = float(cached_data['time_m1'])
else:
    t0 = time.time()
    tok_m1 = AutoTokenizer.from_pretrained(m1_name)
    model_m1 = AutoModelForSequenceClassification.from_pretrained(m1_name)
    model_m1.to(device)
    model_m1.eval()
    pred_m1, prob_m1 = [], []
    batch_size_m1 = 32
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size_m1):
            batch = texts[i:i+batch_size_m1]
            inputs = tok_m1(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model_m1(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            p_fake = probs[:, 1].cpu().tolist()
            p_class = torch.argmax(logits, dim=-1).cpu().tolist()
            pred_m1.extend(p_class)
            prob_m1.extend(p_fake)
            if (i // batch_size_m1) % 15 == 0 or (i + batch_size_m1 >= len(texts)):
                print(f"  [M1 - Narrativaai] Procesados {min(i+batch_size_m1, len(texts))}/{len(texts)}...", flush=True)
    time_m1 = time.time() - t0
    del model_m1, tok_m1
    if torch.cuda.is_available(): torch.cuda.empty_cache()

lat_m1 = (time_m1 / len(texts)) * 1000
acc_m1 = accuracy_score(y_true, pred_m1)
prec_m1, rec_m1, f1_m1, _ = precision_recall_fscore_support(y_true, pred_m1, average="macro", zero_division=0)
fpr_m1, tpr_m1, _ = roc_curve(y_true, prob_m1)
auc_m1 = auc(fpr_m1, tpr_m1)
pr_rec_m1, pr_prec_m1, _ = precision_recall_curve(y_true, prob_m1)
ap_m1 = average_precision_score(y_true, prob_m1)
cm_m1 = confusion_matrix(y_true, pred_m1)

results['BETO_Narrativaai'] = {
    'Nombre': 'BETO Fake News (Narrativaai)',
    'ID': m1_name,
    'Accuracy': acc_m1, 'Precision': prec_m1, 'Recall': rec_m1, 'F1-Score': f1_m1, 'AUC': auc_m1, 'AP': ap_m1,
    'Tiempo Total (s)': time_m1, 'Latencia (ms)': lat_m1, 'CM': cm_m1.tolist()
}
predictions_raw['BETO_Narrativaai'] = {'y_pred': pred_m1, 'y_prob': prob_m1}

print(f"[OK] M1 evaluado en {time_m1:.2f}s ({lat_m1:.2f} ms/muestra)", flush=True)
print(f"  Accuracy: {acc_m1:.4f} | F1-Score: {f1_m1:.4f} | AUC-ROC: {auc_m1:.4f}", flush=True)

# Guardar matriz M1
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusion - BETO Fake News (Narrativaai)', fontsize=12, fontweight='bold')
ax.set_xlabel('Prediccion del Modelo')
ax.set_ylabel('Etiqueta Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m1_beto_fakenews.png'), dpi=300)
plt.close(fig)

# =========================================================================
# MODELO 2: SaBERT (VerificadoProfesional/SaBERT-Spanish-Fake-News)
# =========================================================================
m2_name = "VerificadoProfesional/SaBERT-Spanish-Fake-News"
print("\n" + "="*80, flush=True)
print(f">>> [2/4] EVALUANDO MODELO 2: SaBERT ({m2_name})...", flush=True)
print("="*80, flush=True)

if 'prob_m2' in cached_data and 'pred_m2' in cached_data and 'time_m2' in cached_data:
    print("[INFO] Cargando predicciones de M2 desde cache...", flush=True)
    prob_m2 = cached_data['prob_m2'].tolist()
    pred_m2 = cached_data['pred_m2'].tolist()
    time_m2 = float(cached_data['time_m2'])
else:
    t0 = time.time()
    tok_m2 = AutoTokenizer.from_pretrained(m2_name)
    model_m2 = AutoModelForSequenceClassification.from_pretrained(m2_name)
    model_m2.to(device)
    model_m2.eval()
    pred_m2, prob_m2 = [], []
    batch_size_m2 = 32
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size_m2):
            batch = texts[i:i+batch_size_m2]
            inputs = tok_m2(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model_m2(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            p_fake = probs[:, 0].cpu().tolist()
            p_class = (probs[:, 0] > 0.5).long().cpu().tolist()
            pred_m2.extend(p_class)
            prob_m2.extend(p_fake)
            if (i // batch_size_m2) % 15 == 0 or (i + batch_size_m2 >= len(texts)):
                print(f"  [M2 - SaBERT] Procesados {min(i+batch_size_m2, len(texts))}/{len(texts)}...", flush=True)
    time_m2 = time.time() - t0
    del model_m2, tok_m2
    if torch.cuda.is_available(): torch.cuda.empty_cache()

lat_m2 = (time_m2 / len(texts)) * 1000
acc_m2 = accuracy_score(y_true, pred_m2)
prec_m2, rec_m2, f1_m2, _ = precision_recall_fscore_support(y_true, pred_m2, average="macro", zero_division=0)
fpr_m2, tpr_m2, _ = roc_curve(y_true, prob_m2)
auc_m2 = auc(fpr_m2, tpr_m2)
pr_rec_m2, pr_prec_m2, _ = precision_recall_curve(y_true, prob_m2)
ap_m2 = average_precision_score(y_true, prob_m2)
cm_m2 = confusion_matrix(y_true, pred_m2)

results['SaBERT_Verificado'] = {
    'Nombre': 'SaBERT (VerificadoProfesional)',
    'ID': m2_name,
    'Accuracy': acc_m2, 'Precision': prec_m2, 'Recall': rec_m2, 'F1-Score': f1_m2, 'AUC': auc_m2, 'AP': ap_m2,
    'Tiempo Total (s)': time_m2, 'Latencia (ms)': lat_m2, 'CM': cm_m2.tolist()
}
predictions_raw['SaBERT_Verificado'] = {'y_pred': pred_m2, 'y_prob': prob_m2}

print(f"[OK] M2 evaluado en {time_m2:.2f}s ({lat_m2:.2f} ms/muestra)", flush=True)
print(f"  Accuracy: {acc_m2:.4f} | F1-Score: {f1_m2:.4f} | AUC-ROC: {auc_m2:.4f}", flush=True)

# Guardar matriz M2
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Greens', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusion - SaBERT (VerificadoProfesional)', fontsize=12, fontweight='bold')
ax.set_xlabel('Prediccion del Modelo')
ax.set_ylabel('Etiqueta Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m2_sabert_fakenews.png'), dpi=300)
plt.close(fig)

# =========================================================================
# MODELO 3: Spanish Fake News Classifier (Juanillaberia/spanish-fake-news-classifier)
# =========================================================================
m3_name = "Juanillaberia/spanish-fake-news-classifier"
print("\n" + "="*80, flush=True)
print(f">>> [3/4] EVALUANDO MODELO 3: Spanish Fake News Classifier ({m3_name})...", flush=True)
print("="*80, flush=True)

if 'prob_m3' in cached_data and 'pred_m3' in cached_data and 'time_m3' in cached_data:
    print("[INFO] Cargando predicciones de M3 desde cache...", flush=True)
    prob_m3 = cached_data['prob_m3'].tolist()
    pred_m3 = cached_data['pred_m3'].tolist()
    time_m3 = float(cached_data['time_m3'])
else:
    t0 = time.time()
    tok_m3 = AutoTokenizer.from_pretrained(m3_name)
    model_m3 = AutoModelForSequenceClassification.from_pretrained(m3_name)
    model_m3.to(device)
    model_m3.eval()
    pred_m3, prob_m3 = [], []
    batch_size_m3 = 32
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size_m3):
            batch = texts[i:i+batch_size_m3]
            inputs = tok_m3(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model_m3(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            p_fake = probs[:, 0].cpu().tolist()
            p_class = (probs[:, 0] > 0.5).long().cpu().tolist()
            pred_m3.extend(p_class)
            prob_m3.extend(p_fake)
            if (i // batch_size_m3) % 15 == 0 or (i + batch_size_m3 >= len(texts)):
                print(f"  [M3 - Juanillaberia] Procesados {min(i+batch_size_m3, len(texts))}/{len(texts)}...", flush=True)
    time_m3 = time.time() - t0
    del model_m3, tok_m3
    if torch.cuda.is_available(): torch.cuda.empty_cache()

lat_m3 = (time_m3 / len(texts)) * 1000
acc_m3 = accuracy_score(y_true, pred_m3)
prec_m3, rec_m3, f1_m3, _ = precision_recall_fscore_support(y_true, pred_m3, average="macro", zero_division=0)
fpr_m3, tpr_m3, _ = roc_curve(y_true, prob_m3)
auc_m3 = auc(fpr_m3, tpr_m3)
pr_rec_m3, pr_prec_m3, _ = precision_recall_curve(y_true, prob_m3)
ap_m3 = average_precision_score(y_true, prob_m3)
cm_m3 = confusion_matrix(y_true, pred_m3)

results['Spanish_FakeNews_Juanillaberia'] = {
    'Nombre': 'Spanish Fake News (Juanillaberia)',
    'ID': m3_name,
    'Accuracy': acc_m3, 'Precision': prec_m3, 'Recall': rec_m3, 'F1-Score': f1_m3, 'AUC': auc_m3, 'AP': ap_m3,
    'Tiempo Total (s)': time_m3, 'Latencia (ms)': lat_m3, 'CM': cm_m3.tolist()
}
predictions_raw['Spanish_FakeNews_Juanillaberia'] = {'y_pred': pred_m3, 'y_prob': prob_m3}

print(f"[OK] M3 evaluado en {time_m3:.2f}s ({lat_m3:.2f} ms/muestra)", flush=True)
print(f"  Accuracy: {acc_m3:.4f} | F1-Score: {f1_m3:.4f} | AUC-ROC: {auc_m3:.4f}", flush=True)

# Guardar matriz M3
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_m3, annot=True, fmt='d', cmap='Oranges', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusion - Spanish Fake News (Juanillaberia)', fontsize=12, fontweight='bold')
ax.set_xlabel('Prediccion del Modelo')
ax.set_ylabel('Etiqueta Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m3_juanillaberia_fakenews.png'), dpi=300)
plt.close(fig)

# =========================================================================
# MODELO 4: LLM Zero-Shot (Qwen/Qwen2.5-1.5B-Instruct)
# =========================================================================
m4_name = "Qwen/Qwen2.5-1.5B-Instruct"
print("\n" + "="*80, flush=True)
print(f">>> [4/4] EVALUANDO MODELO 4: LLM Zero-Shot Fact-Checker ({m4_name})...", flush=True)
print("="*80, flush=True)

t0 = time.time()
tok_m4 = AutoTokenizer.from_pretrained(m4_name, padding_side='left')
if tok_m4.pad_token is None:
    tok_m4.pad_token = tok_m4.eos_token

model_m4 = AutoModelForCausalLM.from_pretrained(m4_name, torch_dtype=torch.float32)
model_m4.to(device)
model_m4.eval()

tok_fake_space = tok_m4.encode(' Fake', add_special_tokens=False)[0]
tok_real_space = tok_m4.encode(' Real', add_special_tokens=False)[0]
tok_fake_nospace = tok_m4.encode('Fake', add_special_tokens=False)[0]
tok_real_nospace = tok_m4.encode('Real', add_special_tokens=False)[0]

prompts = [
    f"<|im_start|>system\nEres un modelo experto en verificacion de datos y deteccion de desinformacion. Clasifica noticias como Real o Fake.<|im_end|>\n<|im_start|>user\nDetermina si la siguiente noticia es Real o Fake. Responde unicamente con una palabra: Real o Fake.\nNoticia: {t[:350]}<|im_end|>\n<|im_start|>assistant\n"
    for t in texts
]

pred_m4 = []
prob_m4 = []
batch_size_m4 = 16

with torch.inference_mode():
    for i in range(0, len(prompts), batch_size_m4):
        batch_p = prompts[i:i+batch_size_m4]
        inputs = tok_m4(batch_p, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        outputs = model_m4(**inputs)
        logits = outputs.logits[:, -1, :]
        
        l_f = torch.maximum(logits[:, tok_fake_space], logits[:, tok_fake_nospace])
        l_r = torch.maximum(logits[:, tok_real_space], logits[:, tok_real_nospace])
        
        p_fake = torch.sigmoid(l_f - l_r).cpu().tolist()
        p_class = (l_f > l_r).long().cpu().tolist()
        
        pred_m4.extend(p_class)
        prob_m4.extend(p_fake)
        
        if (i // batch_size_m4) % 10 == 0 or (i + batch_size_m4 >= len(prompts)):
            print(f"  [M4 - Qwen 1.5B] Procesados {min(i+batch_size_m4, len(prompts))}/{len(prompts)}...", flush=True)

time_m4 = time.time() - t0
lat_m4 = (time_m4 / len(texts)) * 1000

acc_m4 = accuracy_score(y_true, pred_m4)
prec_m4, rec_m4, f1_m4, _ = precision_recall_fscore_support(y_true, pred_m4, average="macro", zero_division=0)
fpr_m4, tpr_m4, _ = roc_curve(y_true, prob_m4)
auc_m4 = auc(fpr_m4, tpr_m4)
pr_rec_m4, pr_prec_m4, _ = precision_recall_curve(y_true, prob_m4)
ap_m4 = average_precision_score(y_true, prob_m4)
cm_m4 = confusion_matrix(y_true, pred_m4)

results['Qwen2.5_1.5B_ZeroShot'] = {
    'Nombre': 'LLM Zero-Shot (Qwen 1.5B)',
    'ID': m4_name,
    'Accuracy': acc_m4, 'Precision': prec_m4, 'Recall': rec_m4, 'F1-Score': f1_m4, 'AUC': auc_m4, 'AP': ap_m4,
    'Tiempo Total (s)': time_m4, 'Latencia (ms)': lat_m4, 'CM': cm_m4.tolist()
}
predictions_raw['Qwen2.5_1.5B_ZeroShot'] = {'y_pred': pred_m4, 'y_prob': prob_m4}

print(f"\n[OK] M4 evaluado en {time_m4:.2f}s ({lat_m4:.2f} ms/muestra)", flush=True)
print(f"  Accuracy: {acc_m4:.4f} | F1-Score: {f1_m4:.4f} | AUC-ROC: {auc_m4:.4f}", flush=True)
print("\nReporte de Clasificacion (M4 - Qwen Zero-Shot):", flush=True)
print(classification_report(y_true, pred_m4, target_names=['Verdadera (0)', 'Falsa (1)'], digits=4), flush=True)

# Guardar matriz M4
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_m4, annot=True, fmt='d', cmap='Purples', xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'], yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
ax.set_title('Matriz de Confusion - LLM Zero-Shot (Qwen2.5-1.5B)', fontsize=12, fontweight='bold')
ax.set_xlabel('Prediccion del Modelo')
ax.set_ylabel('Etiqueta Real')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'cm_m4_qwen_fakenews.png'), dpi=300)
plt.close(fig)

del model_m4, tok_m4
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Guardar cache de predicciones completa
np.savez_compressed(
    cache_file,
    prob_m1=prob_m1, pred_m1=pred_m1, time_m1=time_m1,
    prob_m2=prob_m2, pred_m2=pred_m2, time_m2=time_m2,
    prob_m3=prob_m3, pred_m3=pred_m3, time_m3=time_m3,
    prob_m4=prob_m4, pred_m4=pred_m4, time_m4=time_m4,
    y_true=y_true
)

# =========================================================================
# 5. GENERACION DE GRAFICOS INTEGRADOS Y CONSOLIDADOS
# =========================================================================
print("\n" + "="*80, flush=True)
print(">>> GENERANDO GRAFICOS CONSOLIDADOS Y COMPARATIVAS...", flush=True)
print("="*80, flush=True)

# 5.1 Matriz de Confusion 2x2 Conjunta
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
cms = [
    (cm_m1, 'BETO Fake News (Narrativaai)', 'Blues', axes[0, 0]),
    (cm_m2, 'SaBERT (VerificadoProfesional)', 'Greens', axes[0, 1]),
    (cm_m3, 'Spanish Fake News (Juanillaberia)', 'Oranges', axes[1, 0]),
    (cm_m4, 'LLM Zero-Shot (Qwen2.5-1.5B)', 'Purples', axes[1, 1])
]

for cm, title, cmap, ax in cms:
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Pred Verdadera (0)', 'Pred Falsa (1)'],
                yticklabels=['Real Verdadera (0)', 'Real Falsa (1)'], ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Valor Real')

plt.suptitle('Comparativa de Matrices de Confusion (Dataset Completo: 2.604 Noticias)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'confusion_matrices_all_4_models.png'), dpi=300)
plt.close(fig)

# 5.2 Tabla Resumen de Metricas
modelos_labels = [
    "BETO (Narrativaai)",
    "SaBERT (VerificadoProf)",
    "BERT Seq (Juanillaberia)",
    "LLM Zero-Shot (Qwen 1.5B)"
]

metricas_df = pd.DataFrame([
    {
        "Modelo": "1. BETO Fake News (Narrativaai)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['BETO_Narrativaai']['Accuracy'], 4),
        "Precision (Macro)": round(results['BETO_Narrativaai']['Precision'], 4),
        "Recall (Macro)": round(results['BETO_Narrativaai']['Recall'], 4),
        "F1-Score (Macro)": round(results['BETO_Narrativaai']['F1-Score'], 4),
        "ROC-AUC": round(results['BETO_Narrativaai']['AUC'], 4),
        "Tiempo Total (s)": round(results['BETO_Narrativaai']['Tiempo Total (s)'], 2),
        "Latencia (ms/m)": round(results['BETO_Narrativaai']['Latencia (ms)'], 2)
    },
    {
        "Modelo": "2. SaBERT (VerificadoProfesional)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['SaBERT_Verificado']['Accuracy'], 4),
        "Precision (Macro)": round(results['SaBERT_Verificado']['Precision'], 4),
        "Recall (Macro)": round(results['SaBERT_Verificado']['Recall'], 4),
        "F1-Score (Macro)": round(results['SaBERT_Verificado']['F1-Score'], 4),
        "ROC-AUC": round(results['SaBERT_Verificado']['AUC'], 4),
        "Tiempo Total (s)": round(results['SaBERT_Verificado']['Tiempo Total (s)'], 2),
        "Latencia (ms/m)": round(results['SaBERT_Verificado']['Latencia (ms)'], 2)
    },
    {
        "Modelo": "3. Spanish Fake News (Juanillaberia)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['Spanish_FakeNews_Juanillaberia']['Accuracy'], 4),
        "Precision (Macro)": round(results['Spanish_FakeNews_Juanillaberia']['Precision'], 4),
        "Recall (Macro)": round(results['Spanish_FakeNews_Juanillaberia']['Recall'], 4),
        "F1-Score (Macro)": round(results['Spanish_FakeNews_Juanillaberia']['F1-Score'], 4),
        "ROC-AUC": round(results['Spanish_FakeNews_Juanillaberia']['AUC'], 4),
        "Tiempo Total (s)": round(results['Spanish_FakeNews_Juanillaberia']['Tiempo Total (s)'], 2),
        "Latencia (ms/m)": round(results['Spanish_FakeNews_Juanillaberia']['Latencia (ms)'], 2)
    },
    {
        "Modelo": "4. LLM Zero-Shot (Qwen 1.5B)",
        "N° Registros": len(y_true),
        "Accuracy": round(results['Qwen2.5_1.5B_ZeroShot']['Accuracy'], 4),
        "Precision (Macro)": round(results['Qwen2.5_1.5B_ZeroShot']['Precision'], 4),
        "Recall (Macro)": round(results['Qwen2.5_1.5B_ZeroShot']['Recall'], 4),
        "F1-Score (Macro)": round(results['Qwen2.5_1.5B_ZeroShot']['F1-Score'], 4),
        "ROC-AUC": round(results['Qwen2.5_1.5B_ZeroShot']['AUC'], 4),
        "Tiempo Total (s)": round(results['Qwen2.5_1.5B_ZeroShot']['Tiempo Total (s)'], 2),
        "Latencia (ms/m)": round(results['Qwen2.5_1.5B_ZeroShot']['Latencia (ms)'], 2)
    }
])

print("\n" + "="*80, flush=True)
print("=== TABLA COMPARATIVA CONSOLIDADA (4 MODELOS) ===", flush=True)
print("="*80, flush=True)
print(metricas_df.to_string(index=False), flush=True)

# 5.3 Grafico de Barras Comparativo de Metricas
x = np.arange(len(modelos_labels))
width = 0.16

fig, ax = plt.subplots(figsize=(13, 6.5))
rects1 = ax.bar(x - 2*width, metricas_df["Accuracy"], width, label="Accuracy", color="#1f77b4")
rects2 = ax.bar(x - 1*width, metricas_df["Precision (Macro)"], width, label="Precision (Macro)", color="#2ca02c")
rects3 = ax.bar(x, metricas_df["Recall (Macro)"], width, label="Recall (Macro)", color="#ff7f0e")
rects4 = ax.bar(x + 1*width, metricas_df["F1-Score (Macro)"], width, label="F1-Score (Macro)", color="#d62728")
rects5 = ax.bar(x + 2*width, metricas_df["ROC-AUC"], width, label="ROC-AUC", color="#9467bd")

ax.set_ylabel("Puntuacion (0.0 - 1.0)", fontsize=12)
ax.set_title("Comparativa Global de Rendimiento en Deteccion de Fake News (2.604 Noticias)", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modelos_labels, fontsize=11, fontweight='semibold')
ax.legend(loc="lower right", frameon=True, shadow=True)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects1, rects2, rects3, rects4, rects5]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
fig.savefig(os.path.join(img_dir, "comparativa_metricas_4_modelos.png"), dpi=300)
plt.close(fig)

# 5.4 Grafico de Tiempos de Inferencia y Latencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

bar_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

ax1.bar(modelos_labels, metricas_df["Tiempo Total (s)"], color=bar_colors, edgecolor='black', alpha=0.85)
ax1.set_title("Tiempo Total de Inferencia (Segundos)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Segundos (s)", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_xticklabels(modelos_labels, rotation=15, ha='right')
for i, v in enumerate(metricas_df["Tiempo Total (s)"]):
    ax1.text(i, v + max(metricas_df["Tiempo Total (s)"])*0.02, f"{v:.1f}s", ha='center', fontweight='bold', fontsize=9)

ax2.bar(modelos_labels, metricas_df["Latencia (ms/m)"], color=bar_colors, edgecolor='black', alpha=0.85)
ax2.set_title("Latencia por Muestra (Milisegundos/noticia)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Milisegundos (ms)", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xticklabels(modelos_labels, rotation=15, ha='right')
for i, v in enumerate(metricas_df["Latencia (ms/m)"]):
    ax2.text(i, v + max(metricas_df["Latencia (ms/m)"])*0.02, f"{v:.1f}ms", ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Evaluacion de Eficiencia Computacional y Latencia', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(img_dir, "comparativa_tiempos_4_modelos.png"), dpi=300)
plt.close(fig)

# 5.5 Curvas ROC Comparativas
fig, ax = plt.subplots(figsize=(8.5, 6.5))

ax.plot(fpr_m1, tpr_m1, label=f"BETO Narrativaai (AUC = {auc_m1:.4f})", color='#1f77b4', lw=2.2)
ax.plot(fpr_m2, tpr_m2, label=f"SaBERT VerificadoProf (AUC = {auc_m2:.4f})", color='#2ca02c', lw=2.2)
ax.plot(fpr_m3, tpr_m3, label=f"Spanish FakeNews Juanillaberia (AUC = {auc_m3:.4f})", color='#d62728', lw=2.2)
ax.plot(fpr_m4, tpr_m4, label=f"LLM Zero-Shot Qwen 1.5B (AUC = {auc_m4:.4f})", color='#9467bd', lw=2.2)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Clasificador Aleatorio (AUC = 0.5000)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos (FPR / 1 - Especificidad)', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR / Sensibilidad)', fontsize=11)
ax.set_title('Curvas ROC Comparativas - Clasificacion de Fake News (4 Modelos)', fontsize=13, fontweight='bold')
ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(img_dir, "curvas_roc_4_modelos.png"), dpi=300)
plt.close(fig)

# 5.6 Curvas Precision-Recall Comparativas
fig, ax = plt.subplots(figsize=(8.5, 6.5))

ax.plot(pr_rec_m1, pr_prec_m1, label=f"BETO Narrativaai (AP = {ap_m1:.4f})", color='#1f77b4', lw=2.2)
ax.plot(pr_rec_m2, pr_prec_m2, label=f"SaBERT VerificadoProf (AP = {ap_m2:.4f})", color='#2ca02c', lw=2.2)
ax.plot(pr_rec_m3, pr_prec_m3, label=f"Spanish FakeNews Juanillaberia (AP = {ap_m3:.4f})", color='#d62728', lw=2.2)
ax.plot(pr_rec_m4, pr_prec_m4, label=f"LLM Zero-Shot Qwen 1.5B (AP = {ap_m4:.4f})", color='#9467bd', lw=2.2)

no_skill_baseline = counts[1] / len(y_true)
ax.plot([0, 1], [no_skill_baseline, no_skill_baseline], 'k--', lw=1.5, label=f'Linea Base Prevalencia ({no_skill_baseline:.3f})')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('Exhaustividad / Recall', fontsize=11)
ax.set_ylabel('Precision / Precision', fontsize=11)
ax.set_title('Curvas Precision-Recall Comparativas (4 Modelos)', fontsize=13, fontweight='bold')
ax.legend(loc="lower left", frameon=True, shadow=True, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(img_dir, "curvas_pr_4_modelos.png"), dpi=300)
plt.close(fig)

# 5.7 Guardar archivo JSON estructurado
json_path = os.path.join(report_dir, "metricas_4_modelos_fakenews.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'dataset_path': dataset_path,
            'total_registros': len(y_true),
            'clase_verdadera_0': int(counts[0]),
            'clase_falsa_1': int(counts[1]),
            'dispositivo': str(device),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'modelos': results
    }, f, indent=4, ensure_ascii=False)

print(f"[INFO] Metricas serializadas en: {json_path}", flush=True)
print("\n" + "="*80, flush=True)
print(">>> EVALUACION COMPLETADA EXITOSAMENTE PARA TODOS LOS MODELOS.", flush=True)
print("="*80, flush=True)