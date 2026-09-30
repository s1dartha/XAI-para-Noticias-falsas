import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")

# 1. Cargar dataset_con_similitudes.csv
sim_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Redundancia/Dataset/dataset_con_similitudes.csv"
print(f"⏳ Leyendo dataset base: {sim_path}")
df = pd.read_csv(sim_path)
print(f"✅ Cargado: {len(df)} registros. Columnas: {df.columns.tolist()}")

# Mapear etiqueta binaria
df['label_num'] = df['class'].map({False: 0, True: 1, 'REAL': 0, 'FAKE': 1, 0: 0, 1: 1}).fillna(0).astype(int)

# 2. Cargar modelo BETO Sensacionalismo
model_name = "JJNeila/bert-spanish-sensationalism-oss"
print(f"⏳ Cargando BETO Sensacionalismo ({model_name})...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# Función de inferencia por lotes
def predict_sensationalism_batch(text_list, batch_size=64, max_len=256):
    probs_all = []
    for i in range(0, len(text_list), batch_size):
        batch = [str(t)[:1000] for t in text_list[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_all.extend(probs.tolist())
    return probs_all

print("⏳ [Paso 1/3] Calculando Sensacionalismo Documental (P_full) para 2,604 noticias...")
t0 = time.time()
df['P_full'] = predict_sensationalism_batch(df['Text'].tolist(), batch_size=64, max_len=256)
t_pfull = time.time() - t0
print(f"✅ P_full computado en {t_pfull:.2f} s ({t_pfull/len(df)*1000:.1f} ms/doc).")

# Segmentar cada noticia en oraciones
print("⏳ [Paso 2/3] Segmentando oraciones y preparando evaluación frasal...")
all_sentences_flat = []
doc_sentence_indices = [] # lista de (start_idx, end_idx)

for text in df['Text']:
    sents = [s.strip() for s in nltk.sent_tokenize(str(text), language='spanish') if len(s.strip()) > 8]
    if len(sents) == 0:
        sents = [str(text).strip()[:200]]
    start_idx = len(all_sentences_flat)
    all_sentences_flat.extend(sents)
    end_idx = len(all_sentences_flat)
    doc_sentence_indices.append((start_idx, end_idx, sents))

print(f"📊 Total de oraciones extraídas en el corpus: {len(all_sentences_flat)}")
print(f"⏳ Evaluando sensacionalismo en todas las {len(all_sentences_flat)} oraciones en GPU...")

t0 = time.time()
all_sentence_probs = predict_sensationalism_batch(all_sentences_flat, batch_size=128, max_len=128)
t_sents = time.time() - t0
print(f"✅ Oraciones evaluadas en {t_sents:.2f} s ({t_sents/len(all_sentences_flat)*1000:.2f} ms/oración).")

# Asignar métricas de frases a cada documento
p_mean_list = []
p_max_list = []
p_top2_list = []
sigma_sens_list = []
dr_list = []
top1_indices_list = []
top2_indices_list = []
ablated_texts = []
doc_needs_ablation = []

print("⏳ [Paso 3/3] Calculando métricas morfológicas y ablaciones virtuales...")
for i, (start_idx, end_idx, sents) in enumerate(doc_sentence_indices):
    s_probs = all_sentence_probs[start_idx:end_idx]
    p_full = df['P_full'].iloc[i]
    
    p_mean = float(np.mean(s_probs))
    p_max = float(np.max(s_probs))
    top_indices = np.argsort(s_probs)[::-1]
    
    if len(s_probs) >= 2:
        p_top2 = float((s_probs[top_indices[0]] + s_probs[top_indices[1]]) / 2.0)
        sigma = float(np.std(s_probs))
    else:
        p_top2 = p_max
        sigma = 0.0
        
    dr = float(p_full / (p_max + 1e-6))
    
    p_mean_list.append(p_mean)
    p_max_list.append(p_max)
    p_top2_list.append(p_top2)
    sigma_sens_list.append(sigma)
    dr_list.append(dr)
    
    # Preparar ablación de Top-2 frases para noticias con k >= 3
    if len(sents) >= 3:
        top2_set = {top_indices[0], top_indices[1]}
        ablated_sents = [s for s_idx, s in enumerate(sents) if s_idx not in top2_set]
        ablated_text = " ".join(ablated_sents)
        ablated_texts.append(ablated_text)
        doc_needs_ablation.append((i, True))
    else:
        doc_needs_ablation.append((i, False))

# Evaluar ablaciones en GPU
print(f"⏳ Evaluando {len(ablated_texts)} noticias abladas para impacto causal de gatillos...")
ablated_probs = predict_sensationalism_batch(ablated_texts, batch_size=64, max_len=256)

delta_p_gatillo_list = [0.0] * len(df)
abl_idx = 0
for doc_idx, has_abl in doc_needs_ablation:
    if has_abl:
        p_abl = ablated_probs[abl_idx]
        abl_idx += 1
        delta_p_gatillo_list[doc_idx] = float(df['P_full'].iloc[doc_idx] - p_abl)
    else:
        delta_p_gatillo_list[doc_idx] = float(df['P_full'].iloc[doc_idx] - p_mean_list[doc_idx])

df['P_mean'] = p_mean_list
df['P_max'] = p_max_list
df['P_top2'] = p_top2_list
df['sigma_sens'] = sigma_sens_list
df['dilution_ratio'] = dr_list
df['delta_p_gatillo'] = delta_p_gatillo_list

# Rellenar nulos de redundancia de manera neutra (133 docs con 1 sola oración)
median_max_sim = df['max_intra_similarity'].dropna().median()
median_mean_sim = df['mean_intra_similarity'].dropna().median()
df['max_intra_similarity_clean'] = df['max_intra_similarity'].fillna(median_max_sim)
df['mean_intra_similarity_clean'] = df['mean_intra_similarity'].fillna(median_mean_sim)

# Discretización de Shannon (7 Bins)
cuts_7 = [-np.inf, 0.5924, 0.6177, 0.6336, 0.8077, 0.8514, 0.9308, np.inf]
df['shannon_bin_7'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)

# Discretización Parsimoniosa (4 Clases)
cuts_4 = [-np.inf, 0.60, 0.81, 0.98, np.inf]
df['shannon_bin_4'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_4, labels=[1, 2, 3, 4]).astype(int)

# Densidad de pares redundantes
df['redundancy_density'] = df['redundant_pairs_count'] / df['num_pairs'].replace(0, 1)

# Guardar features consolidadas
out_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_2604.csv"
df.to_csv(out_csv, index=False)
print(f"🎉 ¡Todas las características morfológicas consolidadas y guardadas en: {out_csv}!")
