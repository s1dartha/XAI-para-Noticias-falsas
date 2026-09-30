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
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")

# 1. Cargar dataset ampliado (N = 4.418)
dataset_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv"
print(f"⏳ Leyendo dataset base: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"✅ Cargado: {len(df)} registros. Columnas: {df.columns.tolist()}")

# Mapear etiqueta canónica
df['label_num'] = df['clase_num'].astype(int)

# 2. Cargar modelo Sensacionalismo BETO
model_sens_name = "JJNeila/bert-spanish-sensationalism-oss"
print(f"⏳ Cargando BETO Sensacionalismo ({model_sens_name})...")
tokenizer_sens = AutoTokenizer.from_pretrained(model_sens_name)
model_sens = AutoModelForSequenceClassification.from_pretrained(model_sens_name).to(device)
model_sens.eval()

def predict_sensationalism_batch(text_list, batch_size=64, max_len=256):
    probs_all = []
    for i in range(0, len(text_list), batch_size):
        batch = [str(t)[:1000] for t in text_list[i:i+batch_size]]
        inputs = tokenizer_sens(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model_sens(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_all.extend(probs.tolist())
    return probs_all

print("⏳ [Paso 1/4] Calculando Sensacionalismo Documental (P_full) para 4.418 noticias...")
t0 = time.time()
df['P_full'] = predict_sensationalism_batch(df['Text'].tolist(), batch_size=64, max_len=256)
t_pfull = time.time() - t0
print(f"✅ P_full computado en {t_pfull:.2f} s ({t_pfull/len(df)*1000:.1f} ms/doc).")

# Segmentar oraciones
print("⏳ [Paso 2/4] Segmentando oraciones...")
all_sentences_flat = []
doc_sentence_indices = []

for text in df['Text']:
    sents = [s.strip() for s in nltk.sent_tokenize(str(text), language='spanish') if len(s.strip()) > 8]
    if len(sents) == 0:
        sents = [str(text).strip()[:200]]
    start_idx = len(all_sentences_flat)
    all_sentences_flat.extend(sents)
    end_idx = len(all_sentences_flat)
    doc_sentence_indices.append((start_idx, end_idx, sents))

print(f"📊 Total de oraciones extraídas en el corpus: {len(all_sentences_flat)}")

# Sensacionalismo por oración
print(f"⏳ Evaluando sensacionalismo en todas las {len(all_sentences_flat)} oraciones en GPU...")
t0 = time.time()
all_sentence_probs = predict_sensationalism_batch(all_sentences_flat, batch_size=128, max_len=128)
t_sents = time.time() - t0
print(f"✅ Oraciones evaluadas para sensacionalismo en {t_sents:.2f} s ({t_sents/len(all_sentences_flat)*1000:.2f} ms/oración).")

# Asignar métricas de frases a cada documento y preparar ablaciones
p_mean_list = []
p_max_list = []
p_top2_list = []
sigma_sens_list = []
dr_list = []
ablated_texts = []
doc_needs_ablation = []

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
    
    if len(sents) >= 3:
        top2_set = {top_indices[0], top_indices[1]}
        ablated_sents = [s for s_idx, s in enumerate(sents) if s_idx not in top2_set]
        ablated_text = " ".join(ablated_sents)
        ablated_texts.append(ablated_text)
        doc_needs_ablation.append((i, True))
    else:
        doc_needs_ablation.append((i, False))

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

# Liberar memoria de modelo sensacionalismo para SBERT
del model_sens
del tokenizer_sens
torch.cuda.empty_cache()

# 3. Cargar SentenceTransformer para Redundancia
sbert_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"⏳ [Paso 3/4] Cargando Sentence-BERT ({sbert_name})...")
sbert_model = SentenceTransformer(sbert_name, device=device)

print(f"⏳ Codificando {len(all_sentences_flat)} oraciones con SBERT...")
t0 = time.time()
all_embeddings = sbert_model.encode(
    all_sentences_flat,
    batch_size=128,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)
t_sbert = time.time() - t0
print(f"✅ SBERT completado en {t_sbert:.2f} s ({t_sbert/len(all_sentences_flat)*1000:.2f} ms/oración).")

# 4. Calcular métricas de redundancia intra-documental
print("⏳ [Paso 4/4] Calculando similitud coseno intra-documento...")
num_sentences_list = []
num_pairs_list = []
mean_sim_list = []
max_sim_list = []
redundant_pairs_list = []
redundancy_density_list = []

TAU_REDUNDANCY = 0.34 # Umbral bayesiano calibrado con GMM en la tesis

for i, (start_idx, end_idx, sents) in enumerate(tqdm(doc_sentence_indices, desc="Redundancia por doc")):
    n_sents = end_idx - start_idx
    num_sentences_list.append(n_sents)
    
    if n_sents < 2:
        num_pairs_list.append(0)
        mean_sim_list.append(np.nan)
        max_sim_list.append(np.nan)
        redundant_pairs_list.append(0)
        redundancy_density_list.append(0.0)
    else:
        doc_emb = all_embeddings[start_idx:end_idx] # (n_sents, 384)
        sim_matrix = torch.matmul(doc_emb, doc_emb.T).cpu().numpy()
        
        # Extraer triángulo superior estricto
        triu_indices = np.triu_indices(n_sents, k=1)
        pairwise_sims = sim_matrix[triu_indices]
        n_pairs = len(pairwise_sims)
        
        mean_sim = float(np.mean(pairwise_sims))
        max_sim = float(np.max(pairwise_sims))
        n_red = int(np.sum(pairwise_sims >= TAU_REDUNDANCY))
        density = float(n_red / n_pairs) if n_pairs > 0 else 0.0
        
        num_pairs_list.append(n_pairs)
        mean_sim_list.append(mean_sim)
        max_sim_list.append(max_sim)
        redundant_pairs_list.append(n_red)
        redundancy_density_list.append(density)

df['num_sentences'] = num_sentences_list
df['num_pairs'] = num_pairs_list
df['mean_intra_similarity'] = mean_sim_list
df['max_intra_similarity'] = max_sim_list
df['redundant_pairs_count'] = redundant_pairs_list
df['redundancy_density'] = redundancy_density_list

# Imputar nulos de redundancia en docs de 1 sola frase con la mediana
median_max_sim = df['max_intra_similarity'].dropna().median()
median_mean_sim = df['mean_intra_similarity'].dropna().median()
df['max_intra_similarity_clean'] = df['max_intra_similarity'].fillna(median_max_sim)
df['mean_intra_similarity_clean'] = df['mean_intra_similarity'].fillna(median_mean_sim)

# Discretizaciones:
# A) 7 Bins de Shannon (Árbol supervisado de entropía)
cuts_7 = [-np.inf, 0.5924, 0.6177, 0.6336, 0.8077, 0.8514, 0.9308, np.inf]
df['shannon_bin_7'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)

# B) 4 Clases Parsimoniosas
cuts_4 = [-np.inf, 0.60, 0.81, 0.98, np.inf]
df['shannon_bin_4'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_4, labels=[1, 2, 3, 4]).astype(int)

# Guardar dataset con features extraídas
out_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_ampliado_4418.csv"
df.to_csv(out_csv, index=False)
print(f"🎉 ¡Todas las características extraídas y guardadas en: {out_csv} ({len(df)} registros)!")
