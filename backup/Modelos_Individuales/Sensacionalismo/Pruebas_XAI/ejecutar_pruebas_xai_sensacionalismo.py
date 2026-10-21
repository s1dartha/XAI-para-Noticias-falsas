#!/usr/bin/env python3
"""
=============================================================================
PIPELINE DE EXPLAINABLE AI (XAI) PARA DETECCIÓN DE SENSACIONALISMO EN ESPAÑOL
Modelo: JJNeila/bert-spanish-sensationalism-oss (BETO fine-tuned)
Hardware: Optimizado para GPU NVIDIA GeForce GTX 1650 (device='cuda')
Evaluación: Muestra balanceada de 40 registros (20 IA Sintético, 20 Amarillismo Real)
=============================================================================
Tesis de Maestría / Investigación en NLP & Data Science
"""

import os
import gc
import time
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, InputXGradient
import shap
from lime.lime_text import LimeTextExplainer

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
TARGET_DIR = SCRIPT_DIR
REPORT_DIR = os.path.join(SENS_ROOT, "reportes")
IMAGE_DIR = os.path.join(REPORT_DIR, "imagenes", "XAI_pruebas_sensacionalismo")

candidate_ama = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_Amarillismo.csv"),
    os.path.join(SENS_ROOT, "dataset_Amarillismo.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_Amarillismo.csv")
]
DATASET_AMA_PATH = next((p for p in candidate_ama if os.path.exists(p)), candidate_ama[0])

candidate_ia = [
    os.path.join(SENS_ROOT, "Dataset", "dataset_IA_sintetico_70.csv"),
    os.path.join(BASE_DIR, "Datasets/sensacionalismo/dataset_IA_sintetico_70.csv")
]
DATASET_IA_PATH = next((p for p in candidate_ia if os.path.exists(p)), candidate_ia[0])

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Dispositivo configurado: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INIT] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    print(f"[INIT] VRAM Asignada Inicial: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")


# =============================================================================
# 1. CARGA DE MODELO Y TOKENIZADOR
# =============================================================================
MODEL_NAME = "JJNeila/bert-spanish-sensationalism-oss"
print(f"\n[MODEL] Cargando {MODEL_NAME} con attn_implementation='eager'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, attn_implementation='eager')
model = model.to(DEVICE)
model.eval()

# ID de etiquetas: 0 = No Sensacionalista, 1 = Sensacionalista
LABEL_NAMES = ["No Sensacionalista", "Sensacionalista"]
print(f"[MODEL] Modelo cargado exitosamente en VRAM ({torch.cuda.memory_allocated(0) / (1024**2):.1f} MB en uso).")


# =============================================================================
# 2. CONSTRUCCIÓN DE LA MUESTRA BALANCEADA (40 REGISTROS)
# =============================================================================
print("\n[DATA] Cargando datasets y construyendo muestra balanceada (40 registros: 20 IA, 20 Amarillismo)...")
df_ama_raw = pd.read_csv(DATASET_AMA_PATH)
df_ia_raw = pd.read_csv(DATASET_IA_PATH)

# Selección de 20 de IA Sintético (10 Sensacionalistas, 10 No Sensacionalistas)
ia_sens_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
ia_nonsens_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

sample_ia = []
for idx in ia_sens_indices:
    sample_ia.append({
        'id': f'IA_SENS_{idx}',
        'fuente_dataset': 'IA Sintético (70 reg.)',
        'texto': df_ia_raw.loc[idx, 'Texto'],
        'clase_real': 1,
        'etiqueta_real': 'Sensacionalista',
        'categoria': df_ia_raw.loc[idx, 'Tema']
    })
for idx in ia_nonsens_indices:
    sample_ia.append({
        'id': f'IA_NONSENS_{idx}',
        'fuente_dataset': 'IA Sintético (70 reg.)',
        'texto': df_ia_raw.loc[idx, 'Texto'],
        'clase_real': 0,
        'etiqueta_real': 'No Sensacionalista',
        'categoria': df_ia_raw.loc[idx, 'Tema']
    })

# Selección de 20 de Amarillismo (10 Sensacionalistas / Amarillistas, 10 No Sensacionalistas / No Amarillistas)
ama_sens_rows = df_ama_raw[df_ama_raw['Amarillismo'] == 'Amarillista'].iloc[:10]
ama_nonsens_rows = df_ama_raw[df_ama_raw['Amarillismo'] == 'No Amarillista'].iloc[[0, 1, 4, 5, 6, 2, 3, 7, 8, 9]]

sample_ama = []
for idx, r in ama_sens_rows.iterrows():
    sample_ama.append({
        'id': f'AMA_SENS_{idx}',
        'fuente_dataset': 'Amarillismo Real (202 reg.)',
        'texto': r['Titular'],
        'clase_real': 1,
        'etiqueta_real': 'Sensacionalista',
        'categoria': str(r.get('Fuente', 'Prensa'))
    })
for idx, r in ama_nonsens_rows.iterrows():
    sample_ama.append({
        'id': f'AMA_NONSENS_{idx}',
        'fuente_dataset': 'Amarillismo Real (202 reg.)',
        'texto': r['Titular'],
        'clase_real': 0,
        'etiqueta_real': 'No Sensacionalista',
        'categoria': str(r.get('Fuente', 'Prensa'))
    })

eval_samples = sample_ia + sample_ama
df_eval = pd.DataFrame(eval_samples)

eval_csv_path_40 = os.path.join(TARGET_DIR, "muestra_evaluacion_xai_40.csv")
df_eval.to_csv(eval_csv_path_40, index=False)
# También guardamos compatibilidad con el nombre anterior
df_eval.to_csv(os.path.join(TARGET_DIR, "muestra_evaluacion_xai_20.csv"), index=False)
print(f"[DATA] Muestra de 40 registros guardada en: {eval_csv_path_40}")


# =============================================================================
# 3. INFERENCIA BASE DEL MODELO SOBRE LA MUESTRA
# =============================================================================
print("\n[INFERENCE] Ejecutando predicción base en GPU para los 40 registros...")
texts = df_eval['texto'].tolist()
inputs_batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)

with torch.no_grad():
    logits_base = model(**inputs_batch).logits
    probs_base = torch.softmax(logits_base, dim=-1).cpu().numpy()

df_eval['prob_no_sens'] = probs_base[:, 0]
df_eval['prob_sens'] = probs_base[:, 1]
df_eval['clase_pred'] = np.argmax(probs_base, axis=1)
df_eval['pred_correcta'] = df_eval['clase_real'] == df_eval['clase_pred']

print(f"[INFERENCE] Aciertos Globales: {df_eval['pred_correcta'].sum()}/40 ({df_eval['pred_correcta'].mean()*100:.1f}%)")
print(f"[INFERENCE] Aciertos IA Sintético: {df_eval[df_eval['fuente_dataset'].str.contains('IA')]['pred_correcta'].sum()}/20 ({df_eval[df_eval['fuente_dataset'].str.contains('IA')]['pred_correcta'].mean()*100:.1f}%)")
print(f"[INFERENCE] Aciertos Amarillismo Real: {df_eval[df_eval['fuente_dataset'].str.contains('Amarillismo')]['pred_correcta'].sum()}/20 ({df_eval[df_eval['fuente_dataset'].str.contains('Amarillismo')]['pred_correcta'].mean()*100:.1f}%)")


# =============================================================================
# 4. IMPLEMENTACIÓN DE LAS 6 TÉCNICAS XAI
# =============================================================================

# Wrapper para inferencia por lotes de texto (utilizado por LIME y SHAP)
def predict_proba_texts(text_list):
    if isinstance(text_list, np.ndarray):
        text_list = text_list.tolist()
    elif isinstance(text_list, str):
        text_list = [text_list]
    elif not isinstance(text_list, list):
        text_list = list(text_list)
        
    encoded = tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**encoded).logits
        probs = torch.softmax(out, dim=-1).cpu().numpy()
    return probs

# Wrapper de embeddings para Captum (Integrated Gradients & InputXGradient)
embedding_layer = model.bert.embeddings.word_embeddings

def forward_with_embeds(embeds, attention_mask):
    out = model(inputs_embeds=embeds, attention_mask=attention_mask)
    return out.logits

ig_engine = IntegratedGradients(forward_with_embeds)
ixg_engine = InputXGradient(forward_with_embeds)
lime_engine = LimeTextExplainer(class_names=LABEL_NAMES, random_state=42)
shap_masker = shap.maskers.Text(tokenizer)
shap_engine = shap.Explainer(predict_proba_texts, shap_masker, output_names=LABEL_NAMES)


def explain_text(text, target_class=1):
    """
    Calcula la atribución de importancia por token para las 6 técnicas XAI:
    1. Integrated Gradients (IG)
    2. Attention Rollout
    3. SHAP
    4. LIME
    5. Gradient-weighted Feature Attribution (Input X Gradient)
    6. Layer-wise Relevance Propagation (LRP - Chefer et al. Transformer-adapted)
    """
    inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    
    attributions = {}
    latencias = {}
    
    # Pre-cálculo de embeddings
    with torch.no_grad():
        input_embeds = embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

    # 1. INTEGRATED GRADIENTS (IG)
    t0 = time.time()
    attr_ig = ig_engine.attribute(
        input_embeds,
        baseline_embeds,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=25
    )
    score_ig = attr_ig.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    attributions['Integrated Gradients'] = score_ig
    latencias['Integrated Gradients'] = time.time() - t0

    # 2. GRADIENT-WEIGHTED FEATURE ATTRIBUTION (Input X Gradient)
    t0 = time.time()
    attr_ixg = ixg_engine.attribute(
        input_embeds,
        additional_forward_args=(attention_mask,),
        target=target_class
    )
    score_ixg = attr_ixg.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    attributions['Gradient * Input'] = score_ixg
    latencias['Gradient * Input'] = time.time() - t0

    # 3. ATTENTION ROLLOUT
    t0 = time.time()
    with torch.no_grad():
        out_att = model(**inputs, output_attentions=True)
    eye = torch.eye(seq_len, device=DEVICE)
    rollout = eye
    for layer_att in out_att.attentions:
        mean_att = layer_att[0].mean(dim=0)
        aug = 0.5 * mean_att + 0.5 * eye
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)
    score_rollout = rollout[0].cpu().numpy()  # Desde [CLS] hacia todos los tokens
    attributions['Attention Rollout'] = score_rollout
    latencias['Attention Rollout'] = time.time() - t0

    # 4. LAYER-WISE RELEVANCE PROPAGATION (LRP - Transformer-Adapted)
    t0 = time.time()
    model.zero_grad()
    out_lrp = model(**inputs, output_attentions=True)
    score_target = out_lrp.logits[0, target_class]
    for a in out_lrp.attentions:
        a.retain_grad()
    score_target.backward()

    R = torch.eye(seq_len, device=DEVICE)
    for att in out_lrp.attentions:
        if att.grad is not None:
            cam = att * att.grad
            cam = torch.clamp(cam, min=0).mean(dim=1).squeeze(0)
            cam = cam + eye
            cam = cam / cam.sum(dim=-1, keepdim=True)
            R = torch.matmul(cam, R)
    score_lrp = R[0].detach().cpu().numpy()
    attributions['LRP (Transformer)'] = score_lrp
    latencias['LRP (Transformer)'] = time.time() - t0

    # 5. LIME
    t0 = time.time()
    exp_lime = lime_engine.explain_instance(
        text,
        predict_proba_texts,
        num_features=seq_len,
        num_samples=100,
        labels=[target_class]
    )
    lime_map = dict(exp_lime.as_list(label=target_class))
    score_lime = np.zeros(seq_len)
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace('##', '').strip('¡!¿?,.:;\"\'').lower()
        if not clean_tok or tok in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        for word, val in lime_map.items():
            clean_word = word.strip('¡!¿?,.:;\"\'').lower()
            if clean_tok in clean_word or clean_word in clean_tok:
                score_lime[i] = val
                break
    attributions['LIME'] = score_lime
    latencias['LIME'] = time.time() - t0

    # 6. SHAP
    t0 = time.time()
    shap_res = shap_engine([text], max_evals=80)
    shap_vals_raw = shap_res.values[0, :, target_class]
    score_shap = np.zeros(seq_len)
    min_len = min(len(shap_vals_raw), seq_len)
    score_shap[:min_len] = shap_vals_raw[:min_len]
    attributions['SHAP'] = score_shap
    latencias['SHAP'] = time.time() - t0

    return tokens, attributions, latencias


# =============================================================================
# 5. SUITE DE VALIDACIÓN: FAITHFULNESS, ABLATION & PERTURBATION COMPARISON
# =============================================================================

def evaluar_fidelidad_instancia(text, tokens, attr_vector, target_class, top_k_ratio=0.20):
    """
    Calcula Comprehensiveness y Sufficiency para un vector de atribución en una instancia.
    Perturbación: Reemplazo por token [MASK] (ID = 0 en BETO).
    Granularidad: Nivel subword token (excluyendo [CLS] y [SEP]).
    """
    seq_len = len(tokens)
    mask_token_id = tokenizer.mask_token_id
    
    # Probabilidad original
    inputs_orig = tokenizer(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        orig_prob = torch.softmax(model(**inputs_orig).logits, dim=-1)[0, target_class].item()

    content_indices = [i for i in range(1, seq_len - 1) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    if not content_indices:
        return {'comprehensiveness': 0.0, 'sufficiency': 0.0, 'orig_prob': orig_prob}

    scores = [(i, attr_vector[i]) for i in content_indices]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    
    k = max(1, int(round(len(content_indices) * top_k_ratio)))
    top_k_indices = set([idx for idx, _ in scores_sorted[:k]])

    # 1. Comprehensiveness (Erasure): Enmascarar los top-k tokens más importantes
    erasure_ids = inputs_orig['input_ids'].clone()
    for idx in top_k_indices:
        erasure_ids[0, idx] = mask_token_id
    with torch.no_grad():
        erasure_prob = torch.softmax(model(input_ids=erasure_ids, attention_mask=inputs_orig['attention_mask']).logits, dim=-1)[0, target_class].item()
    comprehensiveness = orig_prob - erasure_prob

    # 2. Sufficiency: Mantener ÚNICAMENTE los top-k tokens (enmascarar el resto del contexto)
    suff_ids = inputs_orig['input_ids'].clone()
    for idx in content_indices:
        if idx not in top_k_indices:
            suff_ids[0, idx] = mask_token_id
    with torch.no_grad():
        suff_prob = torch.softmax(model(input_ids=suff_ids, attention_mask=inputs_orig['attention_mask']).logits, dim=-1)[0, target_class].item()
    sufficiency = orig_prob - suff_prob

    return {
        'comprehensiveness': comprehensiveness,
        'sufficiency': sufficiency,
        'orig_prob': orig_prob,
        'erasure_prob': erasure_prob,
        'suff_prob': suff_prob,
        'k_masked': k
    }


def calcular_curvas_morf_lorf(text, tokens, attr_vector, target_class, steps=11):
    """
    Calcula curvas de perturbación acumulativa MoRF (Most Relevant First)
    y LoRF (Least Relevant First) con reemplazo de tokens por [MASK].
    steps: 11 puntos (0.0, 0.1, 0.2, ..., 1.0)
    """
    seq_len = len(tokens)
    mask_token_id = tokenizer.mask_token_id
    inputs_orig = tokenizer(text, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        orig_prob = torch.softmax(model(**inputs_orig).logits, dim=-1)[0, target_class].item()

    content_indices = [i for i in range(1, seq_len - 1) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    num_content = len(content_indices)
    if num_content == 0:
        return np.ones(steps) * orig_prob, np.ones(steps) * orig_prob

    scores = [(i, attr_vector[i]) for i in content_indices]
    morf_order = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    lorf_order = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=False)]

    ratios = np.linspace(0.0, 1.0, steps)
    morf_probs = []
    lorf_probs = []

    for r in ratios:
        num_to_mask = int(round(r * num_content))
        
        # MoRF
        ids_morf = inputs_orig['input_ids'].clone()
        for idx in morf_order[:num_to_mask]:
            ids_morf[0, idx] = mask_token_id
        with torch.no_grad():
            p_morf = torch.softmax(model(input_ids=ids_morf, attention_mask=inputs_orig['attention_mask']).logits, dim=-1)[0, target_class].item()
        morf_probs.append(p_morf)

        # LoRF
        ids_lorf = inputs_orig['input_ids'].clone()
        for idx in lorf_order[:num_to_mask]:
            ids_lorf[0, idx] = mask_token_id
        with torch.no_grad():
            p_lorf = torch.softmax(model(input_ids=ids_lorf, attention_mask=inputs_orig['attention_mask']).logits, dim=-1)[0, target_class].item()
        lorf_probs.append(p_lorf)

    return np.array(morf_probs), np.array(lorf_probs)


def evaluar_perturbaciones_cambio_clase(text, tokens, attr_vector, target_class, top_k_ratio=0.20, sample_seed=42):
    """
    Evalúa si la predicción de clase del modelo cambia frente a 3 estrategias de perturbación
    sobre el top-20% de tokens más importantes seleccionados por la técnica XAI:
    1. Enmascaramiento: Sustitución por token [MASK] (ID = 0 en BETO).
    2. Eliminación: Borrado físico del token (contracción de secuencia y reindexación).
    3. Ruido Aleatorio: Sustitución por token aleatorio extraído del vocabulario de BETO.
    """
    seq_len = len(tokens)
    inputs_orig = tokenizer(text, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        logits_orig = model(**inputs_orig).logits
        prob_orig = torch.softmax(logits_orig, dim=-1)[0, target_class].item()
        pred_orig = torch.argmax(logits_orig, dim=-1).item()

    content_indices = [i for i in range(1, seq_len - 1) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    if not content_indices:
        return {
            'pred_orig': pred_orig,
            'prob_orig': prob_orig,
            'mask_flipped': False, 'mask_prob': prob_orig, 'mask_pred': pred_orig,
            'del_flipped': False, 'del_prob': prob_orig, 'del_pred': pred_orig,
            'noise_flipped': False, 'noise_prob': prob_orig, 'noise_pred': pred_orig,
            'k_perturbed': 0
        }

    scores = [(i, attr_vector[i]) for i in content_indices]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    k = max(1, int(round(len(content_indices) * top_k_ratio)))
    top_k_indices = set([idx for idx, _ in scores_sorted[:k]])

    # 1. Enmascaramiento ([MASK])
    ids_mask = inputs_orig['input_ids'].clone()
    for idx in top_k_indices:
        ids_mask[0, idx] = tokenizer.mask_token_id
    with torch.no_grad():
        out_mask = model(input_ids=ids_mask, attention_mask=inputs_orig['attention_mask']).logits
        prob_mask = torch.softmax(out_mask, dim=-1)[0, target_class].item()
        pred_mask = torch.argmax(out_mask, dim=-1).item()
    mask_flipped = (pred_mask != pred_orig)

    # 2. Eliminación (Borrado físico)
    keep_indices = [i for i in range(seq_len) if i not in top_k_indices]
    ids_del = inputs_orig['input_ids'][:, keep_indices]
    mask_del = inputs_orig['attention_mask'][:, keep_indices]
    with torch.no_grad():
        out_del = model(input_ids=ids_del, attention_mask=mask_del).logits
        prob_del = torch.softmax(out_del, dim=-1)[0, target_class].item()
        pred_del = torch.argmax(out_del, dim=-1).item()
    del_flipped = (pred_del != pred_orig)

    # 3. Ruido Aleatorio (Token aleatorio del vocabulario BETO)
    rng = np.random.RandomState(sample_seed)
    # Rango [6, tokenizer.vocab_size - 1] para evitar tokens especiales [UNK], [PAD], [CLS], [SEP], [MASK]
    random_tokens = rng.randint(6, tokenizer.vocab_size, size=len(top_k_indices))
    ids_noise = inputs_orig['input_ids'].clone()
    for idx, r_tok in zip(sorted(list(top_k_indices)), random_tokens):
        ids_noise[0, idx] = int(r_tok)
    with torch.no_grad():
        out_noise = model(input_ids=ids_noise, attention_mask=inputs_orig['attention_mask']).logits
        prob_noise = torch.softmax(out_noise, dim=-1)[0, target_class].item()
        pred_noise = torch.argmax(out_noise, dim=-1).item()
    noise_flipped = (pred_noise != pred_orig)

    return {
        'pred_orig': pred_orig,
        'prob_orig': prob_orig,
        'mask_flipped': bool(mask_flipped),
        'mask_prob': float(prob_mask),
        'mask_pred': int(pred_mask),
        'del_flipped': bool(del_flipped),
        'del_prob': float(prob_del),
        'del_pred': int(pred_del),
        'noise_flipped': bool(noise_flipped),
        'noise_prob': float(prob_noise),
        'noise_pred': int(pred_noise),
        'k_perturbed': k
    }


# =============================================================================
# 6. TEST DE ALEATORIZACIÓN EN CASCADA (ADEBAYO ET AL., 2018)
# =============================================================================

def ejecutar_sanidad_adebayo(ref_text, target_class=1):
    """
    Aplica aleatorización en cascada de parámetros desde las capas superiores hacia inferiores.
    Niveles evaluados:
      0: Modelo Original intacto
      1: Randomize Classifier Head
      2: Classifier + Encoder Layer 11
      3: Classifier + Encoder Layers 11, 10
      4: Classifier + Encoder Layers 11, 10, 9, 8
      5: Randomize All (Classifier + Todas las 12 capas)
    """
    print("\n[SANITY] Ejecutando Adebayo Cascading Parameter Randomization...")
    tokens, base_attrs, _ = explain_text(ref_text, target_class)
    methods = list(base_attrs.keys())
    
    levels = ["0_Orig", "1_Classifier", "2_Layer11", "3_Layers10-11", "4_Layers8-11", "5_AllLayers"]
    spearman_results = {m: [] for m in methods}
    cosine_results = {m: [] for m in methods}
    
    for m in methods:
        spearman_results[m].append(1.0)
        cosine_results[m].append(1.0)

    test_model = copy.deepcopy(model)
    
    cascade_definitions = [
        ("1_Classifier", lambda m: [
            nn.init.normal_(m.classifier.weight, std=0.02),
            nn.init.zeros_(m.classifier.bias)
        ]),
        ("2_Layer11", lambda m: [
            m.bert.encoder.layer[11].apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
        ]),
        ("3_Layers10-11", lambda m: [
            m.bert.encoder.layer[10].apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
        ]),
        ("4_Layers8-11", lambda m: [
            [m.bert.encoder.layer[ly].apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None) for ly in [8, 9]]
        ]),
        ("5_AllLayers", lambda m: [
            [m.bert.encoder.layer[ly].apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None) for ly in range(8)]
        ])
    ]

    for level_name, perturb_fn in cascade_definitions:
        perturb_fn(test_model)
        test_model.eval()
        
        inputs = tokenizer(ref_text, return_tensors='pt').to(DEVICE)
        seq_len = len(tokens)
        eye = torch.eye(seq_len, device=DEVICE)
        
        # 1. IG alterado
        def fwd_alt(emb, mask):
            return test_model(inputs_embeds=emb, attention_mask=mask).logits
        ig_alt = IntegratedGradients(fwd_alt)
        ixg_alt = InputXGradient(fwd_alt)
        emb_layer_alt = test_model.bert.embeddings.word_embeddings
        with torch.no_grad():
            emb_alt = emb_layer_alt(inputs['input_ids'])
            base_alt = torch.zeros_like(emb_alt)
        attr_ig_alt = ig_alt.attribute(emb_alt, base_alt, additional_forward_args=(inputs['attention_mask'],), target=target_class, n_steps=20).sum(dim=-1).squeeze(0).detach().cpu().numpy()
        attr_ixg_alt = ixg_alt.attribute(emb_alt, additional_forward_args=(inputs['attention_mask'],), target=target_class).sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # 2. Rollout alterado
        with torch.no_grad():
            out_att_alt = test_model(**inputs, output_attentions=True)
        rollout_alt = eye
        for la in out_att_alt.attentions:
            ma = la[0].mean(dim=0)
            au = 0.5 * ma + 0.5 * eye
            au = au / au.sum(dim=-1, keepdim=True)
            rollout_alt = torch.matmul(au, rollout_alt)
        attr_rollout_alt = rollout_alt[0].cpu().numpy()

        # 3. LRP alterado
        test_model.zero_grad()
        out_lrp_alt = test_model(**inputs, output_attentions=True)
        sc = out_lrp_alt.logits[0, target_class]
        for a in out_lrp_alt.attentions:
            a.retain_grad()
        sc.backward()
        R_alt = torch.eye(seq_len, device=DEVICE)
        for att in out_lrp_alt.attentions:
            if att.grad is not None:
                cam = att * att.grad
                cam = torch.clamp(cam, min=0).mean(dim=1).squeeze(0)
                cam = cam + eye
                cam = cam / cam.sum(dim=-1, keepdim=True)
                R_alt = torch.matmul(cam, R_alt)
        attr_lrp_alt = R_alt[0].detach().cpu().numpy()

        # 4. LIME alterado
        def pred_lime_alt(txts):
            if isinstance(txts, np.ndarray): txts = txts.tolist()
            enc = tokenizer(txts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                return torch.softmax(test_model(**enc).logits, dim=-1).cpu().numpy()
        exp_l_alt = lime_engine.explain_instance(ref_text, pred_lime_alt, num_features=seq_len, num_samples=60, labels=[target_class])
        l_map = dict(exp_l_alt.as_list(label=target_class))
        attr_lime_alt = np.zeros(seq_len)
        for i, tok in enumerate(tokens):
            ctok = tok.replace('##', '').strip('¡!¿?,.:;\"\'').lower()
            if not ctok or tok in ['[CLS]', '[SEP]']: continue
            for w, v in l_map.items():
                if ctok in w.lower():
                    attr_lime_alt[i] = v
                    break

        # 5. SHAP alterado
        shap_alt_engine = shap.Explainer(pred_lime_alt, shap_masker, output_names=LABEL_NAMES)
        shap_alt_res = shap_alt_engine([ref_text], max_evals=60)
        raw_sh = shap_alt_res.values[0, :, target_class]
        attr_shap_alt = np.zeros(seq_len)
        ml = min(len(raw_sh), seq_len)
        attr_shap_alt[:ml] = raw_sh[:ml]

        curr_attrs = {
            'Integrated Gradients': attr_ig_alt,
            'Gradient * Input': attr_ixg_alt,
            'Attention Rollout': attr_rollout_alt,
            'LRP (Transformer)': attr_lrp_alt,
            'LIME': attr_lime_alt,
            'SHAP': attr_shap_alt
        }

        content_mask = [i for i in range(1, seq_len - 1) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
        for m in methods:
            v_orig = base_attrs[m][content_mask]
            v_alt = curr_attrs[m][content_mask]
            
            if np.std(v_orig) == 0 or np.std(v_alt) == 0:
                rho = 0.0
            else:
                rho, _ = spearmanr(v_orig, v_alt)
                if np.isnan(rho): rho = 0.0
            spearman_results[m].append(float(rho))
            
            norm_o = np.linalg.norm(v_orig)
            norm_a = np.linalg.norm(v_alt)
            if norm_o == 0 or norm_a == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(v_orig, v_alt) / (norm_o * norm_a))
            cosine_results[m].append(cos_sim)

    del test_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return levels, spearman_results, cosine_results


# =============================================================================
# 7. EJECUCIÓN PRINCIPAL SOBRE LA MUESTRA COMPLETA (40 REGISTROS)
# =============================================================================
print("\n" + "="*70)
print("INICIANDO PROCESAMIENTO XAI SOBRE LOS 40 REGISTROS EXPERIMENTALES")
print("="*70)

all_results = []
latencies_accum = {m: [] for m in ['Integrated Gradients', 'Attention Rollout', 'SHAP', 'LIME', 'Gradient * Input', 'LRP (Transformer)']}
morfs_accum = {m: [] for m in latencies_accum.keys()}
lorfs_accum = {m: [] for m in latencies_accum.keys()}

# Heatmaps de referencia representativos (2 de IA Sintético y 2 de Prensa Real)
reference_samples_for_heatmap = {
    'IA_Sensacionalista': 0,      # IA_SENS_0
    'IA_No_Sensacionalista': 10,  # IA_NONSENS_1
    'Amarillismo_Real': 20,       # AMA_SENS_0
    'No_Amarillismo_Real': 30     # AMA_NONSENS_9
}

tokens_and_attrs_saved = {}

t_start_global = time.time()

for i, row in df_eval.iterrows():
    s_id = row['id']
    texto = row['texto']
    target_cls = int(row['clase_pred']) # Atribuir a la predicción realizada por el modelo
    print(f"[{i+1}/40] Analizando {s_id} ({row['fuente_dataset'][:12]}, L:{len(texto)}, Target:{target_cls})...")
    
    tokens, attrs, lat = explain_text(texto, target_class=target_cls)
    
    # Guardar para heatmaps de muestras clave
    for k_name, idx_target in reference_samples_for_heatmap.items():
        if i == idx_target:
            tokens_and_attrs_saved[k_name] = {
                'id': s_id,
                'texto': texto,
                'tokens': tokens,
                'attrs': attrs,
                'clase_real': row['clase_real'],
                'clase_pred': row['clase_pred'],
                'prob_sens': row['prob_sens']
            }

    # Evaluar Fidelidad, Curvas y Perturbaciones para cada método XAI
    for method_name, attr_vec in attrs.items():
        latencies_accum[method_name].append(lat[method_name])
        
        # Test de fidelidad (Top 20%)
        fid = evaluar_fidelidad_instancia(texto, tokens, attr_vec, target_class=target_cls, top_k_ratio=0.20)
        
        # Curvas MoRF y LoRF
        m_curve, l_curve = calcular_curvas_morf_lorf(texto, tokens, attr_vec, target_class=target_cls, steps=11)
        morfs_accum[method_name].append(m_curve)
        lorfs_accum[method_name].append(l_curve)
        
        auc_morf = float(np.trapezoid(m_curve, dx=0.1))
        auc_lorf = float(np.trapezoid(l_curve, dx=0.1))
        delta_auc = auc_lorf - auc_morf

        # Test de Perturbación con 3 Estrategias (Enmascaramiento, Eliminación, Ruido Aleatorio)
        seed_m = 42 + i * 10 + list(attrs.keys()).index(method_name)
        pert_res = evaluar_perturbaciones_cambio_clase(texto, tokens, attr_vec, target_class=target_cls, top_k_ratio=0.20, sample_seed=seed_m)

        all_results.append({
            'sample_id': s_id,
            'fuente': row['fuente_dataset'],
            'clase_real': int(row['clase_real']),
            'clase_pred': int(row['clase_pred']),
            'metodo_xai': method_name,
            'comprehensiveness': float(fid['comprehensiveness']),
            'sufficiency': float(fid['sufficiency']),
            'auc_morf': auc_morf,
            'auc_lorf': auc_lorf,
            'delta_auc': delta_auc,
            'latencia_seg': float(lat[method_name]),
            # Nuevas métricas de cambio de clase por perturbación
            'mask_flipped': pert_res['mask_flipped'],
            'del_flipped': pert_res['del_flipped'],
            'noise_flipped': pert_res['noise_flipped'],
            'prob_orig': pert_res['prob_orig'],
            'prob_mask': pert_res['mask_prob'],
            'prob_del': pert_res['del_prob'],
            'prob_noise': pert_res['noise_prob']
        })

print(f"\n[DONE] Pipeline completado en {time.time() - t_start_global:.2f} segundos.")
df_xai_metrics = pd.DataFrame(all_results)


# =============================================================================
# 8. EJECUCIÓN DEL SANITY CHECK DE ADEBAYO
# =============================================================================
ref_headline = df_eval.loc[0, 'texto']
levels_adebayo, spearman_adebayo, cosine_adebayo = ejecutar_sanidad_adebayo(ref_headline, target_class=1)


# =============================================================================
# 9. CONSOLIDACIÓN DE MÉTRICAS Y EXPORTACIÓN JSON
# =============================================================================
summary_metrics = {}
print("\n" + "="*70)
print("RESUMEN GENERAL DE MÉTRICAS DE INTERPRETABILIDAD XAI (40 NOTICIAS)")
print("="*70)

for method in latencies_accum.keys():
    sub = df_xai_metrics[df_xai_metrics['metodo_xai'] == method]
    comp_mean = sub['comprehensiveness'].mean()
    comp_std = sub['comprehensiveness'].std()
    suff_mean = sub['sufficiency'].mean()
    suff_std = sub['sufficiency'].std()
    auc_m_mean = sub['auc_morf'].mean()
    auc_l_mean = sub['auc_lorf'].mean()
    delta_auc_mean = sub['delta_auc'].mean()
    lat_mean = np.mean(latencies_accum[method])
    lat_total = np.sum(latencies_accum[method])

    # Métricas de cambio de clase (Global, IA Sintético, Amarillismo Real)
    mask_flips_global = int(sub['mask_flipped'].sum())
    del_flips_global = int(sub['del_flipped'].sum())
    noise_flips_global = int(sub['noise_flipped'].sum())

    sub_ia = sub[sub['fuente'].str.contains('IA')]
    mask_flips_ia = int(sub_ia['mask_flipped'].sum())
    del_flips_ia = int(sub_ia['del_flipped'].sum())
    noise_flips_ia = int(sub_ia['noise_flipped'].sum())

    sub_ama = sub[sub['fuente'].str.contains('Amarillismo')]
    mask_flips_ama = int(sub_ama['mask_flipped'].sum())
    del_flips_ama = int(sub_ama['del_flipped'].sum())
    noise_flips_ama = int(sub_ama['noise_flipped'].sum())

    summary_metrics[method] = {
        'comprehensiveness_mean': round(float(comp_mean), 6),
        'comprehensiveness_std': round(float(comp_std), 6),
        'sufficiency_mean': round(float(suff_mean), 6),
        'sufficiency_std': round(float(suff_std), 6),
        'auc_morf_mean': round(float(auc_m_mean), 6),
        'auc_lorf_mean': round(float(auc_l_mean), 6),
        'delta_auc_mean': round(float(delta_auc_mean), 6),
        'latencia_promedio_seg': round(float(lat_mean), 4),
        'latencia_total_seg': round(float(lat_total), 2),
        'cambio_clase': {
            'global_40': {'mask': mask_flips_global, 'del': del_flips_global, 'noise': noise_flips_global},
            'ia_sintetico_20': {'mask': mask_flips_ia, 'del': del_flips_ia, 'noise': noise_flips_ia},
            'amarillismo_real_20': {'mask': mask_flips_ama, 'del': del_flips_ama, 'noise': noise_flips_ama}
        },
        'adebayo_spearman': {lvl: round(float(val), 4) for lvl, val in zip(levels_adebayo, spearman_adebayo[method])},
        'adebayo_cosine': {lvl: round(float(val), 4) for lvl, val in zip(levels_adebayo, cosine_adebayo[method])}
    }
    
    print(f"\n--- {method} ---")
    print(f"  Comprehensiveness:           {comp_mean:+.4f} ± {comp_std:.4f} (Ideal: alto > 0.3)")
    print(f"  Sufficiency:                 {suff_mean:+.4f} ± {suff_std:.4f} (Ideal: bajo ~ 0.0)")
    print(f"  AUC MoRF (Most Rel First):   {auc_m_mean:.4f} (Ideal: bajo)")
    print(f"  AUC LoRF (Least Rel First):  {auc_l_mean:.4f} (Ideal: alto)")
    print(f"  Delta AUC (LoRF - MoRF):     {delta_auc_mean:+.4f} (Ideal: positivo > 0)")
    print(f"  Flips IA (20 reg):           Mask: {mask_flips_ia}/20 | Del: {del_flips_ia}/20 | Noise: {noise_flips_ia}/20")
    print(f"  Flips Amarillismo (20 reg):  Mask: {mask_flips_ama}/20 | Del: {del_flips_ama}/20 | Noise: {noise_flips_ama}/20")
    print(f"  Latencia media GPU:          {lat_mean:.3f} s / instancia")

json_output_path = os.path.join(REPORT_DIR, "metricas_xai_sensacionalismo.json")
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'resumen_metodos': summary_metrics,
        'metricas_por_instancia': all_results,
        'niveles_adebayo': levels_adebayo,
        'total_registros': 40,
        'hardware': {
            'dispositivo': str(DEVICE),
            'gpu_nombre': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'vram_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0
        }
    }, f, indent=2, ensure_ascii=False)
print(f"\n[EXPORT] Archivo JSON con todas las métricas guardado en: {json_output_path}")


# =============================================================================
# 10. GENERACIÓN Y EXPORTACIÓN DE FIGURAS Y GRÁFICOS
# =============================================================================
print("\n[VISUALIZATION] Generando y exportando figuras de alta resolución (300 DPI)...")
methods_list = list(summary_metrics.keys())

# FIGURA 1: Comparativa de Fidelidad (Comprehensiveness vs Sufficiency)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(methods_list))
width = 0.35

comp_vals = [summary_metrics[m]['comprehensiveness_mean'] for m in methods_list]
comp_errs = [summary_metrics[m]['comprehensiveness_std'] for m in methods_list]
suff_vals = [summary_metrics[m]['sufficiency_mean'] for m in methods_list]
suff_errs = [summary_metrics[m]['sufficiency_std'] for m in methods_list]

rects1 = ax.bar(x - width/2, comp_vals, width, yerr=comp_errs, label='Comprehensiveness (Erasure) ↑ [Ideal: Alto > 0.3]', color='#2b5c8f', capsize=4, alpha=0.9)
rects2 = ax.bar(x + width/2, suff_vals, width, yerr=suff_errs, label='Sufficiency ↓ [Ideal: Cercano a 0]', color='#d95f02', capsize=4, alpha=0.9)

ax.set_ylabel('Impacto en Probabilidad P(y*)', fontsize=12, fontweight='bold')
ax.set_title('Test de Fidelidad XAI: Comprehensiveness vs. Sufficiency (40 Muestras Balanceadas)\nModelo: JJNeila/bert-spanish-sensationalism-oss', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods_list, rotation=15, ha='right', fontsize=10, fontweight='semibold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
fig1_path = os.path.join(IMAGE_DIR, "xai_faithfulness_comprehensiveness_sufficiency.png")
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig1_path}")


# FIGURA 2: Curvas de Ablación MoRF vs LoRF (Grid 2x3 para los 6 métodos)
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()
steps_x = np.linspace(0.0, 1.0, 11) * 100

for idx, m in enumerate(methods_list):
    ax = axes[idx]
    m_curves_arr = np.array(morfs_accum[m]) # (40, 11)
    l_curves_arr = np.array(lorfs_accum[m]) # (40, 11)
    
    m_mean = m_curves_arr.mean(axis=0)
    m_std = m_curves_arr.std(axis=0)
    l_mean = l_curves_arr.mean(axis=0)
    l_std = l_curves_arr.std(axis=0)
    
    ax.plot(steps_x, m_mean, marker='o', color='#d95f02', linewidth=2.2, label=f'MoRF (AUC={summary_metrics[m]["auc_morf_mean"]:.3f})')
    ax.fill_between(steps_x, np.clip(m_mean - m_std, 0, 1), np.clip(m_mean + m_std, 0, 1), color='#d95f02', alpha=0.15)
    
    ax.plot(steps_x, l_mean, marker='s', color='#2b5c8f', linewidth=2.2, label=f'LoRF (AUC={summary_metrics[m]["auc_lorf_mean"]:.3f})')
    ax.fill_between(steps_x, np.clip(l_mean - l_std, 0, 1), np.clip(l_mean + l_std, 0, 1), color='#2b5c8f', alpha=0.15)
    
    ax.set_title(f"{m}\nΔAUC = {summary_metrics[m]['delta_auc_mean']:+.3f}", fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('% Tokens Enmascarados ([MASK])', fontsize=9)
    ax.set_ylabel('Probabilidad P(y*)', fontsize=9)
    ax.legend(loc='lower left', fontsize=9)

fig.suptitle('Curvas de Perturbación Ablativa MoRF vs. LoRF (Muestra 40 Noticias)\n(Most Relevant First vs. Least Relevant First)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2_path = os.path.join(IMAGE_DIR, "xai_curvas_morf_lorf_comparativa.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig2_path}")


# FIGURA 3: Sanity Check de Adebayo (Aleatorización en Cascada)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
colors_adebayo = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, m in enumerate(methods_list):
    ax1.plot(levels_adebayo, summary_metrics[m]['adebayo_spearman'].values(), marker='o', linewidth=2, color=colors_adebayo[idx], label=m)
    ax2.plot(levels_adebayo, summary_metrics[m]['adebayo_cosine'].values(), marker='s', linewidth=2, color=colors_adebayo[idx], label=m)

ax1.set_title('Sensibilidad a Parámetros (Spearman Rank ρ)\nIdeal: Caída pronunciada hacia 0', fontsize=11, fontweight='bold')
ax1.set_ylabel('Correlación Spearman ρ', fontsize=10, fontweight='semibold')
ax1.set_xlabel('Nivel de Aleatorización en Cascada', fontsize=10)
ax1.set_xticklabels(levels_adebayo, rotation=25, ha='right')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.legend(loc='upper right', fontsize=8.5)

ax2.set_title('Sensibilidad a Parámetros (Similitud Coseno)\nIdeal: Degradación estructural ortogonal', fontsize=11, fontweight='bold')
ax2.set_ylabel('Similitud Coseno', fontsize=10, fontweight='semibold')
ax2.set_xlabel('Nivel de Aleatorización en Cascada', fontsize=10)
ax2.set_xticklabels(levels_adebayo, rotation=25, ha='right')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.legend(loc='upper right', fontsize=8.5)

fig.suptitle('Sanity Check de Adebayo et al. (2018): Aleatorización en Cascada de Capas', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig3_path = os.path.join(IMAGE_DIR, "xai_parameter_randomization_adebayo.png")
fig.savefig(fig3_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig3_path}")


# FIGURA 4: Comparativa de Latencia Computacional en GPU
fig, ax = plt.subplots(figsize=(9, 5))
lat_vals = [summary_metrics[m]['latencia_promedio_seg'] for m in methods_list]
bars = ax.barh(methods_list, lat_vals, color='#3182bd', alpha=0.85, edgecolor='#08519c', height=0.55)

for bar in bars:
    w = bar.get_width()
    ax.text(w + (max(lat_vals)*0.01), bar.get_y() + bar.get_height()/2, f'{w:.3f} s', ha='left', va='center', fontsize=9.5, fontweight='bold')

ax.set_xlabel('Tiempo de Cómputo Promedio por Muestra (segundos en GPU CUDA)', fontsize=10, fontweight='bold')
ax.set_title('Eficiencia Computacional en GPU (NVIDIA GTX 1650):\nLatencia Promedio por Técnica XAI (40 Muestras)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(0, max(lat_vals) * 1.15)
plt.tight_layout()
fig4_path = os.path.join(IMAGE_DIR, "xai_latencia_por_metodo.png")
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig4_path}")


# FIGURAS 5-8: Mapas de Calor (Heatmaps) Comparativos por Muestra Representativa
def generar_heatmap_comparativo(muestra_data, filename_out, titulo_prefix):
    s_tokens = muestra_data['tokens']
    s_attrs = muestra_data['attrs']
    
    valid_idx = [i for i, t in enumerate(s_tokens) if t not in ['[CLS]', '[SEP]', '[PAD]']]
    clean_tokens = [s_tokens[i] for i in valid_idx]
    
    attr_matrix = []
    row_labels = []
    for m in methods_list:
        v = np.array(s_attrs[m])[valid_idx]
        max_abs = np.max(np.abs(v))
        if max_abs > 0:
            v_norm = v / max_abs
        else:
            v_norm = v
        attr_matrix.append(v_norm)
        row_labels.append(m)
        
    attr_matrix = np.array(attr_matrix)
    
    fig_w = max(10, len(clean_tokens) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    
    sns.heatmap(
        attr_matrix,
        cmap='coolwarm',
        center=0.0,
        cbar_kws={'label': 'Atribución Relativa Normalizada [-1, +1]', 'orientation': 'horizontal', 'pad': 0.25, 'shrink': 0.5},
        yticklabels=row_labels,
        xticklabels=clean_tokens,
        ax=ax,
        linewidths=0.5,
        linecolor='#ffffff'
    )
    
    ax.set_xticklabels(clean_tokens, rotation=45, ha='right', fontsize=9.5)
    ax.set_yticklabels(row_labels, fontsize=9.5, fontweight='semibold')
    
    subt = f"Titular: \"{muestra_data['texto'][:75]}...\"\nPred: {LABEL_NAMES[muestra_data['clase_pred']]} (P_Sens={muestra_data['prob_sens']:.4f}) | Real: {LABEL_NAMES[muestra_data['clase_real']]}"
    ax.set_title(f"{titulo_prefix}\n{subt}", fontsize=11, fontweight='bold', pad=12)
    
    plt.tight_layout()
    out_p = os.path.join(IMAGE_DIR, filename_out)
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"  -> Guardada: {out_p}")


generar_heatmap_comparativo(
    tokens_and_attrs_saved['IA_Sensacionalista'],
    "xai_heatmap_ia_sensacionalista.png",
    "Heatmap XAI Comparativo: Noticia Sintética IA Sensacionalista (IA_SENS_0)"
)

generar_heatmap_comparativo(
    tokens_and_attrs_saved['IA_No_Sensacionalista'],
    "xai_heatmap_ia_no_sensacionalista.png",
    "Heatmap XAI Comparativo: Noticia Sintética IA Sobria (IA_NONSENS_1)"
)

generar_heatmap_comparativo(
    tokens_and_attrs_saved['Amarillismo_Real'],
    "xai_heatmap_amarillismo_real_sensacionalista.png",
    "Heatmap XAI Comparativo: Titular Prensa Amarillista Real (AMA_SENS_0)"
)

generar_heatmap_comparativo(
    tokens_and_attrs_saved['No_Amarillismo_Real'],
    "xai_heatmap_amarillismo_real_no_sensacionalista.png",
    "Heatmap XAI Comparativo: Titular Prensa Objetivo Real (AMA_NONSENS_9)"
)


# =============================================================================
# NUEVAS FIGURAS 9, 10 y 11: CAMBIO DE CLASE PREDICHA POR PERTURBACIÓN
# =============================================================================

# FIGURA 9: Dataset IA Sintético (20 Muestras)
fig, ax = plt.subplots(figsize=(11, 6))
x_idx = np.arange(len(methods_list))
w_bar = 0.26

flips_ia_mask = [summary_metrics[m]['cambio_clase']['ia_sintetico_20']['mask'] for m in methods_list]
flips_ia_del = [summary_metrics[m]['cambio_clase']['ia_sintetico_20']['del'] for m in methods_list]
flips_ia_noise = [summary_metrics[m]['cambio_clase']['ia_sintetico_20']['noise'] for m in methods_list]

b1 = ax.bar(x_idx - w_bar, flips_ia_mask, w_bar, label='Enmascaramiento ([MASK])', color='#2b5c8f', edgecolor='#17395c', alpha=0.9)
b2 = ax.bar(x_idx, flips_ia_del, w_bar, label='Eliminación (Token Deletion)', color='#d95f02', edgecolor='#8c3c00', alpha=0.9)
b3 = ax.bar(x_idx + w_bar, flips_ia_noise, w_bar, label='Ruido Aleatorio (Random Token)', color='#2ca02c', edgecolor='#1a611a', alpha=0.9)

# Anotaciones numéricas
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        pct = (h / 20) * 100
        ax.annotate(f'{int(h)}\n({pct:.0f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_ylabel('Noticias con Cambio de Decisión (Total: 20)', fontsize=11, fontweight='bold')
ax.set_title('Cambio de Clase Predicha por Estrategia de Perturbación (Top 20% Tokens)\nDataset: IA Sintético (20 Noticias Balanceadas)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_idx)
ax.set_xticklabels(methods_list, rotation=15, ha='right', fontsize=10, fontweight='semibold')
ax.set_ylim(0, 22)
ax.axhline(0, color='gray', linewidth=0.8)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')
plt.tight_layout()
fig9_path = os.path.join(IMAGE_DIR, "xai_cambio_clase_perturbaciones_ia.png")
fig.savefig(fig9_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig9_path}")


# FIGURA 10: Dataset Amarillismo Real (20 Muestras)
fig, ax = plt.subplots(figsize=(11, 6))

flips_ama_mask = [summary_metrics[m]['cambio_clase']['amarillismo_real_20']['mask'] for m in methods_list]
flips_ama_del = [summary_metrics[m]['cambio_clase']['amarillismo_real_20']['del'] for m in methods_list]
flips_ama_noise = [summary_metrics[m]['cambio_clase']['amarillismo_real_20']['noise'] for m in methods_list]

b1 = ax.bar(x_idx - w_bar, flips_ama_mask, w_bar, label='Enmascaramiento ([MASK])', color='#2b5c8f', edgecolor='#17395c', alpha=0.9)
b2 = ax.bar(x_idx, flips_ama_del, w_bar, label='Eliminación (Token Deletion)', color='#d95f02', edgecolor='#8c3c00', alpha=0.9)
b3 = ax.bar(x_idx + w_bar, flips_ama_noise, w_bar, label='Ruido Aleatorio (Random Token)', color='#2ca02c', edgecolor='#1a611a', alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        pct = (h / 20) * 100
        ax.annotate(f'{int(h)}\n({pct:.0f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_ylabel('Noticias con Cambio de Decisión (Total: 20)', fontsize=11, fontweight='bold')
ax.set_title('Cambio de Clase Predicha por Estrategia de Perturbación (Top 20% Tokens)\nDataset: Amarillismo Prensa Real (20 Noticias Balanceadas)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_idx)
ax.set_xticklabels(methods_list, rotation=15, ha='right', fontsize=10, fontweight='semibold')
ax.set_ylim(0, 22)
ax.axhline(0, color='gray', linewidth=0.8)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')
plt.tight_layout()
fig10_path = os.path.join(IMAGE_DIR, "xai_cambio_clase_perturbaciones_amarillismo.png")
fig.savefig(fig10_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig10_path}")


# FIGURA 11: Comparativa Global Side-by-Side (IA Sintético vs. Amarillismo Real)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Subplot 1: IA Sintético
b1 = ax1.bar(x_idx - w_bar, flips_ia_mask, w_bar, label='[MASK]', color='#2b5c8f', edgecolor='#17395c', alpha=0.9)
b2 = ax1.bar(x_idx, flips_ia_del, w_bar, label='Eliminación', color='#d95f02', edgecolor='#8c3c00', alpha=0.9)
b3 = ax1.bar(x_idx + w_bar, flips_ia_noise, w_bar, label='Ruido Vocab.', color='#2ca02c', edgecolor='#1a611a', alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        pct = (h / 20) * 100
        ax1.annotate(f'{int(h)} ({pct:.0f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_ylabel('Noticias con Inversión de Clase (Total: 20)', fontsize=11, fontweight='bold')
ax1.set_title('Dataset IA Sintético (Alta Polaridad Léxica)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_idx)
ax1.set_xticklabels(methods_list, rotation=20, ha='right', fontsize=9.5)
ax1.set_ylim(0, 22)
ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

# Subplot 2: Amarillismo Real
b4 = ax2.bar(x_idx - w_bar, flips_ama_mask, w_bar, label='[MASK]', color='#2b5c8f', edgecolor='#17395c', alpha=0.9)
b5 = ax2.bar(x_idx, flips_ama_del, w_bar, label='Eliminación', color='#d95f02', edgecolor='#8c3c00', alpha=0.9)
b6 = ax2.bar(x_idx + w_bar, flips_ama_noise, w_bar, label='Ruido Vocab.', color='#2ca02c', edgecolor='#1a611a', alpha=0.9)

for bars in [b4, b5, b6]:
    for bar in bars:
        h = bar.get_height()
        pct = (h / 20) * 100
        ax2.annotate(f'{int(h)} ({pct:.0f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_title('Dataset Amarillismo Prensa Real (Ambigüedad Contextual)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_idx)
ax2.set_xticklabels(methods_list, rotation=20, ha='right', fontsize=9.5)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

fig.suptitle('Impacto Causal en Decisión por Estrategia de Perturbación (Top 20% Tokens)\nComparativa entre Prensa Digital Real y Noticias Sintéticas IA', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig11_path = os.path.join(IMAGE_DIR, "xai_cambio_clase_perturbaciones_comparativa_global.png")
fig.savefig(fig11_path, dpi=300)
plt.close(fig)
print(f"  -> Guardada: {fig11_path}")

print("\n" + "="*70)
print("EJECUCIÓN DEL PIPELINE XAI FINALIZADA CON ÉXITO")
print(f"Directorio de artefactos e imágenes: {IMAGE_DIR}")
print(f"Directorio de métricas JSON: {json_output_path}")
print("="*70)
