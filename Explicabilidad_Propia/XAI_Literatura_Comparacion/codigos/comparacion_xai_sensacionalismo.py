#!/usr/bin/env python3
"""
=============================================================================
BENCHMARK COMPARATIVO RIGUROSO: 6 MÉTODOS XAI DE LITERATURA (SUBTOKENS) VS.
NUESTRA EXPLICABILIDAD PROPIA ORACIONAL / FRASAL (ESTILO MEXGEN) EN SENSACIONALISMO
=============================================================================
Modelo: JJNeila/bert-spanish-sensationalism-oss (~110M params, BETO Encoder-Only)
Hardware: GPU NVIDIA GeForce GTX 1650 (device='cuda', attn_implementation='eager')
Muestra Unificada (N=40):
  - 20 Noticias de IA Sintético (10 Sensacionalistas, 10 No Sensacionalistas)
  - 20 Noticias de Amarillismo Real (10 Sensacionalistas, 10 No Sensacionalistas)
Regímenes Evaluados sobre la MISMA Muestra (N=40):
  - Régimen A: Titular / Apertura (donde operan canónicamente los 6 XAI)
  - Régimen B: Noticia Completa Consolidada (Titular + Copete + Cuerpo)
  - Referencia Poblacional: Corpus Completo de 272 Noticias (202 Reales + 70 IA)

Autor: Investigación Doctoral / Maestría en Ciencia de Datos y PLN
Fecha: Octubre 2026
=============================================================================
"""

import os
import gc
import re
import time
import copy
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    spearmanr, pearsonr, wilcoxon, ttest_rel, friedmanchisquare, fisher_exact, mannwhitneyu
)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, InputXGradient
import shap
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore")

# Configuración de reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Estilos visuales de publicación científica (300 DPI)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10.5,
    'axes.labelsize': 11.5,
    'axes.titlesize': 12.5,
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Rutas del Proyecto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TESIS_ROOT = os.path.abspath(os.path.join(COMP_ROOT, "..", ".."))
SENS_ROOT = os.path.join(TESIS_ROOT, "Modelos_Individuales", "Sensacionalismo")

REPORT_DIR = os.path.join(COMP_ROOT, "reportes")
DATA_OUT_DIR = os.path.join(REPORT_DIR, "datos")
IMG_OUT_DIR = os.path.join(REPORT_DIR, "imagenes", "sensacionalismo")

os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

DATASET_AMA_PATH = os.path.join(SENS_ROOT, "Dataset", "dataset_Amarillismo.csv")
DATASET_IA_PATH = os.path.join(SENS_ROOT, "Dataset", "dataset_IA_sintetico_70.csv")
EXISTING_FRASES_CSV = os.path.join(SENS_ROOT, "reportes", "datos_sensacionalismo_frases.csv")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Dispositivo de ejecución: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INIT] GPU detectada: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 1. CARGA DEL MODELO BETO DE SENSACIONALISMO
# =============================================================================
MODEL_NAME = "JJNeila/bert-spanish-sensationalism-oss"
print(f"\n[1/7] Cargando modelo y tokenizador: {MODEL_NAME} (attn_implementation='eager')...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, attn_implementation='eager').to(DEVICE)
model.eval()
LABEL_NAMES = ["No Sensacionalista", "Sensacionalista"]
embedding_layer = model.bert.embeddings.word_embeddings


# =============================================================================
# 2. SEGMENTACIÓN DE FRASES / CLÁUSULAS Y CONSTRUCCIÓN DE LA MUESTRA N=40
# =============================================================================
def split_into_phrases(text):
    """
    Segmentación en frases y cláusulas informativas delimitadas por puntuación
    (, ; : . ! ? ¡ ¿ saltos de línea, guiones y comillas).
    Filtro sintáctico: >= 2 palabras y >= 6 caracteres.
    """
    raw_pieces = re.split(r'[,;:.!?¡¿\n—–\"\'«»()]+', str(text))
    phrases = []
    for p in raw_pieces:
        clean = p.strip()
        words = clean.split()
        if len(words) >= 2 and len(clean) >= 6:
            phrases.append(clean)
    if not phrases:
        phrases = [str(text).strip()]
    return phrases


print("\n[2/7] Construyendo la Muestra Balanceada Exacta de N=40 Registros (Alineada en Titular y Documento)...")
df_ama_raw = pd.read_csv(DATASET_AMA_PATH)
df_ia_raw = pd.read_csv(DATASET_IA_PATH)

ia_sens_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
ia_nonsens_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

samples_40 = []
for idx in ia_sens_indices + ia_nonsens_indices:
    is_sens = idx in ia_sens_indices
    txt = str(df_ia_raw.loc[idx, 'Texto']).strip()
    samples_40.append({
        'sample_id': f"IA_{'SENS' if is_sens else 'NONSENS'}_{idx}",
        'doc_id': f"IA_{idx}",
        'fuente': 'IA Sintético (20 reg.)',
        'clase_real': 1 if is_sens else 0,
        'etiqueta_real': 'Sensacionalista' if is_sens else 'No Sensacionalista',
        'texto_titular': txt,
        'texto_completo': txt
    })

ama_sens_rows = df_ama_raw[df_ama_raw['Amarillismo'] == 'Amarillista'].iloc[:10]
ama_nonsens_rows = df_ama_raw[df_ama_raw['Amarillismo'] == 'No Amarillista'].iloc[[0, 1, 4, 5, 6, 2, 3, 7, 8, 9]]

for is_sens, rows_df in [(True, ama_sens_rows), (False, ama_nonsens_rows)]:
    for idx, r in rows_df.iterrows():
        titular = str(r['Titular']).strip()
        copete = str(r.get('Copete/Resumen', '')).strip() if str(r.get('Copete/Resumen', '')) != 'nan' else ''
        cuerpo = str(r['Cuerpo']).strip() if str(r['Cuerpo']) != 'nan' else ''
        full_text = re.sub(r'\s+', ' ', f"{titular}. {copete} {cuerpo}".strip())[:1200]
        samples_40.append({
            'sample_id': f"AMA_{'SENS' if is_sens else 'NONSENS'}_{idx}",
            'doc_id': f"AMA_{idx}",
            'fuente': 'Prensa Real Amarillismo (20 reg.)',
            'clase_real': 1 if is_sens else 0,
            'etiqueta_real': 'Sensacionalista' if is_sens else 'No Sensacionalista',
            'texto_titular': titular,
            'texto_completo': full_text
        })

df_sample40 = pd.DataFrame(samples_40)
print(f"      Muestra construida: {len(df_sample40)} registros (20 IA Sintético, 20 Prensa Real).")


# =============================================================================
# 3. MOTORES DE INFERENCIA Y EXPLICABILIDAD (6 XAI LITERATURA + MEXGEN PROPIO)
# =============================================================================
def predict_proba_texts(text_list, max_len=256):
    if isinstance(text_list, np.ndarray):
        text_list = text_list.tolist()
    elif isinstance(text_list, str):
        text_list = [text_list]
    elif not isinstance(text_list, list):
        text_list = list(text_list)

    probs_all = []
    batch_size = 32
    for i in range(0, len(text_list), batch_size):
        batch = [str(t) if str(t).strip() else "[PAD]" for t in text_list[i:i+batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**encoded).logits
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            probs_all.append(probs)
    return np.vstack(probs_all)


def forward_with_embeds(embeds, attention_mask):
    return model(inputs_embeds=embeds, attention_mask=attention_mask).logits


ig_engine = IntegratedGradients(forward_with_embeds)
ixg_engine = InputXGradient(forward_with_embeds)
lime_engine = LimeTextExplainer(class_names=LABEL_NAMES, random_state=SEED)
shap_masker = shap.maskers.Text(tokenizer)


def explain_all_methods(text, target_class=1, max_len=256):
    """
    Calcula las atribuciones a nivel de subtoken para:
    1. Integrated Gradients (IG)
    2. Attention Rollout
    3. SHAP (Partition Explainer)
    4. LIME
    5. Input × Gradient
    6. LRP (Transformer - Chefer et al., 2021)
    7. Propio MexGen (Atribución por Frases/Cláusulas Gatillo proyectada a tokens y evaluada por frase)
    """
    inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)

    attributions = {}
    latencias = {}

    with torch.no_grad():
        input_embeds = embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

    # 1. Integrated Gradients (IG)
    t0 = time.perf_counter()
    attr_ig = ig_engine.attribute(
        input_embeds, baseline_embeds,
        additional_forward_args=(attention_mask,),
        target=target_class, n_steps=25, internal_batch_size=8
    )
    attributions['1. Integrated Gradients'] = attr_ig.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    latencias['1. Integrated Gradients'] = time.perf_counter() - t0
    del attr_ig

    # 2. Attention Rollout
    t0 = time.perf_counter()
    with torch.no_grad():
        out_att = model(**inputs, output_attentions=True)
    eye = torch.eye(seq_len, device=DEVICE)
    rollout = eye
    for layer_att in out_att.attentions:
        mean_att = layer_att[0].mean(dim=0)
        aug = 0.5 * mean_att + 0.5 * eye
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)
    attributions['2. Attention Rollout'] = rollout[0].cpu().numpy()
    latencias['2. Attention Rollout'] = time.perf_counter() - t0
    del out_att

    # 3. SHAP (Partition Explainer)
    t0 = time.perf_counter()
    def pred_fn_local(txts):
        return predict_proba_texts(txts, max_len=max_len)
    shap_engine_local = shap.Explainer(pred_fn_local, shap_masker, output_names=LABEL_NAMES)
    shap_res = shap_engine_local([text], max_evals=80)
    shap_vals_raw = shap_res.values[0, :, target_class]
    score_shap = np.zeros(seq_len)
    min_len = min(len(shap_vals_raw), seq_len)
    score_shap[:min_len] = shap_vals_raw[:min_len]
    attributions['3. SHAP (Partition)'] = score_shap
    latencias['3. SHAP (Partition)'] = time.perf_counter() - t0

    # 4. LIME
    t0 = time.perf_counter()
    exp_lime = lime_engine.explain_instance(
        text, pred_fn_local,
        num_features=min(seq_len, 40),
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
    attributions['4. LIME'] = score_lime
    latencias['4. LIME'] = time.perf_counter() - t0

    # 5. Input × Gradient
    t0 = time.perf_counter()
    attr_ixg = ixg_engine.attribute(
        input_embeds,
        additional_forward_args=(attention_mask,),
        target=target_class
    )
    attributions['5. Input × Gradient'] = attr_ixg.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    latencias['5. Input × Gradient'] = time.perf_counter() - t0
    del attr_ixg

    # 6. LRP (Transformer - Chefer et al., 2021)
    t0 = time.perf_counter()
    model.zero_grad()
    out_lrp = model(**inputs, output_attentions=True)
    score_target = out_lrp.logits[0, target_class]
    for a in out_lrp.attentions:
        a.retain_grad()
    score_target.backward()
    R = torch.eye(seq_len, device=DEVICE)
    for att in out_lrp.attentions:
        if att.grad is not None:
            cam = (att.detach() * att.grad.detach())
            cam = torch.clamp(cam, min=0).mean(dim=1).squeeze(0)
            cam = cam + eye
            cam = cam / cam.sum(dim=-1, keepdim=True)
            R = torch.matmul(cam, R)
    attributions['6. LRP (Transformer)'] = R[0].detach().cpu().numpy()
    latencias['6. LRP (Transformer)'] = time.perf_counter() - t0
    del out_lrp, score_target, R
    model.zero_grad(set_to_none=True)
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # 7. Propio MexGen (Frases / Cláusulas Gatillo)
    t0 = time.perf_counter()
    phrases = split_into_phrases(text)
    if len(phrases) <= 1:
        # Si es una sola cláusula, subdividimos por sintagmas / conectores (en, que, para, con, por, de)
        sub_clauses = [p.strip() for p in re.split(r'\b(?:que|para|como|cuando|donde|porque|aunque|según|tras|ante)\b', text, flags=re.IGNORECASE) if len(p.strip().split()) >= 2]
        if len(sub_clauses) >= 2:
            phrases = sub_clauses

    probs_phrases_mat = predict_proba_texts(phrases, max_len=128)
    probs_phrases_target = probs_phrases_mat[:, target_class]

    # Proyectar importancia de cada frase hacia los subtokens que la componen
    score_mexgen_tok = np.zeros(seq_len)
    # Tokenizamos cada frase y alineamos con la secuencia de tokens
    cursor = 1
    for p_str, p_val in zip(phrases, probs_phrases_target):
        p_toks = tokenizer.tokenize(p_str)
        n_pt = len(p_toks)
        if n_pt == 0:
            continue
        # Búsqueda de ventana deslizante cercana al cursor
        best_match = -1
        for start_idx in range(max(1, cursor - 3), min(seq_len - 1, cursor + len(p_toks) + 15)):
            window = tokens[start_idx:start_idx + n_pt]
            matches = sum(1 for a, b in zip(window, p_toks) if a == b)
            if matches >= max(1, int(0.6 * n_pt)):
                best_match = start_idx
                break
        if best_match != -1:
            end_idx = min(seq_len - 1, best_match + n_pt)
            # Pequeño desempate posicional decreciente dentro de la frase para ordenar MoRF suavemente
            for pos_in_p, idx_t in enumerate(range(best_match, end_idx)):
                score_mexgen_tok[idx_t] = float(p_val) - pos_in_p * 1e-5
            cursor = end_idx
        else:
            # Fallback léxico por coincidencia de subtoken
            p_set = set(p_toks)
            for idx_t in range(1, seq_len - 1):
                if score_mexgen_tok[idx_t] == 0 and tokens[idx_t] in p_set:
                    score_mexgen_tok[idx_t] = float(p_val)

    # Cualquier token de puntuación/conector no asignado recibe la media vecina
    for idx_t in range(1, seq_len - 1):
        if score_mexgen_tok[idx_t] == 0:
            score_mexgen_tok[idx_t] = float(np.mean(probs_phrases_target)) * 0.5

    attributions['7. Propio (MexGen Frases)'] = score_mexgen_tok
    latencias['7. Propio (MexGen Frases)'] = time.perf_counter() - t0

    mexgen_phrase_info = {
        'phrases': phrases,
        'probs_target': probs_phrases_target,
        'probs_sens': probs_phrases_mat[:, 1]
    }

    return inputs, tokens, attributions, latencias, mexgen_phrase_info


# =============================================================================
# 4. EVALUACIÓN DE FIDELIDAD, CURVAS MORF/LORF Y PERTURBACIÓN CAUSAL
# =============================================================================
def evaluate_token_method(inputs, tokens, attr_vector, target_class, top_k_ratio=0.20, seed=42):
    """
    Evalúa Comprehensiveness, Sufficiency, Curvas MoRF/LoRF (11 pasos) y
    las 3 estrategias de perturbación (Mask, Deletion, Random Vocab Noise) al Top-20% de tokens.
    """
    seq_len = len(tokens)
    mask_id = tokenizer.mask_token_id
    input_ids = inputs['input_ids']
    att_mask = inputs['attention_mask']

    with torch.no_grad():
        logits_orig = model(input_ids=input_ids, attention_mask=att_mask).logits
        probs_orig = torch.softmax(logits_orig, dim=-1)[0]
        prob_orig_target = probs_orig[target_class].item()
        prob_orig_sens = probs_orig[1].item()
        pred_orig = int(torch.argmax(logits_orig, dim=-1).item())

    content_indices = [i for i in range(1, seq_len - 1) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    num_content = len(content_indices)
    if num_content == 0:
        return None

    scores = [(i, float(attr_vector[i])) for i in content_indices]
    morf_order = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    lorf_order = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=False)]

    k = max(1, int(round(num_content * top_k_ratio)))
    top_k_set = set(morf_order[:k])

    # 1. Comprehensiveness (Enmascarar Top 20%)
    ids_mask = input_ids.clone()
    for idx in top_k_set:
        ids_mask[0, idx] = mask_id
    with torch.no_grad():
        out_mask = model(input_ids=ids_mask, attention_mask=att_mask).logits
        probs_mask = torch.softmax(out_mask, dim=-1)[0]
        prob_mask_target = probs_mask[target_class].item()
        prob_mask_sens = probs_mask[1].item()
        pred_mask = int(torch.argmax(out_mask, dim=-1).item())
    comprehensiveness = prob_orig_target - prob_mask_target
    mask_flipped = (pred_mask != pred_orig)

    # 2. Sufficiency (Retener SOLO Top 20%, enmascarar el resto)
    ids_suff = input_ids.clone()
    for idx in content_indices:
        if idx not in top_k_set:
            ids_suff[0, idx] = mask_id
    with torch.no_grad():
        out_suff = model(input_ids=ids_suff, attention_mask=att_mask).logits
        prob_suff_target = torch.softmax(out_suff, dim=-1)[0, target_class].item()
    sufficiency = prob_orig_target - prob_suff_target

    # 3. Eliminación Física (Token Deletion del Top 20%)
    keep_indices = [i for i in range(seq_len) if i not in top_k_set]
    ids_del = input_ids[:, keep_indices]
    mask_del = att_mask[:, keep_indices]
    with torch.no_grad():
        out_del = model(input_ids=ids_del, attention_mask=mask_del).logits
        probs_del = torch.softmax(out_del, dim=-1)[0]
        prob_del_target = probs_del[target_class].item()
        prob_del_sens = probs_del[1].item()
        pred_del = int(torch.argmax(out_del, dim=-1).item())
    del_drop_target = prob_orig_target - prob_del_target
    del_flipped = (pred_del != pred_orig)

    # 4. Ruido Aleatorio de Vocabulario (Random Noise en Top 20%)
    rng = np.random.RandomState(seed)
    rand_toks = rng.randint(6, tokenizer.vocab_size, size=len(top_k_set))
    ids_noise = input_ids.clone()
    for idx, r_tok in zip(sorted(list(top_k_set)), rand_toks):
        ids_noise[0, idx] = int(r_tok)
    with torch.no_grad():
        out_noise = model(input_ids=ids_noise, attention_mask=att_mask).logits
        probs_noise = torch.softmax(out_noise, dim=-1)[0]
        prob_noise_target = probs_noise[target_class].item()
        prob_noise_sens = probs_noise[1].item()
        pred_noise = int(torch.argmax(out_noise, dim=-1).item())
    noise_drop_target = prob_orig_target - prob_noise_target
    noise_flipped = (pred_noise != pred_orig)

    # 5. Curvas MoRF y LoRF (11 pasos: 0.0 a 1.0)
    steps = 11
    ratios = np.linspace(0.0, 1.0, steps)
    morf_probs = []
    lorf_probs = []
    for r in ratios:
        n_m = int(round(r * num_content))
        im = input_ids.clone()
        for idx in morf_order[:n_m]:
            im[0, idx] = mask_id
        il = input_ids.clone()
        for idx in lorf_order[:n_m]:
            il[0, idx] = mask_id
        with torch.no_grad():
            pm = torch.softmax(model(input_ids=im, attention_mask=att_mask).logits, dim=-1)[0, target_class].item()
            pl = torch.softmax(model(input_ids=il, attention_mask=att_mask).logits, dim=-1)[0, target_class].item()
        morf_probs.append(pm)
        lorf_probs.append(pl)

    auc_morf = float(np.trapezoid(morf_probs, dx=0.1))
    auc_lorf = float(np.trapezoid(lorf_probs, dx=0.1))
    delta_auc = auc_lorf - auc_morf

    return {
        'pred_orig': pred_orig,
        'prob_orig_target': prob_orig_target,
        'prob_orig_sens': prob_orig_sens,
        'comprehensiveness': comprehensiveness,
        'sufficiency': sufficiency,
        'auc_morf': auc_morf,
        'auc_lorf': auc_lorf,
        'delta_auc': delta_auc,
        'morf_curve': morf_probs,
        'lorf_curve': lorf_probs,
        'mask_prob_target': prob_mask_target,
        'mask_prob_sens': prob_mask_sens,
        'mask_pred': pred_mask,
        'mask_flipped': bool(mask_flipped),
        'del_prob_target': prob_del_target,
        'del_prob_sens': prob_del_sens,
        'del_drop_target': del_drop_target,
        'del_pred': pred_del,
        'del_flipped': bool(del_flipped),
        'noise_prob_target': prob_noise_target,
        'noise_prob_sens': prob_noise_sens,
        'noise_drop_target': noise_drop_target,
        'noise_pred': pred_noise,
        'noise_flipped': bool(noise_flipped),
        'k_tokens_perturbed': k,
        'total_content_tokens': num_content
    }


def evaluate_native_phrase_ablation(text, mexgen_phrase_info, target_class, max_len=256):
    """
    Evalúa la intervención causal nativa de nuestro método Propio (MexGen):
    - Extirpación quirúrgica de la Frase/Cláusula Top-1
    - Extirpación quirúrgica de las Frases/Cláusulas Top-2
    - Retención únicamente de la Frase Top-1 (Suficiencia Oracional)
    - Extirpación de Frase de Control Aleatoria
    """
    phrases = mexgen_phrase_info['phrases']
    probs_target = mexgen_phrase_info['probs_target']
    k = len(phrases)

    p_orig_mat = predict_proba_texts([text], max_len=max_len)[0]
    prob_orig_target = float(p_orig_mat[target_class])
    prob_orig_sens = float(p_orig_mat[1])
    pred_orig = int(np.argmax(p_orig_mat))

    sorted_idx = np.argsort(probs_target)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_set = set(sorted_idx[:2]) if k >= 2 else {top1_idx}

    # Sin Top-1
    p_no_top1_list = [p for j, p in enumerate(phrases) if j != top1_idx]
    txt_no_top1 = " ".join(p_no_top1_list) if p_no_top1_list else "[PAD]"

    # Sin Top-2
    p_no_top2_list = [p for j, p in enumerate(phrases) if j not in top2_set]
    txt_no_top2 = " ".join(p_no_top2_list) if p_no_top2_list else txt_no_top1

    # Solo Top-1 (Suficiencia Oracional)
    txt_only_top1 = phrases[top1_idx]

    # Sin Frase Control
    other_idx = [j for j in range(k) if j not in top2_set]
    if other_idx:
        ctrl_idx = other_idx[len(other_idx) // 2]
        txt_no_ctrl = " ".join([p for j, p in enumerate(phrases) if j != ctrl_idx])
    else:
        txt_no_ctrl = txt_no_top1

    res_mat = predict_proba_texts([txt_no_top1, txt_no_top2, txt_only_top1, txt_no_ctrl], max_len=max_len)

    prob_no_top1_target = float(res_mat[0, target_class])
    pred_no_top1 = int(np.argmax(res_mat[0]))

    prob_no_top2_target = float(res_mat[1, target_class])
    pred_no_top2 = int(np.argmax(res_mat[1]))

    prob_only_top1_target = float(res_mat[2, target_class])

    prob_no_ctrl_target = float(res_mat[3, target_class])
    pred_no_ctrl = int(np.argmax(res_mat[3]))

    return {
        'num_phrases': k,
        'top1_phrase': phrases[top1_idx],
        'drop_top1_target': prob_orig_target - prob_no_top1_target,
        'flip_top1': bool(pred_no_top1 != pred_orig),
        'drop_top2_target': prob_orig_target - prob_no_top2_target,
        'flip_top2': bool(pred_no_top2 != pred_orig),
        'suff_only_top1': prob_orig_target - prob_only_top1_target,
        'drop_ctrl_target': prob_orig_target - prob_no_ctrl_target,
        'flip_ctrl': bool(pred_no_ctrl != pred_orig),
        'prob_no_top1_sens': float(res_mat[0, 1]),
        'prob_no_top2_sens': float(res_mat[1, 1])
    }


# =============================================================================
# 5. EJECUCIÓN DEL BENCHMARK SOBRE LA MUESTRA N=40 (TITULAR Y DOCUMENTO COMPLETO)
# =============================================================================
print("\n[3/7] Ejecutando Benchmark en Régimen A (Titulares N=40) y Régimen B (Noticias Completas N=40)...")

records_regime_a = []  # Titular
records_regime_b = []  # Documento Completo
native_phrases_a = []
native_phrases_b = []

curves_a = {m: {'morf': [], 'lorf': []} for m in [
    '1. Integrated Gradients', '2. Attention Rollout', '3. SHAP (Partition)',
    '4. LIME', '5. Input × Gradient', '6. LRP (Transformer)', '7. Propio (MexGen Frases)'
]}
curves_b = {m: {'morf': [], 'lorf': []} for m in curves_a.keys()}

t_bench_start = time.perf_counter()

for i, row in df_sample40.iterrows():
    sid = row['sample_id']
    doc_id = row['doc_id']
    fuente = row['fuente']
    clase_real = int(row['clase_real'])

    # --- RÉGIMEN A: TITULAR / APERTURA ---
    txt_a = row['texto_titular']
    p_a_mat = predict_proba_texts([txt_a], max_len=128)[0]
    pred_a = int(np.argmax(p_a_mat))

    inp_a, toks_a, attrs_a, lats_a, mex_info_a = explain_all_methods(txt_a, target_class=pred_a, max_len=128)
    nat_a = evaluate_native_phrase_ablation(txt_a, mex_info_a, target_class=pred_a, max_len=128)
    nat_a.update({'sample_id': sid, 'doc_id': doc_id, 'fuente': fuente, 'clase_real': clase_real, 'pred_orig': pred_a, 'regimen': 'Titular (Max 128)'})
    native_phrases_a.append(nat_a)

    for m_idx, (m_name, attr_vec) in enumerate(attrs_a.items()):
        ev = evaluate_token_method(inp_a, toks_a, attr_vec, target_class=pred_a, top_k_ratio=0.20, seed=SEED + i*10 + m_idx)
        curves_a[m_name]['morf'].append(ev['morf_curve'])
        curves_a[m_name]['lorf'].append(ev['lorf_curve'])
        rec = {
            'sample_id': sid, 'doc_id': doc_id, 'fuente': fuente,
            'clase_real': clase_real, 'pred_orig': pred_a,
            'regimen': 'Titular', 'metodo': m_name,
            'latencia_seg': lats_a[m_name],
            **{k: v for k, v in ev.items() if k not in ['morf_curve', 'lorf_curve']}
        }
        records_regime_a.append(rec)

    # --- RÉGIMEN B: DOCUMENTO COMPLETO (TITULAR + COPETE + CUERPO) ---
    txt_b = row['texto_completo']
    p_b_mat = predict_proba_texts([txt_b], max_len=256)[0]
    pred_b = int(np.argmax(p_b_mat))

    inp_b, toks_b, attrs_b, lats_b, mex_info_b = explain_all_methods(txt_b, target_class=pred_b, max_len=256)
    nat_b = evaluate_native_phrase_ablation(txt_b, mex_info_b, target_class=pred_b, max_len=256)
    nat_b.update({'sample_id': sid, 'doc_id': doc_id, 'fuente': fuente, 'clase_real': clase_real, 'pred_orig': pred_b, 'regimen': 'Documento Completo (Max 256)'})
    native_phrases_b.append(nat_b)

    for m_idx, (m_name, attr_vec) in enumerate(attrs_b.items()):
        ev = evaluate_token_method(inp_b, toks_b, attr_vec, target_class=pred_b, top_k_ratio=0.20, seed=SEED + 500 + i*10 + m_idx)
        curves_b[m_name]['morf'].append(ev['morf_curve'])
        curves_b[m_name]['lorf'].append(ev['lorf_curve'])
        rec = {
            'sample_id': sid, 'doc_id': doc_id, 'fuente': fuente,
            'clase_real': clase_real, 'pred_orig': pred_b,
            'regimen': 'Documento Completo', 'metodo': m_name,
            'latencia_seg': lats_b[m_name],
            **{k: v for k, v in ev.items() if k not in ['morf_curve', 'lorf_curve']}
        }
        records_regime_b.append(rec)

    if (i + 1) % 5 == 0 or (i + 1) == len(df_sample40):
        print(f"      Progreso: {i+1}/40 registros evaluados ({time.perf_counter() - t_bench_start:.1f} s)...")

df_reg_a = pd.DataFrame(records_regime_a)
df_reg_b = pd.DataFrame(records_regime_b)
df_nat_a = pd.DataFrame(native_phrases_a)
df_nat_b = pd.DataFrame(native_phrases_b)

df_reg_a.to_csv(os.path.join(DATA_OUT_DIR, "sensacionalismo_benchmark_titulares_40.csv"), index=False)
df_reg_b.to_csv(os.path.join(DATA_OUT_DIR, "sensacionalismo_benchmark_documentos_40.csv"), index=False)
df_nat_b.to_csv(os.path.join(DATA_OUT_DIR, "sensacionalismo_ablacion_nativa_frases_40.csv"), index=False)


# =============================================================================
# 6. SANITY CHECK DE ADEBAYO ET AL. (2018) INCLUYENDO EL MÉTODO PROPIO
# =============================================================================
print("\n[4/7] Ejecutando Sanity Check de Adebayo (Aleatorización en Cascada de Parámetros) para los 7 métodos...")
ref_text_sanity = df_sample40.loc[0, 'texto_titular']
_, toks_san, base_attrs_san, _, _ = explain_all_methods(ref_text_sanity, target_class=1, max_len=128)
methods_list = list(base_attrs_san.keys())
levels_adebayo = ["0_Orig", "1_Classifier", "2_Layer11", "3_Layers10-11", "4_Layers8-11", "5_AllLayers"]

spearman_adebayo = {m: [1.0] for m in methods_list}
cosine_adebayo = {m: [1.0] for m in methods_list}

orig_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

cascade_steps = [
    ("1_Classifier", lambda m: [nn.init.normal_(m.classifier.weight, std=0.02), nn.init.zeros_(m.classifier.bias)]),
    ("2_Layer11", lambda m: m.bert.encoder.layer[11].apply(lambda mod: mod.reset_parameters() if hasattr(mod, 'reset_parameters') else None)),
    ("3_Layers10-11", lambda m: m.bert.encoder.layer[10].apply(lambda mod: mod.reset_parameters() if hasattr(mod, 'reset_parameters') else None)),
    ("4_Layers8-11", lambda m: [m.bert.encoder.layer[ly].apply(lambda mod: mod.reset_parameters() if hasattr(mod, 'reset_parameters') else None) for ly in [8, 9]]),
    ("5_AllLayers", lambda m: [m.bert.encoder.layer[ly].apply(lambda mod: mod.reset_parameters() if hasattr(mod, 'reset_parameters') else None) for ly in range(8)])
]

content_mask_san = [i for i in range(1, len(toks_san) - 1) if toks_san[i] not in ['[CLS]', '[SEP]', '[PAD]']]

for lvl_name, perturb_fn in cascade_steps:
    perturb_fn(model)
    model.eval()
    _, _, alt_attrs_san, _, _ = explain_all_methods(ref_text_sanity, target_class=1, max_len=128)
    for m in methods_list:
        v_o = base_attrs_san[m][content_mask_san]
        v_a = alt_attrs_san[m][content_mask_san]
        if np.std(v_o) < 1e-12 or np.std(v_a) < 1e-12:
            rho = 0.0
        else:
            rho, _ = spearmanr(v_o, v_a)
            if np.isnan(rho):
                rho = 0.0
        spearman_adebayo[m].append(float(rho))
        no, na = np.linalg.norm(v_o), np.linalg.norm(v_a)
        cos_val = float(np.dot(v_o, v_a) / (no * na)) if (no > 0 and na > 0) else 0.0
        cosine_adebayo[m].append(cos_val)

# Restaurar pesos exactos del modelo original
model.load_state_dict(orig_state_dict)
model.eval()
print("      Pesos originales de BETO restaurados exitosamente tras Sanity Check.")


# =============================================================================
# 7. ANÁLISIS ESTADÍSTICO RIGUROSO Y PRUEBAS DE HIPÓTESIS (P-VALORES)
# =============================================================================
print("\n[5/7] Calculando estadísticas inferenciales, intervalos de confianza y significancia (Wilcoxon / t-test)...")

def summarize_regime(df_reg, curves_dict):
    summary = {}
    shapiro_ref = df_reg[df_reg['metodo'] == '3. SHAP (Partition)']['comprehensiveness'].values
    mex_ref = df_reg[df_reg['metodo'] == '7. Propio (MexGen Frases)']['comprehensiveness'].values

    for m in methods_list:
        sub = df_reg[df_reg['metodo'] == m]
        sub_ia = sub[sub['fuente'].str.contains('IA')]
        sub_ama = sub[sub['fuente'].str.contains('Prensa Real')]

        # Subconjunto de titulares/documentos predichos como Sensacionalistas (1 -> 0 flips)
        sub_sens = sub[sub['pred_orig'] == 1]
        sub_ama_sens = sub_ama[sub_ama['pred_orig'] == 1]
        sub_ia_sens = sub_ia[sub_ia['pred_orig'] == 1]

        comp_vals = sub['comprehensiveness'].values
        suff_vals = sub['sufficiency'].values
        dauc_vals = sub['delta_auc'].values
        del_drop_vals = sub['del_drop_target'].values

        # Test Wilcoxon contra 0 (¿Es el cambio de probabilidad significativamente > 0?)
        try:
            _, p_wilcox_zero = wilcoxon(comp_vals, alternative='greater')
        except Exception:
            p_wilcox_zero = 1.0

        # Test pareado contra nuestro método Propio MexGen
        if m != '7. Propio (MexGen Frases)':
            try:
                _, p_vs_mexgen_comp = wilcoxon(comp_vals, mex_ref)
            except Exception:
                p_vs_mexgen_comp = 1.0
        else:
            p_vs_mexgen_comp = 1.0

        summary[m] = {
            'comprehensiveness_mean': float(np.mean(comp_vals)),
            'comprehensiveness_std': float(np.std(comp_vals, ddof=1)),
            'comprehensiveness_median': float(np.median(comp_vals)),
            'comprehensiveness_iqr': float(np.percentile(comp_vals, 75) - np.percentile(comp_vals, 25)),
            'p_val_comp_gt_zero': float(p_wilcox_zero),
            'p_val_vs_mexgen_comp': float(p_vs_mexgen_comp),
            'sufficiency_mean': float(np.mean(suff_vals)),
            'sufficiency_std': float(np.std(suff_vals, ddof=1)),
            'del_drop_mean': float(np.mean(del_drop_vals)),
            'del_drop_std': float(np.std(del_drop_vals, ddof=1)),
            'auc_morf_mean': float(sub['auc_morf'].mean()),
            'auc_lorf_mean': float(sub['auc_lorf'].mean()),
            'delta_auc_mean': float(np.mean(dauc_vals)),
            'delta_auc_std': float(np.std(dauc_vals, ddof=1)),
            'latencia_mean_seg': float(sub['latencia_seg'].mean()),
            'latencia_total_40_seg': float(sub['latencia_seg'].sum()),
            'flips_global_40': {
                'mask': int(sub['mask_flipped'].sum()),
                'del': int(sub['del_flipped'].sum()),
                'noise': int(sub['noise_flipped'].sum()),
                'mask_pct': float(sub['mask_flipped'].mean() * 100),
                'del_pct': float(sub['del_flipped'].mean() * 100),
                'noise_pct': float(sub['noise_flipped'].mean() * 100)
            },
            'flips_ia_20': {
                'mask': int(sub_ia['mask_flipped'].sum()),
                'del': int(sub_ia['del_flipped'].sum()),
                'noise': int(sub_ia['noise_flipped'].sum()),
                'sens_to_nonsens_mask': int(sub_ia_sens['mask_flipped'].sum()),
                'sens_to_nonsens_del': int(sub_ia_sens['del_flipped'].sum()),
                'sens_total': int(len(sub_ia_sens))
            },
            'flips_ama_real_20': {
                'mask': int(sub_ama['mask_flipped'].sum()),
                'del': int(sub_ama['del_flipped'].sum()),
                'noise': int(sub_ama['noise_flipped'].sum()),
                'sens_to_nonsens_mask': int(sub_ama_sens['mask_flipped'].sum()),
                'sens_to_nonsens_del': int(sub_ama_sens['del_flipped'].sum()),
                'sens_total': int(len(sub_ama_sens)),
                'comp_ama_mean': float(sub_ama['comprehensiveness'].mean()),
                'del_drop_ama_mean': float(sub_ama['del_drop_target'].mean())
            },
            'avg_morf_curve': [float(x) for x in np.mean(curves_dict[m]['morf'], axis=0)],
            'avg_lorf_curve': [float(x) for x in np.mean(curves_dict[m]['lorf'], axis=0)]
        }
    return summary

summary_a = summarize_regime(df_reg_a, curves_a)
summary_b = summarize_regime(df_reg_b, curves_b)

# Resumen de Ablación Nativa de Frases (MexGen Oracional Directo) en N=40 y N=272
df_frases_272 = pd.read_csv(EXISTING_FRASES_CSV)
ama_272 = df_frases_272[df_frases_272['fuente'].str.contains('Amarillismo')]
ama_sens_40_full = ama_272[ama_272['pred_full'] == 1]

native_mexgen_summary = {
    'muestra_40_titulares': {
        'drop_top1_mean': float(df_nat_a['drop_top1_target'].mean()),
        'drop_top2_mean': float(df_nat_a['drop_top2_target'].mean()),
        'suff_only_top1_mean': float(df_nat_a['suff_only_top1'].mean()),
        'flips_top1_global': int(df_nat_a['flip_top1'].sum()),
        'flips_top2_global': int(df_nat_a['flip_top2'].sum()),
        'flips_ctrl_global': int(df_nat_a['flip_ctrl'].sum()),
        'flips_top1_ama_real': int(df_nat_a[df_nat_a['fuente'].str.contains('Prensa Real')]['flip_top1'].sum()),
        'flips_top2_ama_real': int(df_nat_a[df_nat_a['fuente'].str.contains('Prensa Real')]['flip_top2'].sum()),
    },
    'muestra_40_documentos_completos': {
        'drop_top1_mean': float(df_nat_b['drop_top1_target'].mean()),
        'drop_top2_mean': float(df_nat_b['drop_top2_target'].mean()),
        'suff_only_top1_mean': float(df_nat_b['suff_only_top1'].mean()),
        'flips_top1_global': int(df_nat_b['flip_top1'].sum()),
        'flips_top2_global': int(df_nat_b['flip_top2'].sum()),
        'flips_ctrl_global': int(df_nat_b['flip_ctrl'].sum()),
        'ama_sens_pred_count': int(len(df_nat_b[(df_nat_b['fuente'].str.contains('Prensa Real')) & (df_nat_b['pred_orig'] == 1)])),
        'ama_sens_flips_top1': int(df_nat_b[(df_nat_b['fuente'].str.contains('Prensa Real')) & (df_nat_b['pred_orig'] == 1)]['flip_top1'].sum()),
        'ama_sens_flips_top2': int(df_nat_b[(df_nat_b['fuente'].str.contains('Prensa Real')) & (df_nat_b['pred_orig'] == 1)]['flip_top2'].sum()),
        'ama_sens_drop_top1_mean': float(df_nat_b[(df_nat_b['fuente'].str.contains('Prensa Real')) & (df_nat_b['pred_orig'] == 1)]['drop_top1_target'].mean()),
        'ama_sens_drop_top2_mean': float(df_nat_b[(df_nat_b['fuente'].str.contains('Prensa Real')) & (df_nat_b['pred_orig'] == 1)]['drop_top2_target'].mean())
    },
    'corpus_completo_272_noticias': {
        'total_noticias': int(len(df_frases_272)),
        'prensa_real_202_pearson_r_mean': float(pearsonr(ama_272['p_full'], ama_272['p_mean'])[0]),
        'prensa_real_202_r2_mean': float(1.0 - np.sum((ama_272['p_full'] - ama_272['p_mean'])**2) / np.sum((ama_272['p_full'] - np.mean(ama_272['p_full']))**2)),
        'prensa_real_sens_40_flips_top1': int(ama_sens_40_full['flip_top1'].sum()),
        'prensa_real_sens_40_flips_top1_pct': float(ama_sens_40_full['flip_top1'].mean() * 100),
        'prensa_real_sens_40_flips_top2': int(ama_sens_40_full['flip_top2'].sum()),
        'prensa_real_sens_40_flips_top2_pct': float(ama_sens_40_full['flip_top2'].mean() * 100),
        'prensa_real_sens_40_flips_rand': int(ama_sens_40_full['flip_rand'].sum()),
        'prensa_real_sens_40_flips_rand_pct': float(ama_sens_40_full['flip_rand'].mean() * 100)
    }
}

full_json_out = {
    'modelo': MODEL_NAME,
    'n_muestra_unificada': 40,
    'regimen_a_titulares_40': summary_a,
    'regimen_b_documentos_40': summary_b,
    'ablacion_oracional_nativa_mexgen': native_mexgen_summary,
    'sanity_check_adebayo': {
        'levels': levels_adebayo,
        'spearman': spearman_adebayo,
        'cosine': cosine_adebayo
    }
}

json_path = os.path.join(DATA_OUT_DIR, "metricas_comparacion_xai_sensacionalismo.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(full_json_out, f, indent=2, ensure_ascii=False)
print(f"      Métricas exportadas a: {json_path}")


# =============================================================================
# 8. GENERACIÓN DE GRÁFICAS CIENTÍFICAS DE ALTA RESOLUCIÓN (300 DPI)
# =============================================================================
print("\n[6/7] Generando figuras comparativas de alta resolución en imagenes/sensacionalismo/...")

short_names = [
    'Int. Gradients', 'Att. Rollout', 'SHAP',
    'LIME', 'Input×Grad', 'LRP (Transf.)', 'Propio (MexGen)'
]
colors_7 = ['#3b82f6', '#94a3b8', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

# --- FIGURA 1: Fidelidad Causal (Comprehensiveness y Sufficiency) en Titulares vs Documentos ---
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6.2))
x = np.arange(len(methods_list))
w = 0.36

for ax_idx, (summ, title_reg) in enumerate([(summary_a, "Régimen A: Titulares / Apertura (N=40)"), (summary_b, "Régimen B: Noticias Completas (N=40)")]):
    ax = axes1[ax_idx]
    comps = [summ[m]['comprehensiveness_mean'] for m in methods_list]
    suffs = [summ[m]['sufficiency_mean'] for m in methods_list]

    b1 = ax.bar(x - w/2, comps, w, label='Comprehensiveness (Erasure Top 20% ↑ Ideal: 1.0)', color='#2563eb', edgecolor='#1e3a8a', alpha=0.9)
    b2 = ax.bar(x + w/2, suffs, w, label='Sufficiency (Retención Top 20% ↓ Ideal: 0.0)', color='#f97316', edgecolor='#7c2d12', alpha=0.9)

    ax.axhline(0, color='#475569', linewidth=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=22, ha='right', fontweight='bold')
    ax.set_ylabel('Cambio en Probabilidad Predicha ΔP')
    ax.set_title(title_reg, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9.5)

    for rect in b1:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.006 if h >= 0 else -0.015
        ax.text(rect.get_x() + rect.get_width()/2, h + offset, f"{h:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold', color='#1e3a8a')
    for rect in b2:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.006 if h >= 0 else -0.015
        ax.text(rect.get_x() + rect.get_width()/2, h + offset, f"{h:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold', color='#9a3412')
    ax.set_ylim(min(min(comps), min(suffs)) - 0.06, max(max(comps), max(suffs)) + 0.07)

fig1.suptitle("Fidelidad Causal Comparativa: 6 XAI de Literatura vs. Explicabilidad Propia (MexGen) en Sensacionalismo", fontsize=14, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1.savefig(os.path.join(IMG_OUT_DIR, "fig1_fidelidad_comprehensiveness_sufficiency.png"), bbox_inches="tight")
plt.close(fig1)

# --- FIGURA 2: Curvas de Ablación Progresiva MoRF vs LoRF y Brecha Causal (ΔAUC) ---
fig2, axes2 = plt.subplots(2, 4, figsize=(17, 8.5))
axes2_flat = axes2.flatten()
ratios_pct = np.linspace(0, 100, 11)

for idx_m, m in enumerate(methods_list):
    ax = axes2_flat[idx_m]
    m_curve = summary_a[m]['avg_morf_curve']
    l_curve = summary_a[m]['avg_lorf_curve']
    d_auc = summary_a[m]['delta_auc_mean']
    auc_m = summary_a[m]['auc_morf_mean']
    auc_l = summary_a[m]['auc_lorf_mean']

    ax.plot(ratios_pct, l_curve, marker='s', markersize=4, linewidth=2.2, color='#10b981', label=f'LoRF (AUC={auc_l:.3f})')
    ax.plot(ratios_pct, m_curve, marker='o', markersize=4, linewidth=2.2, color='#ef4444', label=f'MoRF (AUC={auc_m:.3f})')
    ax.fill_between(ratios_pct, m_curve, l_curve, where=(np.array(l_curve) >= np.array(m_curve)), color='#10b981', alpha=0.18, label=f'ΔAUC = {d_auc:+.3f}')
    ax.fill_between(ratios_pct, m_curve, l_curve, where=(np.array(l_curve) < np.array(m_curve)), color='#ef4444', alpha=0.22)

    ax.set_title(f"{short_names[idx_m]}\n(ΔAUC = {d_auc:+.4f})", fontweight='bold', fontsize=11)
    ax.set_xlabel("% Tokens Enmascarados")
    ax.set_ylabel("Probabilidad Clase Predicha")
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc='lower left', fontsize=8.5, frameon=True, facecolor='white')

# En el 8vo panel ponemos el ranking directo de ΔAUC en Titulares vs Documentos
ax_rank = axes2_flat[7]
dauc_a = [summary_a[m]['delta_auc_mean'] for m in methods_list]
dauc_b = [summary_b[m]['delta_auc_mean'] for m in methods_list]
y_pos = np.arange(len(methods_list))
ax_rank.barh(y_pos - 0.18, dauc_a, height=0.35, color='#6366f1', label='Titulares (N=40)')
ax_rank.barh(y_pos + 0.18, dauc_b, height=0.35, color='#ec4899', label='Doc. Completo (N=40)')
ax_rank.axvline(0, color='black', linewidth=0.9, linestyle='--')
ax_rank.set_yticks(y_pos)
ax_rank.set_yticklabels(short_names, fontweight='bold', fontsize=9.5)
ax_rank.set_xlabel("Brecha Causal ΔAUC (LoRF - MoRF ↑)")
ax_rank.set_title("Resumen Brecha Causal ΔAUC", fontweight='bold', fontsize=11)
ax_rank.legend(loc='lower right', fontsize=8.5, frameon=True)

fig2.suptitle("Curvas de Perturbación Ablativa MoRF vs. LoRF y Brecha Causal (ΔAUC) — Muestra Unificada N=40", fontsize=14, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(IMG_OUT_DIR, "fig2_curvas_morf_lorf_7_metodos.png"), bbox_inches="tight")
plt.close(fig2)

# --- FIGURA 3: Tasa de Cambio de Etiqueta (Prediction Flip Rate) y Caída de Probabilidad ---
fig3, axes3 = plt.subplots(1, 2, figsize=(15.5, 6.2))

# Panel Izquierdo: Flips Globales en Titulares (N=40) bajo Mask, Deletion, Random Noise + Ablación Nativa Frase
ax = axes3[0]
mask_flips = [summary_a[m]['flips_global_40']['mask'] for m in methods_list]
del_flips = [summary_a[m]['flips_global_40']['del'] for m in methods_list]
noise_flips = [summary_a[m]['flips_global_40']['noise'] for m in methods_list]

w3 = 0.26
r1 = ax.bar(x - w3, mask_flips, w3, label='Enmascaramiento [MASK] (Top 20%)', color='#3b82f6', edgecolor='#1d4ed8')
r2 = ax.bar(x, del_flips, w3, label='Eliminación Física (Top 20%)', color='#10b981', edgecolor='#047857')
r3 = ax.bar(x + w3, noise_flips, w3, label='Ruido Vocabulario (Top 20%)', color='#f59e0b', edgecolor='#b45309')

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=22, ha='right', fontweight='bold')
ax.set_ylabel("N.° de Noticias que Cambian de Etiqueta (de 40)")
ax.set_title("A. Cambio de Clase en Muestra N=40 (Perturbación Top 20% Tokens)", fontweight='bold')
ax.legend(loc='upper left', frameon=True, facecolor='white', fontsize=9)
for bars_grp in [r1, r2, r3]:
    for b in bars_grp:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.18, f"{int(h)}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_ylim(0, max(max(mask_flips), max(del_flips), max(noise_flips)) + 2.5)

# Panel Derecho: Prensa Real Amarillista — Comparación Directa de Desclasificación (1 -> 0) Token XAI vs Ablación de Frases Gatillo (MexGen)
ax2 = axes3[1]
labels_comp = [
    'IG (Mask 20%)', 'Rollout (Mask 20%)', 'SHAP (Mask 20%)',
    'LIME (Mask 20%)', 'IxG (Mask 20%)', 'LRP (Mask 20%)',
    'MexGen (Top-1 Frase)\n[Muestra N=40 Doc]', 'MexGen (Top-2 Frases)\n[Muestra N=40 Doc]',
    'MexGen (Top-2 Frases)\n[Corpus Real N=40 Sens]'
]
ama_sens_tot_a = summary_a['1. Integrated Gradients']['flips_ama_real_20']['sens_total']
vals_pct = [
    summary_a['1. Integrated Gradients']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    summary_a['2. Attention Rollout']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    summary_a['3. SHAP (Partition)']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    summary_a['4. LIME']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    summary_a['5. Input × Gradient']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    summary_a['6. LRP (Transformer)']['flips_ama_real_20']['sens_to_nonsens_mask'] / max(1, ama_sens_tot_a) * 100,
    native_mexgen_summary['muestra_40_documentos_completos']['ama_sens_flips_top1'] / max(1, native_mexgen_summary['muestra_40_documentos_completos']['ama_sens_pred_count']) * 100,
    native_mexgen_summary['muestra_40_documentos_completos']['ama_sens_flips_top2'] / max(1, native_mexgen_summary['muestra_40_documentos_completos']['ama_sens_pred_count']) * 100,
    native_mexgen_summary['corpus_completo_272_noticias']['prensa_real_sens_40_flips_top2_pct']
]
bar_cols = ['#60a5fa', '#94a3b8', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f472b6', '#ec4899', '#be185d']
y_p = np.arange(len(labels_comp))
bars_h = ax2.barh(y_p, vals_pct, color=bar_cols, edgecolor='#1e293b', height=0.62)
ax2.set_yticks(y_p)
ax2.set_yticklabels(labels_comp, fontweight='bold', fontsize=9.5)
ax2.set_xlabel("% de Noticias Sensacionalistas Reales que Invierten su Etiqueta (1 → 0)")
ax2.set_title("B. Eficacia Causal en Prensa Real (Desclasificación 1 → 0)", fontweight='bold')
for b, v in zip(bars_h, vals_pct):
    ax2.text(v + 1.0, b.get_y() + b.get_height()/2, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=9.5)
ax2.set_xlim(0, max(vals_pct) + 12)

fig3.suptitle("Análisis de Inversión de Etiqueta (Prediction Flip Rate): Perturbación de Subtokens vs. Ablación de Frases Gatillo", fontsize=13.5, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(IMG_OUT_DIR, "fig3_tasas_inversion_clase_flips.png"), bbox_inches="tight")
plt.close(fig3)

# --- FIGURA 4: Control de Sanidad de Adebayo y Benchmark de Latencia Computacional ---
fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.8))

ax_s = axes4[0]
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
for idx_m, m in enumerate(methods_list):
    ax_s.plot(levels_adebayo, spearman_adebayo[m], marker=markers[idx_m], markersize=7, linewidth=2.2, color=colors_7[idx_m], label=short_names[idx_m])
ax_s.axhline(0, color='#475569', linestyle=':', linewidth=1.2)
ax_s.set_ylabel("Correlación de Rangos de Spearman (ρ)")
ax_s.set_title("A. Control de Sanidad de Adebayo (Aleatorización en Cascada)", fontweight='bold')
ax_s.set_xticklabels(levels_adebayo, rotation=20, ha='right', fontweight='bold')
ax_s.legend(loc='lower left', fontsize=8.5, frameon=True, facecolor='white', ncol=2)
ax_s.set_ylim(-0.75, 1.08)

ax_l = axes4[1]
lats_ms_a = [summary_a[m]['latencia_mean_seg'] * 1000 for m in methods_list]
lats_ms_b = [summary_b[m]['latencia_mean_seg'] * 1000 for m in methods_list]
b_la = ax_l.bar(x - 0.18, lats_ms_a, width=0.35, color='#3b82f6', edgecolor='#1e3a8a', label='Titular (ms/muestra)')
b_lb = ax_l.bar(x + 0.18, lats_ms_b, width=0.35, color='#ec4899', edgecolor='#831843', label='Doc. Completo (ms/muestra)')
ax_l.set_xticks(x)
ax_l.set_xticklabels(short_names, rotation=22, ha='right', fontweight='bold')
ax_l.set_ylabel("Latencia Media en GPU (milisegundos - escala log)")
ax_l.set_yscale('log')
ax_l.set_title("B. Costo Computacional en GPU por Muestra (Escala Log)", fontweight='bold')
ax_l.legend(loc='upper left', frameon=True, facecolor='white')
for b in b_la:
    h = b.get_height()
    ax_l.text(b.get_x() + b.get_width()/2, h * 1.12, f"{h:.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e3a8a')
for b in b_lb:
    h = b.get_height()
    ax_l.text(b.get_x() + b.get_width()/2, h * 1.12, f"{h:.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#831843')
ax_l.set_ylim(1, max(lats_ms_b) * 3.5)

fig4.suptitle("Fiabilidad Mecanística (Test de Adebayo) y Escalabilidad Operativa en GPU", fontsize=13.5, fontweight='bold', y=1.02)
fig4.tight_layout()
fig4.savefig(os.path.join(IMG_OUT_DIR, "fig4_adebayo_sanidad_y_latencia.png"), bbox_inches="tight")
plt.close(fig4)

# --- FIGURA 5: Contraste Estructural Titular vs Documento Completo (Efecto Dilución Contextual vs Efecto Gatillo) ---
fig5, axes5 = plt.subplots(1, 2, figsize=(15, 5.8))

# En las 20 noticias de Prensa Real de la muestra N=40, comparamos P(Sensacionalismo en Titular) vs P(Sensacionalismo en Documento Completo)
sub_a_first = df_reg_a[df_reg_a['metodo'] == '1. Integrated Gradients'].set_index('doc_id')
sub_b_first = df_reg_b[df_reg_b['metodo'] == '1. Integrated Gradients'].set_index('doc_id')
merged_ab = sub_a_first[['fuente', 'clase_real', 'prob_orig_sens', 'pred_orig']].join(
    sub_b_first[['prob_orig_sens', 'pred_orig']], lsuffix='_titular', rsuffix='_doc'
).join(df_nat_b.set_index('doc_id')[['drop_top1_target', 'drop_top2_target', 'flip_top1', 'flip_top2', 'num_phrases']])

ama_ab = merged_ab[merged_ab['fuente'].str.contains('Prensa Real')]
ia_ab = merged_ab[merged_ab['fuente'].str.contains('IA')]

ax5_1 = axes5[0]
ax5_1.scatter(ama_ab['prob_orig_sens_titular'], ama_ab['prob_orig_sens_doc'], c=ama_ab['clase_real'], cmap='coolwarm', s=95, edgecolors='black', alpha=0.9, label='Prensa Real (20)')
ax5_1.scatter(ia_ab['prob_orig_sens_titular'], ia_ab['prob_orig_sens_doc'], c=ia_ab['clase_real'], cmap='coolwarm', marker='^', s=95, edgecolors='black', alpha=0.9, label='IA Sintético (20)')
ax5_1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identidad (Sin Dilución)')
ax5_1.axhline(0.5, color='#ef4444', linestyle=':', alpha=0.7)
ax5_1.axvline(0.5, color='#ef4444', linestyle=':', alpha=0.7)
ax5_1.set_xlabel("P(Sensacionalista) evaluando SOLO el Titular")
ax5_1.set_ylabel("P(Sensacionalista) evaluando el DOCUMENTO COMPLETO")
ax5_1.set_title("A. Efecto Dilución Contextual en la Muestra N=40", fontweight='bold')
ax5_1.annotate("Zona de Dilución Contextual:\nTitular parece Sensacionalista (>0.5)\npero Cuerpo Sobrio lo amortigua (<0.12)",
               xy=(0.76, 0.05), xytext=(0.42, 0.25),
               arrowprops=dict(facecolor='#1e293b', shrink=0.08, width=1.5, headwidth=6),
               fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#fef3c7", ec="#d97706"))
ax5_1.legend(loc='upper left', frameon=True, facecolor='white')

# Panel Derecho: Caída de Probabilidad en Artículos Sensacionalistas Reales (AMA_2, AMA_5, AMA_8, AMA_3, AMA_14)
ax5_2 = axes5[1]
sens_docs_ids = ['AMA_2', 'AMA_5', 'AMA_8', 'AMA_3', 'AMA_14']
x_docs = np.arange(len(sens_docs_ids))
p_orig_docs = [merged_ab.loc[d, 'prob_orig_sens_doc'] for d in sens_docs_ids]
p_shap_docs = [df_reg_b[(df_reg_b['doc_id'] == d) & (df_reg_b['metodo'] == '3. SHAP (Partition)')]['mask_prob_sens'].values[0] for d in sens_docs_ids]
p_mex1_docs = [df_nat_b.set_index('doc_id').loc[d, 'prob_no_top1_sens'] for d in sens_docs_ids]
p_mex2_docs = [df_nat_b.set_index('doc_id').loc[d, 'prob_no_top2_sens'] for d in sens_docs_ids]

w5 = 0.20
ax5_2.bar(x_docs - 1.5*w5, p_orig_docs, w5, label='P(Documento Original)', color='#1e293b')
ax5_2.bar(x_docs - 0.5*w5, p_shap_docs, w5, label='P(Tras SHAP Mask Top 20% Tokens)', color='#10b981')
ax5_2.bar(x_docs + 0.5*w5, p_mex1_docs, w5, label='P(Tras MexGen Ablación Top-1 Frase)', color='#f472b6')
ax5_2.bar(x_docs + 1.5*w5, p_mex2_docs, w5, label='P(Tras MexGen Ablación Top-2 Frases)', color='#be185d')
ax5_2.axhline(0.50, color='#ef4444', linestyle='--', linewidth=1.5, label='Umbral Decisión (0.50)')
ax5_2.set_xticks(x_docs)
ax5_2.set_xticklabels(['AMA_2\n(Apophis)', 'AMA_5\n(Corazón Cerdo)', 'AMA_8\n(Harvard Peso)', 'AMA_3\n(Dinosaurios)', 'AMA_14\n(Megaterremoto)'], fontweight='bold', fontsize=9)
ax5_2.set_ylabel("Probabilidad de Sensacionalismo P(y=1|x)")
ax5_2.set_title("B. Impacto Causal en Noticias Amarillistas Reales de la Muestra", fontweight='bold')
ax5_2.legend(loc='upper left', fontsize=8.2, frameon=True, facecolor='white')
ax5_2.set_ylim(0, 1.12)

fig5.suptitle("Por qué los Métodos Token-Level y la Explicabilidad Propia Oracional (MexGen) Operan en Escalas Complementarias", fontsize=13.5, fontweight='bold', y=1.02)
fig5.tight_layout()
fig5.savefig(os.path.join(IMG_OUT_DIR, "fig5_dilucion_contextual_vs_gatillo_mexgen.png"), bbox_inches="tight")
plt.close(fig5)

print("\n[7/7] ¡Benchmark de Sensacionalismo finalizado con éxito!")
