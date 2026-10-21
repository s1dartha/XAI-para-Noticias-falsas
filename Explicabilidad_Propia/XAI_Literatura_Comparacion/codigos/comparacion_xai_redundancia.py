#!/usr/bin/env python3
"""
=============================================================================
BENCHMARK COMPARATIVO RIGUROSO: 6 MÉTODOS XAI DE LITERATURA (SUBTOKENS/PALABRAS)
VS. EXPLICABILIDAD PROPIA ORACIONAL NO TAUTOLÓGICA (ESTILO MEXGEN) EN REDUNDANCIA
=============================================================================
Modelo Unificado: hiiamsid/sentence_similarity_spanish_es
  (SBERT Clásico con Mean-Pooling sobre BETO dccuchile/bert-base-spanish-wwm-cased, d=768)
Hardware: GPU NVIDIA GeForce GTX 1650 (device='cuda', attn_implementation='eager')

Diseño de Muestra Estrictamente Unificada y Comparable:
  1. Cohorte A — Suite de 20 Pares de Oraciones (10 Curados + 10 Aleatorios del Corpus):
     - Evalúa directamente los 6 métodos XAI de la literatura sobre el par (A, B)
       (Fast-IG, Input×Gradient, Saliency, Attention Centrality, LIME-Light, KernelSHAP-Light)
       junto con nuestro método Propio (MexGen Sub-Clausal / Oracional).
  2. Cohorte B — Muestra Estratificada de 40 Documentos Completos del Corpus (K >= 3 oraciones):
     - 20 Documentos con Redundancia Estricta (max_intra_similarity >= 0.5996: 10 Fake, 10 Real)
     - 20 Documentos No Redundantes (max_intra_similarity < 0.5996: 10 Fake, 10 Real)
     - Sobre EXACTAMENTE los mismos 40 documentos, se evalúan:
       a) Los 6 métodos XAI de la literatura (perturbando el top-20% de tokens del par maximal (i*, j*)
          mediante Enmascaramiento [MASK], Eliminación Física y Ruido de Vocabulario, y re-evaluando
          la matriz de similitud intra-documental completa y el cambio de estado redundante >= 0.60).
       b) Nuestra Explicabilidad Propia Oracional No Tautológica (MexGen):
          - Ablación de la oración crítica s* del par maximal y análisis del segundo estadístico de orden S_(2)
            (diagnóstico forense Bucle Único vs. Red Multi-Bucle Persistente en Fake vs. Real).
          - Ablación No Tautológica sobre la Similitud Media Global del Documento (S_bar_doc) al retirar
            la oración de máxima centralidad semántica (s_hub, MoRF Oracional) vs. Control Aleatorio Irrestricto (1/K)
            vs. Oración Periférica (s_min, LoRF Oracional).
  3. Referencia Poblacional Completa: N=1,820 documentos elegibles del corpus (N=712 con Redundancia Estricta >= 0.60).

Autor: Investigación Doctoral / Maestría en Ciencia de Datos y PLN
Fecha: Octubre 2026
=============================================================================
"""

import os
import gc
import re
import time
import math
import copy
import json
import warnings
import numpy as np
import pandas as pd
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    spearmanr, pearsonr, wilcoxon, ttest_rel, fisher_exact, mannwhitneyu
)
from sklearn.linear_model import Ridge, LinearRegression

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TESIS_ROOT = os.path.abspath(os.path.join(COMP_ROOT, "..", ".."))
RED_ROOT = os.path.join(TESIS_ROOT, "Modelos_Individuales", "Redundancia")

REPORT_DIR = os.path.join(COMP_ROOT, "reportes")
DATA_OUT_DIR = os.path.join(REPORT_DIR, "datos")
IMG_OUT_DIR = os.path.join(REPORT_DIR, "imagenes", "redundancia")

os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

DATASET_SIM_PATH = os.path.join(RED_ROOT, "Dataset", "dataset_con_similitudes.csv")
MEXGEN_FULL_CSV = os.path.join(RED_ROOT, "Reportes", "Umbral_Redundancia", "validacion_explicabilidad_mexgen.csv")
CURATED_JSON_PATH = os.path.join(RED_ROOT, "Reportes", "XAI", "xai_experimental_results.json")

TAU_STRICT = 0.5996
TAU_PAWSX = 0.5340
TAU_HYPER = 0.7287

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Dispositivo de ejecución: {DEVICE}")


# =============================================================================
# 1. CARGA DEL MODELO GANADOR BETO SBERT (d=768)
# =============================================================================
MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"
print(f"\n[1/7] Cargando modelo siamés Sentence-BERT: {MODEL_NAME} (attn_implementation='eager')...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation='eager').to(DEVICE)
model.eval()
emb_layer = model.get_input_embeddings()


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)


def encode_texts(text_list, curr_model=model, max_len=128):
    if isinstance(text_list, str):
        text_list = [text_list]
    clean_list = [str(t).strip() if str(t).strip() else "[PAD]" for t in text_list]
    tok = tokenizer(clean_list, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(DEVICE)
    with torch.no_grad():
        out = curr_model(**tok)[0]
        vecs = F.normalize(mean_pool(out, tok['attention_mask']), p=2, dim=1)
    return vecs


def compute_similarity(text_a, text_b, curr_model=model):
    vecs = encode_texts([text_a, text_b], curr_model=curr_model)
    return float((vecs[0] * vecs[1]).sum().item())


# =============================================================================
# 2. IMPLEMENTACIÓN DE LOS 6 MÉTODOS XAI DE LITERATURA + MEXGEN PROPIO
# =============================================================================
def explain_pair_all_methods(text_a, text_b, curr_model=model):
    """
    Calcula las atribuciones de importancia para el par (text_a, text_b) bajo:
    1. Fast Integrated Gradients (Fast-IG, 10 pasos)
    2. Input × Gradient (IxG)
    3. Saliency Vanilla (Norma L2 de Gradiente)
    4. Attention Centrality (Capa 12)
    5. LIME-Light (25 perturbaciones Ridge)
    6. KernelSHAP-Light (25 coaliciones de Shapley)
    7. Propio (MexGen Sub-Clausal / Sintagmático proyectado a tokens)
    """
    tok_a = tokenizer(text_a, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    tok_b = tokenizer(text_b, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    tokens_a = tokenizer.convert_ids_to_tokens(tok_a['input_ids'][0])
    tokens_b = tokenizer.convert_ids_to_tokens(tok_b['input_ids'][0])
    len_a, len_b = len(tokens_a), len(tokens_b)

    curr_emb_layer = curr_model.get_input_embeddings()
    latencias = {}
    attrs = {}

    # --- 2 & 3. Input × Gradient y Saliency Vanilla ---
    t0 = time.perf_counter()
    curr_model.zero_grad()
    embeds_a = curr_emb_layer(tok_a['input_ids']).clone().detach().requires_grad_(True)
    embeds_b = curr_emb_layer(tok_b['input_ids']).clone().detach().requires_grad_(True)
    out_a = curr_model(inputs_embeds=embeds_a, attention_mask=tok_a['attention_mask'])[0]
    out_b = curr_model(inputs_embeds=embeds_b, attention_mask=tok_b['attention_mask'])[0]
    vec_a = F.normalize(mean_pool(out_a, tok_a['attention_mask']), p=2, dim=1)
    vec_b = F.normalize(mean_pool(out_b, tok_b['attention_mask']), p=2, dim=1)
    sim_val = (vec_a * vec_b).sum()
    sim_val.backward()

    grad_a = embeds_a.grad[0]
    grad_b = embeds_b.grad[0]
    sal_a = grad_a.norm(dim=-1).detach().cpu().numpy()
    sal_b = grad_b.norm(dim=-1).detach().cpu().numpy()
    ixg_a = (embeds_a.detach()[0] * grad_a).sum(dim=-1).cpu().numpy()
    ixg_b = (embeds_b.detach()[0] * grad_b).sum(dim=-1).cpu().numpy()
    t_grad = time.perf_counter() - t0

    attrs['2. Input × Gradient'] = (ixg_a, ixg_b)
    attrs['3. Saliency Vanilla'] = (sal_a, sal_b)
    latencias['2. Input × Gradient'] = t_grad
    latencias['3. Saliency Vanilla'] = t_grad * 0.5

    # --- 1. Fast Integrated Gradients (10 pasos) ---
    t0 = time.perf_counter()
    steps = 10
    ea_base = embeds_a.detach()
    eb_base = embeds_b.detach()
    zero_a = torch.zeros_like(ea_base)
    zero_b = torch.zeros_like(eb_base)
    vec_b_fixed = vec_b.detach()
    vec_a_fixed = vec_a.detach()

    accum_ga = torch.zeros_like(ea_base)
    for st in range(1, steps + 1):
        alpha = st / steps
        ia = (zero_a + alpha * (ea_base - zero_a)).clone().detach().requires_grad_(True)
        oa = curr_model(inputs_embeds=ia, attention_mask=tok_a['attention_mask'])[0]
        va = F.normalize(mean_pool(oa, tok_a['attention_mask']), p=2, dim=1)
        s = (va * vec_b_fixed).sum()
        s.backward()
        accum_ga += ia.grad
    ig_a = ((ea_base - zero_a) * (accum_ga / steps)).sum(dim=-1)[0].detach().cpu().numpy()

    accum_gb = torch.zeros_like(eb_base)
    for st in range(1, steps + 1):
        alpha = st / steps
        ib = (zero_b + alpha * (eb_base - zero_b)).clone().detach().requires_grad_(True)
        ob = curr_model(inputs_embeds=ib, attention_mask=tok_b['attention_mask'])[0]
        vb = F.normalize(mean_pool(ob, tok_b['attention_mask']), p=2, dim=1)
        s = (vec_a_fixed * vb).sum()
        s.backward()
        accum_gb += ib.grad
    ig_b = ((eb_base - zero_b) * (accum_gb / steps)).sum(dim=-1)[0].detach().cpu().numpy()

    attrs['1. Fast-IG (10 pasos)'] = (ig_a, ig_b)
    latencias['1. Fast-IG (10 pasos)'] = time.perf_counter() - t0

    # --- 4. Attention Centrality (Capa 12) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        oa_att = curr_model(**tok_a, output_attentions=True)
        ob_att = curr_model(**tok_b, output_attentions=True)
    att_a = oa_att.attentions[-1][0].mean(dim=0).cpu().numpy()
    att_b = ob_att.attentions[-1][0].mean(dim=0).cpu().numpy()
    att_score_a = att_a.sum(axis=0) / max(1, att_a.shape[0])
    att_score_b = att_b.sum(axis=0) / max(1, att_b.shape[0])
    attrs['4. Attention (Capa 12)'] = (att_score_a, att_score_b)
    latencias['4. Attention (Capa 12)'] = time.perf_counter() - t0

    # Función auxiliar para proyectar pesos de palabras a subtokens de BERT
    def project_words_to_tokens(words, word_coefs, tokens):
        score_tok = np.zeros(len(tokens))
        for idx_t in range(1, len(tokens) - 1):
            clean_t = tokens[idx_t].replace('##', '').strip('.,;:!?¡¿"\'()').lower()
            if not clean_t:
                continue
            for w, val in zip(words, word_coefs):
                cw = w.strip('.,;:!?¡¿"\'()').lower()
                if clean_t in cw or cw in clean_t:
                    score_tok[idx_t] = float(val)
                    break
        return score_tok

    # --- 5. LIME-Light (25 perturbaciones en lote en GPU) ---
    t0 = time.perf_counter()
    def run_lime_side(target_txt, ref_vec, n_samples=25):
        words = target_txt.split()
        n = len(words)
        if n == 0:
            return [], np.array([])
        rng = np.random.RandomState(SEED)
        masks = rng.binomial(1, 0.7, size=(n_samples, n))
        masks[0] = 1
        pert_txts = [' '.join([words[i] for i in range(n) if m[i] == 1]) or '[PAD]' for m in masks]
        p_vecs = encode_texts(pert_txts, curr_model=curr_model)
        sims = (p_vecs @ ref_vec.T).squeeze(-1).cpu().numpy()
        dists = np.sum(1 - masks, axis=1) / max(n, 1)
        weights = np.exp(-(dists ** 2) / 0.25)
        reg = Ridge(alpha=1.0)
        reg.fit(masks, sims, sample_weight=weights)
        return words, reg.coef_

    wa, ca = run_lime_side(text_a, vec_b_fixed)
    wb, cb = run_lime_side(text_b, vec_a_fixed)
    lime_tok_a = project_words_to_tokens(wa, ca, tokens_a)
    lime_tok_b = project_words_to_tokens(wb, cb, tokens_b)
    attrs['5. LIME-Light'] = (lime_tok_a, lime_tok_b)
    latencias['5. LIME-Light'] = time.perf_counter() - t0

    # --- 6. KernelSHAP-Light (25 coaliciones en lote en GPU) ---
    t0 = time.perf_counter()
    def run_shap_side(target_txt, ref_vec, n_samples=25):
        words = target_txt.split()
        n = len(words)
        if n == 0:
            return [], np.array([])
        rng = np.random.RandomState(SEED)
        masks = rng.binomial(1, 0.5, size=(n_samples, n))
        masks[0] = 1
        if n_samples > 1:
            masks[1] = 0
        pert_txts = [' '.join([words[i] for i in range(n) if m[i] == 1]) or '[PAD]' for m in masks]
        p_vecs = encode_texts(pert_txts, curr_model=curr_model)
        sims = (p_vecs @ ref_vec.T).squeeze(-1).cpu().numpy()
        weights = []
        for m in masks:
            k = int(np.sum(m))
            if k == 0 or k == n:
                weights.append(1000.0)
            else:
                weights.append((n - 1) / (math.comb(n, k) * k * (n - k) + 1e-9))
        reg = LinearRegression()
        reg.fit(masks, sims, sample_weight=weights)
        return words, reg.coef_

    swa, sca = run_shap_side(text_a, vec_b_fixed)
    swb, scb = run_shap_side(text_b, vec_a_fixed)
    shap_tok_a = project_words_to_tokens(swa, sca, tokens_a)
    shap_tok_b = project_words_to_tokens(swb, scb, tokens_b)
    attrs['6. KernelSHAP-Light'] = (shap_tok_a, shap_tok_b)
    latencias['6. KernelSHAP-Light'] = time.perf_counter() - t0

    # --- 7. Propio MexGen (Ablación de Cláusulas / Sintagmas y Alineación Semántica) ---
    t0 = time.perf_counter()
    def split_clauses(txt):
        pieces = [p.strip() for p in re.split(r'[,;:.!?¡¿—–()]+|\b(?:que|donde|cuando|como|para|por|con|según|mientras)\b', txt, flags=re.IGNORECASE) if len(p.strip().split()) >= 2]
        return pieces if len(pieces) >= 2 else txt.split()

    clauses_a = split_clauses(text_a)
    clauses_b = split_clauses(text_b)
    va_clauses = encode_texts(clauses_a, curr_model=curr_model)
    vb_clauses = encode_texts(clauses_b, curr_model=curr_model)
    sim_ca = (va_clauses @ vec_b_fixed.T).squeeze(-1).cpu().numpy()
    sim_cb = (vb_clauses @ vec_a_fixed.T).squeeze(-1).cpu().numpy()

    mex_tok_a = project_words_to_tokens(clauses_a, sim_ca, tokens_a)
    mex_tok_b = project_words_to_tokens(clauses_b, sim_cb, tokens_b)
    attrs['7. Propio (MexGen)'] = (mex_tok_a, mex_tok_b)
    latencias['7. Propio (MexGen)'] = time.perf_counter() - t0

    return tok_a, tok_b, tokens_a, tokens_b, float(sim_val.item()), attrs, latencias


# =============================================================================
# 3. EVALUACIÓN DE FIDELIDAD Y PERTURBACIONES SOBRE UN PAR (A, B)
# =============================================================================
def evaluate_pair_perturbations(tok_a, tok_b, tokens_a, tokens_b, base_sim, score_a, score_b, top_k_ratio=0.20, seed=42):
    """
    Evalúa en el par (A, B):
    - Comprehensiveness (Eliminación / Erasure Top 20% como en Reporte_XAI_Redundancia.md)
    - Sufficiency (Retención Top 20%)
    - Curvas MoRF y LoRF (0%, 20%, 40%, 60%, 80%) y ΔAUC
    - 3 Estrategias de Perturbación al Top 20% (Mask [MASK], Token Deletion, Random Vocab Noise)
      midiendo tanto ΔSimilitud como el cambio de estado de Redundancia Estricta (Sim >= 0.5996 -> < 0.5996).
    """
    mask_id = tokenizer.mask_token_id
    ids_a, mask_a = tok_a['input_ids'], tok_a['attention_mask']
    ids_b, mask_b = tok_b['input_ids'], tok_b['attention_mask']

    cont_a = [i for i in range(1, len(tokens_a) - 1) if tokens_a[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    cont_b = [i for i in range(1, len(tokens_b) - 1) if tokens_b[i] not in ['[CLS]', '[SEP]', '[PAD]']]

    ka = max(1, int(math.ceil(top_k_ratio * len(cont_a))))
    kb = max(1, int(math.ceil(top_k_ratio * len(cont_b))))

    morf_a = [idx for idx, _ in sorted([(i, float(score_a[i])) for i in cont_a], key=lambda x: x[1], reverse=True)]
    morf_b = [idx for idx, _ in sorted([(i, float(score_b[i])) for i in cont_b], key=lambda x: x[1], reverse=True)]
    lorf_a = list(reversed(morf_a))
    lorf_b = list(reversed(morf_b))

    top_a_set = set(morf_a[:ka])
    top_b_set = set(morf_b[:kb])

    def sim_from_ids(ia, ma, ib, mb):
        with torch.no_grad():
            oa = model(input_ids=ia, attention_mask=ma)[0]
            ob = model(input_ids=ib, attention_mask=mb)[0]
            va = F.normalize(mean_pool(oa, ma), p=2, dim=1)
            vb = F.normalize(mean_pool(ob, mb), p=2, dim=1)
            return float((va * vb).sum().item())

    # 1. Perturbación por Eliminación Física (Deletion / Erasure canónico en SBERT)
    keep_a = [i for i in range(len(tokens_a)) if i not in top_a_set]
    keep_b = [i for i in range(len(tokens_b)) if i not in top_b_set]
    sim_del = sim_from_ids(ids_a[:, keep_a], mask_a[:, keep_a], ids_b[:, keep_b], mask_b[:, keep_b])
    comp_del = base_sim - sim_del

    # 2. Sufficiency (Retener solo Top 20% tokens + [CLS]/[SEP])
    suff_keep_a = [0] + sorted(list(top_a_set)) + [len(tokens_a) - 1]
    suff_keep_b = [0] + sorted(list(top_b_set)) + [len(tokens_b) - 1]
    sim_suff = sim_from_ids(ids_a[:, suff_keep_a], mask_a[:, suff_keep_a], ids_b[:, suff_keep_b], mask_b[:, suff_keep_b])
    suff_gap = abs(base_sim - sim_suff)

    # 3. Perturbación por Enmascaramiento ([MASK])
    im_a, im_b = ids_a.clone(), ids_b.clone()
    for idx in top_a_set:
        im_a[0, idx] = mask_id
    for idx in top_b_set:
        im_b[0, idx] = mask_id
    sim_mask = sim_from_ids(im_a, mask_a, im_b, mask_b)
    comp_mask = base_sim - sim_mask

    # 4. Perturbación por Ruido Aleatorio de Vocabulario
    rng = np.random.RandomState(seed)
    in_a, in_b = ids_a.clone(), ids_b.clone()
    for idx, rt in zip(sorted(list(top_a_set)), rng.randint(6, tokenizer.vocab_size, size=len(top_a_set))):
        in_a[0, idx] = int(rt)
    for idx, rt in zip(sorted(list(top_b_set)), rng.randint(6, tokenizer.vocab_size, size=len(top_b_set))):
        in_b[0, idx] = int(rt)
    sim_noise = sim_from_ids(in_a, mask_a, in_b, mask_b)
    comp_noise = base_sim - sim_noise

    # 5. Curvas MoRF y LoRF (0, 20, 40, 60, 80%)
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    morf_sims = [base_sim]
    lorf_sims = [base_sim]
    for frac in fractions[1:]:
        n_a = int(math.ceil(frac * len(cont_a)))
        n_b = int(math.ceil(frac * len(cont_b)))
        ma_set, mb_set = set(morf_a[:n_a]), set(morf_b[:n_b])
        la_set, lb_set = set(lorf_a[:n_a]), set(lorf_b[:n_b])

        ka_m = [i for i in range(len(tokens_a)) if i not in ma_set]
        kb_m = [i for i in range(len(tokens_b)) if i not in mb_set]
        morf_sims.append(sim_from_ids(ids_a[:, ka_m], mask_a[:, ka_m], ids_b[:, kb_m], mask_b[:, kb_m]))

        ka_l = [i for i in range(len(tokens_a)) if i not in la_set]
        kb_l = [i for i in range(len(tokens_b)) if i not in lb_set]
        lorf_sims.append(sim_from_ids(ids_a[:, ka_l], mask_a[:, ka_l], ids_b[:, kb_l], mask_b[:, kb_l]))

    auc_morf = float(np.trapezoid(morf_sims, dx=0.2))
    auc_lorf = float(np.trapezoid(lorf_sims, dx=0.2))
    delta_auc = auc_lorf - auc_morf

    orig_redundant = (base_sim >= TAU_STRICT)
    return {
        'base_sim': base_sim,
        'orig_redundant': bool(orig_redundant),
        'comprehensiveness_del': comp_del,
        'comprehensiveness_mask': comp_mask,
        'comprehensiveness_noise': comp_noise,
        'sufficiency': suff_gap,
        'sim_del': sim_del,
        'sim_mask': sim_mask,
        'sim_noise': sim_noise,
        'flip_del': bool((sim_del >= TAU_STRICT) != orig_redundant),
        'flip_mask': bool((sim_mask >= TAU_STRICT) != orig_redundant),
        'flip_noise': bool((sim_noise >= TAU_STRICT) != orig_redundant),
        'auc_morf': auc_morf,
        'auc_lorf': auc_lorf,
        'delta_auc': delta_auc,
        'morf_curve': morf_sims,
        'lorf_curve': lorf_sims
    }


# =============================================================================
# 4. COHORTE A: EVALUACIÓN SOBRE LOS 20 PARES (10 CURADOS + 10 ALEATORIOS)
# =============================================================================
print("\n[2/7] Cargando y evaluando Cohorte A: 20 Pares de Oraciones (10 Curados + 10 Aleatorios del Corpus)...")

with open(CURATED_JSON_PATH, 'r', encoding='utf-8') as f:
    curated_data = json.load(f)['pairs_data']

pairs_cohort = []
for p in curated_data:
    pairs_cohort.append({
        'pair_id': f"CUR_{p['id']:02d}",
        'subset': 'Curados (10)',
        'category': p['type'],
        'text_a': p['text_a'],
        'text_b': p['text_b']
    })

# Extraer los mismos 10 pares aleatorios del corpus (semilla 42)
df_corpus = pd.read_csv(DATASET_SIM_PATH)

high_cand = []
for idx, row in df_corpus[df_corpus['max_intra_similarity'] >= 0.75].iterrows():
    sents = [s.strip() for s in nltk.sent_tokenize(str(row['Text']), language='spanish') if 25 <= len(s.strip()) <= 260]
    if len(sents) < 2:
        continue
    vecs = encode_texts(sents)
    smat = (vecs @ vecs.T).cpu().numpy()
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            if 0.75 <= smat[i, j] <= 0.98:
                high_cand.append({'doc_id': int(idx), 'category': 'Redundante', 'text_a': sents[i], 'text_b': sents[j]})
    if len(high_cand) >= 35:
        break

low_cand = []
for idx, row in df_corpus[(df_corpus['mean_intra_similarity'] < 0.35) & (df_corpus['num_sentences'] >= 3)].sample(60, random_state=SEED).iterrows():
    sents = [s.strip() for s in nltk.sent_tokenize(str(row['Text']), language='spanish') if 25 <= len(s.strip()) <= 260]
    if len(sents) < 2:
        continue
    vecs = encode_texts(sents)
    smat = (vecs @ vecs.T).cpu().numpy()
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            if 0.15 <= smat[i, j] <= 0.38:
                low_cand.append({'doc_id': int(idx), 'category': 'No Redundante', 'text_a': sents[i], 'text_b': sents[j]})
    if len(low_cand) >= 35:
        break

rng_sel = np.random.RandomState(SEED)
sel_h = [high_cand[i] for i in rng_sel.choice(len(high_cand), size=5, replace=False)]
sel_l = [low_cand[i] for i in rng_sel.choice(len(low_cand), size=5, replace=False)]
for idx_r, p in enumerate(sel_h + sel_l, 1):
    pairs_cohort.append({
        'pair_id': f"RND_{idx_r:02d}",
        'subset': 'Aleatorios Corpus (10)',
        'category': p['category'],
        'text_a': p['text_a'],
        'text_b': p['text_b']
    })

records_pairs_20 = []
curves_pairs_20 = {}

for i, p in enumerate(pairs_cohort):
    tok_a, tok_b, ta, tb, b_sim, attrs_p, lats_p = explain_pair_all_methods(p['text_a'], p['text_b'])
    for m_idx, (m_name, (sa, sb)) in enumerate(attrs_p.items()):
        if m_name not in curves_pairs_20:
            curves_pairs_20[m_name] = {'morf': [], 'lorf': []}
        ev = evaluate_pair_perturbations(tok_a, tok_b, ta, tb, b_sim, sa, sb, top_k_ratio=0.20, seed=SEED + i*10 + m_idx)
        curves_pairs_20[m_name]['morf'].append(ev['morf_curve'])
        curves_pairs_20[m_name]['lorf'].append(ev['lorf_curve'])
        records_pairs_20.append({
            'pair_id': p['pair_id'],
            'subset': p['subset'],
            'category': p['category'],
            'metodo': m_name,
            'latencia_ms': lats_p[m_name] * 1000.0,
            **{k: v for k, v in ev.items() if k not in ['morf_curve', 'lorf_curve']}
        })

df_pairs_20 = pd.DataFrame(records_pairs_20)
df_pairs_20.to_csv(os.path.join(DATA_OUT_DIR, "redundancia_benchmark_20_pares.csv"), index=False)
print("      Cohorte A (20 Pares: 10 Curados + 10 Aleatorios) completada.")


# =============================================================================
# 5. COHORTE B: MUESTRA UNIFICADA DE N=40 DOCUMENTOS COMPLETOS (LITERATURA VS MEXGEN)
# =============================================================================
print("\n[3/7] Construyendo y evaluando Cohorte B: 40 Documentos Completos Balanceados del Corpus...")
print("      (20 con Redundancia Estricta >= 0.60 [10 Fake, 10 Real] + 20 No Redundantes [10 Fake, 10 Real])")

# Seleccionamos documentos con >= 4 oraciones bien formadas
eligible_docs = []
for idx, row in df_corpus[df_corpus['num_sentences'] >= 4].iterrows():
    sents = [s.strip() for s in nltk.sent_tokenize(str(row['Text']), language='spanish') if len(s.strip()) > 12]
    if 4 <= len(sents) <= 20:
        eligible_docs.append({
            'doc_id': int(idx),
            'label': int(row['class']),
            'max_sim_pre': float(row['max_intra_similarity']),
            'sents': sents
        })
df_elig = pd.DataFrame(eligible_docs)

# Muestreo estratificado exacto con semilla 42:
# En los 20 redundantes incluimos tanto casos de bucle único como casos multi-bucle representativos del corpus
red_fake = df_elig[(df_elig['max_sim_pre'] >= 0.62) & (df_elig['label'] == 1)].sample(10, random_state=SEED)
red_real = df_elig[(df_elig['max_sim_pre'] >= 0.62) & (df_elig['label'] == 0)].sample(10, random_state=SEED)
nonred_fake = df_elig[(df_elig['max_sim_pre'] < 0.52) & (df_elig['max_sim_pre'] >= 0.35) & (df_elig['label'] == 1)].sample(10, random_state=SEED)
nonred_real = df_elig[(df_elig['max_sim_pre'] < 0.52) & (df_elig['max_sim_pre'] >= 0.35) & (df_elig['label'] == 0)].sample(10, random_state=SEED)

df_sample_docs40 = pd.concat([red_fake, red_real, nonred_fake, nonred_real], ignore_index=True)

doc40_token_records = []
doc40_mexgen_records = []

t_doc_start = time.perf_counter()

for i, row in df_sample_docs40.iterrows():
    doc_id = int(row['doc_id'])
    label = int(row['label'])
    sents = row['sents']
    k = len(sents)

    # Matriz de similitud original del documento (k x k)
    vecs_orig = encode_texts(sents)
    sim_mat = (vecs_orig @ vecs_orig.T).cpu().numpy()
    triu_idx = np.triu_indices(k, k=1)
    pair_sims = sim_mat[triu_idx]
    sorted_pairs = np.sort(pair_sims)[::-1]

    s_max1 = float(sorted_pairs[0])
    s_max2 = float(sorted_pairs[1]) if len(sorted_pairs) >= 2 else s_max1
    s_mean_orig = float(np.mean(pair_sims))
    orig_is_redundant = (s_max1 >= TAU_STRICT)

    # Identificar el par maximal (i*, j*) y la oración crítica s* de mayor centralidad
    max_pair_pos = np.unravel_index(np.argmax(sim_mat - np.eye(k) * 999.0), sim_mat.shape)
    i_star, j_star = int(max_pair_pos[0]), int(max_pair_pos[1])
    row_means = np.array([(np.sum(sim_mat[idx]) - sim_mat[idx, idx]) / (k - 1) for idx in range(k)])
    s_star = i_star if row_means[i_star] >= row_means[j_star] else j_star
    s_second_in_pair = j_star if s_star == i_star else i_star
    s_hub = int(np.argmax(row_means))
    s_min = int(np.argmin(row_means))

    # --- A. EVALUACIÓN DE EXPLICABILIDAD PROPIA ORACIONAL (MEXGEN NO TAUTOLÓGICO) EN EL DOCUMENTO ---
    t_mex0 = time.perf_counter()
    # 1. Ablación de 1 oración crítica s* (Máximo residual S_(2) fuera de s*)
    keep_no_star = [idx for idx in range(k) if idx != s_star]
    sub_no_star = sim_mat[np.ix_(keep_no_star, keep_no_star)]
    s_max_after_1 = float(np.max(sub_no_star[np.triu_indices(len(keep_no_star), k=1)]))
    drop_max_mex1 = s_max1 - s_max_after_1
    flip_mex1 = (orig_is_redundant != (s_max_after_1 >= TAU_STRICT))

    # 2. Ablación de hasta 2 oraciones críticas (s* y la oración del segundo par maximal si persiste bucle)
    max2_pos = np.unravel_index(np.argmax(sub_no_star - np.eye(len(keep_no_star)) * 999.0), sub_no_star.shape)
    s_star2_local = int(max2_pos[0])
    s_star2_global = keep_no_star[s_star2_local]
    keep_no_2 = [idx for idx in keep_no_star if idx != s_star2_global]
    if len(keep_no_2) >= 2:
        sub_no_2 = sim_mat[np.ix_(keep_no_2, keep_no_2)]
        s_max_after_2 = float(np.max(sub_no_2[np.triu_indices(len(keep_no_2), k=1)]))
    else:
        s_max_after_2 = s_max_after_1
    drop_max_mex2 = s_max1 - s_max_after_2
    flip_mex2 = (orig_is_redundant != (s_max_after_2 >= TAU_STRICT))

    # 3. Control Aleatorio Irrestricto (1/K) sobre S_max
    max_rand_list = []
    for r_idx in range(k):
        kr = [idx for idx in range(k) if idx != r_idx]
        sr = sim_mat[np.ix_(kr, kr)]
        max_rand_list.append(float(np.max(sr[np.triu_indices(len(kr), k=1)])))
    s_max_rand_unres = float(np.mean(max_rand_list))
    drop_max_rand_unres = s_max1 - s_max_rand_unres

    # 4. Ablación No Tautológica sobre la Similitud Media Global (S_bar_doc)
    keep_no_hub = [idx for idx in range(k) if idx != s_hub]
    sub_no_hub = sim_mat[np.ix_(keep_no_hub, keep_no_hub)]
    s_mean_after_hub = float(np.mean(sub_no_hub[np.triu_indices(len(keep_no_hub), k=1)]))
    delta_mean_morf = s_mean_orig - s_mean_after_hub

    keep_no_min = [idx for idx in range(k) if idx != s_min]
    sub_no_min = sim_mat[np.ix_(keep_no_min, keep_no_min)]
    s_mean_after_min = float(np.mean(sub_no_min[np.triu_indices(len(keep_no_min), k=1)]))
    delta_mean_lorf = s_mean_orig - s_mean_after_min
    lat_mex_doc = (time.perf_counter() - t_mex0) * 1000.0

    doc40_mexgen_records.append({
        'doc_id': doc_id, 'label': label,
        'clase_noticia': 'Falsa (1)' if label == 1 else 'Real (0)',
        'estrato': 'Redundante (>=0.60)' if orig_is_redundant else 'No Redundante (<0.60)',
        'num_sentences': k,
        's_max1': s_max1, 's_max2': s_max2,
        's_mean_orig': s_mean_orig,
        's_max_after_1_sent': s_max_after_1,
        'drop_max_1_sent': drop_max_mex1,
        'flip_1_sent': bool(flip_mex1),
        's_max_after_2_sents': s_max_after_2,
        'drop_max_2_sents': drop_max_mex2,
        'flip_2_sents': bool(flip_mex2),
        'drop_max_rand_unres': drop_max_rand_unres,
        'delta_mean_morf_hub': delta_mean_morf,
        'delta_mean_lorf_min': delta_mean_lorf,
        'multi_loop_persistent': bool(orig_is_redundant and (s_max_after_1 >= TAU_STRICT)),
        'latencia_ms': lat_mex_doc
    })

    # --- B. EVALUACIÓN DE LOS 6 MÉTODOS XAI DE LITERATURA SOBRE EL PAR MAXIMAL (i*, j*) DEL MISMO DOCUMENTO ---
    sent_a, sent_b = sents[i_star], sents[j_star]
    tok_a, tok_b, ta, tb, b_sim, attrs_doc, lats_doc = explain_pair_all_methods(sent_a, sent_b)

    for m_idx, (m_name, (sa, sb)) in enumerate(attrs_doc.items()):
        ev_pair = evaluate_pair_perturbations(tok_a, tok_b, ta, tb, b_sim, sa, sb, top_k_ratio=0.20, seed=SEED + 1000 + i*10 + m_idx)

        # ¿Qué pasa con el máximo global del documento S_max(D) cuando el par maximal (i*, j*) es perturbado al top-20% de tokens?
        # Reconstruimos los vectores de sent_a y sent_b tras eliminación del top-20% de tokens y re-calculamos la matriz del documento
        cont_a = [ix for ix in range(1, len(ta) - 1) if ta[ix] not in ['[CLS]', '[SEP]', '[PAD]']]
        cont_b = [ix for ix in range(1, len(tb) - 1) if tb[ix] not in ['[CLS]', '[SEP]', '[PAD]']]
        ka = max(1, int(math.ceil(0.20 * len(cont_a))))
        kb = max(1, int(math.ceil(0.20 * len(cont_b))))
        top_a = set([ix for ix, _ in sorted([(ix, float(sa[ix])) for ix in cont_a], key=lambda x: x[1], reverse=True)[:ka]])
        top_b = set([ix for ix, _ in sorted([(ix, float(sb[ix])) for ix in cont_b], key=lambda x: x[1], reverse=True)[:kb]])

        keep_a = [ix for ix in range(len(ta)) if ix not in top_a]
        keep_b = [ix for ix in range(len(tb)) if ix not in top_b]
        with torch.no_grad():
            oa_d = model(input_ids=tok_a['input_ids'][:, keep_a], attention_mask=tok_a['attention_mask'][:, keep_a])[0]
            ob_d = model(input_ids=tok_b['input_ids'][:, keep_b], attention_mask=tok_b['attention_mask'][:, keep_b])[0]
            va_d = F.normalize(mean_pool(oa_d, tok_a['attention_mask'][:, keep_a]), p=2, dim=1)[0]
            vb_d = F.normalize(mean_pool(ob_d, tok_b['attention_mask'][:, keep_b]), p=2, dim=1)[0]

        vecs_mod = vecs_orig.clone()
        vecs_mod[i_star] = va_d
        vecs_mod[j_star] = vb_d
        smat_mod = (vecs_mod @ vecs_mod.T).cpu().numpy()
        doc_max_after_tok_del = float(np.max(smat_mod[triu_idx]))
        doc_mean_after_tok_del = float(np.mean(smat_mod[triu_idx]))
        doc_flip_tok_del = (orig_is_redundant != (doc_max_after_tok_del >= TAU_STRICT))

        doc40_token_records.append({
            'doc_id': doc_id, 'label': label,
            'estrato': 'Redundante (>=0.60)' if orig_is_redundant else 'No Redundante (<0.60)',
            'metodo': m_name,
            'latencia_ms': lats_doc[m_name] * 1000.0,
            'pair_base_sim': b_sim,
            'pair_comp_del': ev_pair['comprehensiveness_del'],
            'pair_comp_mask': ev_pair['comprehensiveness_mask'],
            'pair_comp_noise': ev_pair['comprehensiveness_noise'],
            'pair_sufficiency': ev_pair['sufficiency'],
            'pair_delta_auc': ev_pair['delta_auc'],
            'pair_flip_del': ev_pair['flip_del'],
            'pair_flip_mask': ev_pair['flip_mask'],
            'pair_flip_noise': ev_pair['flip_noise'],
            'doc_s_max_orig': s_max1,
            'doc_s_max_after_tok_del': doc_max_after_tok_del,
            'doc_drop_s_max_tok_del': s_max1 - doc_max_after_tok_del,
            'doc_flip_tok_del': bool(doc_flip_tok_del),
            'doc_delta_mean_tok_del': s_mean_orig - doc_mean_after_tok_del
        })

    if (i + 1) % 10 == 0 or (i + 1) == len(df_sample_docs40):
        print(f"      Progreso Cohorte B: {i+1}/40 documentos evaluados ({time.perf_counter() - t_doc_start:.1f} s)...")

df_doc40_tok = pd.DataFrame(doc40_token_records)
df_doc40_mex = pd.DataFrame(doc40_mexgen_records)

df_doc40_tok.to_csv(os.path.join(DATA_OUT_DIR, "redundancia_benchmark_40_docs_metodos_xai.csv"), index=False)
df_doc40_mex.to_csv(os.path.join(DATA_OUT_DIR, "redundancia_benchmark_40_docs_mexgen_oracional.csv"), index=False)


# =============================================================================
# 6. SANITY CHECK DE ADEBAYO ET AL. (2018) EN REDUNDANCIA (7 MÉTODOS)
# =============================================================================
print("\n[4/7] Ejecutando Sanity Check de Adebayo en cascada sobre SBERT para todos los métodos...")
ref_p = pairs_cohort[0]
_, _, ta_s, tb_s, _, base_attrs_s, _ = explain_pair_all_methods(ref_p['text_a'], ref_p['text_b'], curr_model=model)
methods_red = sorted(base_attrs_s.keys())

stages_adebayo = [
    ("0_Orig (0 Capas)", 12),
    ("1_Top1 (Capa 11)", 11),
    ("2_Top2 (Capas 11-10)", 10),
    ("3_Top4 (Capas 11-8)", 8),
    ("4_Top6 (Capas 11-6)", 6),
    ("5_All (Capas 11-0)", 0)
]

spearman_red_adebayo = {m: [] for m in methods_red}
orig_sbert_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def randomize_layer(layer_mod):
    for m in layer_mod.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

for st_name, min_ly in stages_adebayo:
    model.load_state_dict(orig_sbert_state)
    if min_ly < 12:
        for ly in range(11, min_ly - 1, -1):
            randomize_layer(model.encoder.layer[ly])
    model.eval()
    _, _, _, _, _, alt_attrs_s, _ = explain_pair_all_methods(ref_p['text_a'], ref_p['text_b'], curr_model=model)
    for m in methods_red:
        vo = np.concatenate([base_attrs_s[m][0][1:-1], base_attrs_s[m][1][1:-1]])
        va = np.concatenate([alt_attrs_s[m][0][1:-1], alt_attrs_s[m][1][1:-1]])
        if np.std(vo) < 1e-12 or np.std(va) < 1e-12:
            rho = 0.0
        else:
            rho, _ = spearmanr(vo, va)
            if np.isnan(rho):
                rho = 0.0
        spearman_red_adebayo[m].append(float(rho))

model.load_state_dict(orig_sbert_state)
model.eval()
print("      Sanity Check de Adebayo finalizado y pesos de SBERT restaurados.")


# =============================================================================
# 7. CONSOLIDACIÓN ESTADÍSTICA Y PRUEBAS DE SIGNIFICANCIA
# =============================================================================
print("\n[5/7] Calculando resúmenes cuantitativos y pruebas de hipótesis (Wilcoxon / t-test / Fisher / Mann-Whitney)...")

summary_pairs20 = {}
for m in methods_red:
    sub = df_pairs_20[df_pairs_20['metodo'] == m]
    sub_red = sub[sub['category'] == 'Redundante']
    summary_pairs20[m] = {
        'comp_del_mean': float(sub['comprehensiveness_del'].mean()),
        'comp_del_std': float(sub['comprehensiveness_del'].std(ddof=1)),
        'comp_mask_mean': float(sub['comprehensiveness_mask'].mean()),
        'comp_noise_mean': float(sub['comprehensiveness_noise'].mean()),
        'sufficiency_mean': float(sub['sufficiency'].mean()),
        'sufficiency_std': float(sub['sufficiency'].std(ddof=1)),
        'auc_morf_mean': float(sub['auc_morf'].mean()),
        'auc_lorf_mean': float(sub['auc_lorf'].mean()),
        'delta_auc_mean': float(sub['delta_auc'].mean()),
        'latencia_ms_mean': float(sub['latencia_ms'].mean()),
        'red_10_comp_del_mean': float(sub_red['comprehensiveness_del'].mean()),
        'red_10_flips_del': int(sub_red['flip_del'].sum()),
        'red_10_flips_mask': int(sub_red['flip_mask'].sum()),
        'red_10_flips_noise': int(sub_red['flip_noise'].sum()),
        'avg_morf_curve': [float(x) for x in np.mean(curves_pairs_20[m]['morf'], axis=0)],
        'avg_lorf_curve': [float(x) for x in np.mean(curves_pairs_20[m]['lorf'], axis=0)]
    }

summary_docs40_xai = {}
for m in methods_red:
    sub = df_doc40_tok[df_doc40_tok['metodo'] == m]
    sub_red = sub[sub['estrato'].str.startswith('Redundante')]
    summary_docs40_xai[m] = {
        'pair_comp_del_mean': float(sub['pair_comp_del'].mean()),
        'pair_comp_del_std': float(sub['pair_comp_del'].std(ddof=1)),
        'pair_comp_mask_mean': float(sub['pair_comp_mask'].mean()),
        'pair_suff_mean': float(sub['pair_sufficiency'].mean()),
        'pair_delta_auc_mean': float(sub['pair_delta_auc'].mean()),
        'red20_pair_comp_del_mean': float(sub_red['pair_comp_del'].mean()),
        'red20_pair_flips_del': int(sub_red['pair_flip_del'].sum()),
        'red20_pair_flips_mask': int(sub_red['pair_flip_mask'].sum()),
        'red20_pair_flips_noise': int(sub_red['pair_flip_noise'].sum()),
        'red20_doc_drop_smax_mean': float(sub_red['doc_drop_s_max_tok_del'].mean()),
        'red20_doc_flips_del': int(sub_red['doc_flip_tok_del'].sum()),
        ' red20_doc_flips_del_pct': float(sub_red['doc_flip_tok_del'].mean() * 100.0),
        'doc_delta_mean_tok_del': float(sub_red['doc_delta_mean_tok_del'].mean()),
        'latencia_ms_mean': float(sub['latencia_ms'].mean())
    }

# Estadísticas de MexGen en los mismos 40 documentos y en el Corpus Completo N=1,820
sub_mex_red20 = df_doc40_mex[df_doc40_mex['estrato'].str.startswith('Redundante')]
t_stat_40, p_val_morf_lorf_40 = ttest_rel(sub_mex_red20['delta_mean_morf_hub'], sub_mex_red20['delta_mean_lorf_min'])

df_mex_1820 = pd.read_csv(MEXGEN_FULL_CSV)
df_mex_712 = df_mex_1820[df_mex_1820['s_max1'] >= TAU_STRICT]
t_stat_712, p_val_morf_lorf_712 = ttest_rel(df_mex_712['delta_mean_hub'], df_mex_712['delta_mean_min'])

real_712 = df_mex_712[df_mex_712['label'] == 0]
fake_712 = df_mex_712[df_mex_712['label'] == 1]

# Tabla de contingencia en N=712: Persistencia Multi-Bucle (max_after_star >= 0.5996)
a_f = int((fake_712['max_after_star'] >= TAU_STRICT).sum())
b_f = int((fake_712['max_after_star'] < TAU_STRICT).sum())
c_r = int((real_712['max_after_star'] >= TAU_STRICT).sum())
d_r = int((real_712['max_after_star'] < TAU_STRICT).sum())
or_fisher, p_fisher = fisher_exact([[a_f, b_f], [c_r, d_r]])

summary_mexgen = {
    'muestra_40_docs_redundantes_20': {
        's_max1_mean': float(sub_mex_red20['s_max1'].mean()),
        's_max_after_1_sent_mean': float(sub_mex_red20['s_max_after_1_sent'].mean()),
        'drop_max_1_sent_mean': float(sub_mex_red20['drop_max_1_sent'].mean()),
        'drop_max_2_sents_mean': float(sub_mex_red20['drop_max_2_sents'].mean()),
        'drop_max_rand_unres_mean': float(sub_mex_red20['drop_max_rand_unres'].mean()),
        'flips_1_sent_count': int(sub_mex_red20['flip_1_sent'].sum()),
        'flips_1_sent_pct': float(sub_mex_red20['flip_1_sent'].mean() * 100.0),
        'flips_2_sents_count': int(sub_mex_red20['flip_2_sents'].sum()),
        'flips_2_sents_pct': float(sub_mex_red20['flip_2_sents'].mean() * 100.0),
        'delta_mean_morf_hub': float(sub_mex_red20['delta_mean_morf_hub'].mean()),
        'delta_mean_lorf_min': float(sub_mex_red20['delta_mean_lorf_min'].mean()),
        'p_val_morf_vs_lorf_40': float(p_val_morf_lorf_40),
        'latencia_ms_mean': float(df_doc40_mex['latencia_ms'].mean())
    },
    'corpus_completo_712_redundantes': {
        'n_total_elegibles': int(len(df_mex_1820)),
        'n_redundancia_estricta': int(len(df_mex_712)),
        's_max1_mean': float(df_mex_712['s_max1'].mean()),
        's_max_after_star_mean': float(df_mex_712['max_after_star'].mean()),
        'drop_max_star_mean': float(df_mex_712['delta_max_star'].mean()),
        'drop_max_rand_unres_mean': float(df_mex_712['delta_max_rand_unres'].mean()),
        'delta_mean_hub_morf': float(df_mex_712['delta_mean_hub'].mean()),
        'delta_mean_min_lorf': float(df_mex_712['delta_mean_min'].mean()),
        'p_val_morf_vs_lorf_712': float(p_val_morf_lorf_712),
        'bucle_unico_flips_total_count': int(b_f + d_r),
        'bucle_unico_flips_total_pct': float((b_f + d_r) / len(df_mex_712) * 100.0),
        'real_310_bucle_unico_flips_pct': float(d_r / len(real_712) * 100.0),
        'real_310_multibucle_persist_pct': float(c_r / len(real_712) * 100.0),
        'fake_402_bucle_unico_flips_pct': float(b_f / len(fake_712) * 100.0),
        'fake_402_multibucle_persist_pct': float(a_f / len(fake_712) * 100.0),
        'fisher_odds_ratio_fake_multibucle': float(or_fisher),
        'fisher_p_value': float(p_fisher)
    }
}

json_red_out = {
    'modelo': MODEL_NAME,
    'umbral_redundancia_estricta': TAU_STRICT,
    'cohorte_a_20_pares': summary_pairs20,
    'cohorte_b_40_documentos_xai': summary_docs40_xai,
    'explicabilidad_propia_mexgen': summary_mexgen,
    'sanity_check_adebayo': {
        'stages': [s[0] for s in stages_adebayo],
        'spearman': spearman_red_adebayo
    }
}

json_red_path = os.path.join(DATA_OUT_DIR, "metricas_comparacion_xai_redundancia.json")
with open(json_red_path, 'w', encoding='utf-8') as f:
    json.dump(json_red_out, f, indent=2, ensure_ascii=False)
print(f"      Métricas de Redundancia exportadas a: {json_red_path}")


# =============================================================================
# 8. GENERACIÓN DE GRÁFICAS DE ALTA RESOLUCIÓN PARA REDUNDANCIA (300 DPI)
# =============================================================================
print("\n[6/7] Generando figuras científicas de Redundancia en imagenes/redundancia/...")

short_red = [
    'Fast-IG (10p)', 'Input×Grad', 'Saliency (L2)',
    'Attention (L12)', 'LIME-Light', 'KernelSHAP', 'Propio (MexGen)'
]
colors_red = ['#10b981', '#3b82f6', '#60a5fa', '#94a3b8', '#f59e0b', '#8b5cf6', '#ec4899']
x_r = np.arange(len(methods_red))

# --- FIGURA 1: Fidelidad Causal en Pares (Cohorte A N=20 Pares y Cohorte B N=40 Docs) ---
fig1, axes1 = plt.subplots(1, 2, figsize=(15.5, 6.0))
w = 0.36

# Panel A: 20 Pares (10 Curados + 10 Aleatorios)
ax = axes1[0]
c_p20 = [summary_pairs20[m]['comp_del_mean'] for m in methods_red]
s_p20 = [summary_pairs20[m]['sufficiency_mean'] for m in methods_red]
b1 = ax.bar(x_r - w/2, c_p20, w, label='Comprensividad (Erasure Top 20% ↑ Ideal: Alto)', color='#10b981', edgecolor='#065f46')
b2 = ax.bar(x_r + w/2, s_p20, w, label='Suficiencia (Retención Top 20% ↓ Ideal: 0.0)', color='#f59e0b', edgecolor='#92400e')
ax.set_xticks(x_r)
ax.set_xticklabels(short_red, rotation=22, ha='right', fontweight='bold')
ax.set_ylabel("Cambio en Similitud Coseno ΔSim")
ax.set_title("A. Fidelidad en Suite de 20 Pares (10 Curados + 10 Aleatorios)", fontweight='bold')
ax.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9)
for b in b1:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:+.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065f46')
for b in b2:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#92400e')
ax.set_ylim(0, max(max(c_p20), max(s_p20)) + 0.06)

# Panel B: Pares Maximales dentro de los 40 Documentos del Corpus
ax2 = axes1[1]
c_d40 = [summary_docs40_xai[m]['pair_comp_del_mean'] for m in methods_red]
s_d40 = [summary_docs40_xai[m]['pair_suff_mean'] for m in methods_red]
b3 = ax2.bar(x_r - w/2, c_d40, w, label='Comprensividad en Par Maximal (i*, j*) ↑', color='#2563eb', edgecolor='#1e3a8a')
b4 = ax2.bar(x_r + w/2, s_d40, w, label='Suficiencia en Par Maximal (i*, j*) ↓', color='#ec4899', edgecolor='#831843')
ax2.set_xticks(x_r)
ax2.set_xticklabels(short_red, rotation=22, ha='right', fontweight='bold')
ax2.set_ylabel("Cambio en Similitud Coseno ΔSim")
ax2.set_title("B. Fidelidad en los 40 Documentos Completos del Corpus", fontweight='bold')
ax2.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9)
for b in b3:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:+.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e3a8a')
for b in b4:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#831843')
ax2.set_ylim(0, max(max(c_d40), max(s_d40)) + 0.06)

fig1.suptitle("Fidelidad Causal en Redundancia Semántica (BETO SBERT): 6 XAI de Literatura vs. Propio (MexGen)", fontsize=13.5, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1.savefig(os.path.join(IMG_OUT_DIR, "fig1_fidelidad_redundancia_pares_y_documentos.png"), bbox_inches="tight")
plt.close(fig1)

# --- FIGURA 2: Curvas de Ablación MoRF vs LoRF y Brecha Causal (ΔAUC) para los 7 Métodos ---
fig2, axes2 = plt.subplots(2, 4, figsize=(17, 8.5))
axes2_f = axes2.flatten()
fracs_pct = [0, 20, 40, 60, 80]

for idx_m, m in enumerate(methods_red):
    ax = axes2_f[idx_m]
    mc = summary_pairs20[m]['avg_morf_curve']
    lc = summary_pairs20[m]['avg_lorf_curve']
    dauc = summary_pairs20[m]['delta_auc_mean']
    ax.plot(fracs_pct, lc, marker='s', linewidth=2.2, color='#3b82f6', label=f"LoRF (AUC={summary_pairs20[m]['auc_lorf_mean']:.3f})")
    ax.plot(fracs_pct, mc, marker='o', linewidth=2.2, color='#ef4444', label=f"MoRF (AUC={summary_pairs20[m]['auc_morf_mean']:.3f})")
    ax.fill_between(fracs_pct, mc, lc, color='#8b5cf6', alpha=0.18, label=f"ΔAUC = {dauc:+.3f}")
    ax.set_title(f"{short_red[idx_m]}\n(ΔAUC = {dauc:+.4f})", fontweight='bold', fontsize=11)
    ax.set_xlabel("% Tokens Eliminados")
    ax.set_ylabel("Similitud Coseno")
    ax.set_xticks(fracs_pct)
    ax.set_ylim(0.12, 0.58)
    ax.legend(loc='lower left', fontsize=8.2, frameon=True, facecolor='white')

ax_rk = axes2_f[7]
dauc_p20 = [summary_pairs20[m]['delta_auc_mean'] for m in methods_red]
dauc_d40 = [summary_docs40_xai[m]['pair_delta_auc_mean'] for m in methods_red]
yp = np.arange(len(methods_red))
ax_rk.barh(yp - 0.18, dauc_p20, height=0.35, color='#10b981', label='Suite 20 Pares')
ax_rk.barh(yp + 0.18, dauc_d40, height=0.35, color='#3b82f6', label='40 Documentos')
ax_rk.set_yticks(yp)
ax_rk.set_yticklabels(short_red, fontweight='bold', fontsize=9.5)
ax_rk.set_xlabel("Brecha Causal ΔAUC (LoRF - MoRF ↑)")
ax_rk.set_title("Comparativa Global ΔAUC", fontweight='bold', fontsize=11)
ax_rk.legend(loc='lower right', fontsize=8.5, frameon=True)

fig2.suptitle("Curvas de Ablación Progresiva MoRF vs. LoRF en Similitud Coseno SBERT (7 Métodos Comparados)", fontsize=14, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(IMG_OUT_DIR, "fig2_curvas_morf_lorf_redundancia_7_metodos.png"), bbox_inches="tight")
plt.close(fig2)

# --- FIGURA 3: La Paradoja Multi-Bucle en Documentos Completos (Por qué Token XAI en 1 Par != Desactivar Redundancia del Documento) ---
fig3, axes3 = plt.subplots(1, 2, figsize=(15.5, 6.2))

# Panel A: En los 20 Documentos Redundantes (>=0.60), % de desclasificación en el Par Aislado vs en el Documento Completo
ax = axes3[0]
pair_flip_pct = [summary_docs40_xai[m]['red20_pair_flips_del'] / 20.0 * 100.0 for m in methods_red]
doc_flip_pct = [summary_docs40_xai[m]['red20_doc_flips_del'] / 20.0 * 100.0 for m in methods_red]

# Añadimos las barras de MexGen Oracional Nativo (Ablación de 1 oración s* y de 2 oraciones críticas)
labels_ext = short_red + ['MexGen Oracional\n(Quitar 1 Oración s*)', 'MexGen Oracional\n(Quitar 2 Oraciones)']
pair_ext = pair_flip_pct + [100.0, 100.0]  # Al retirar la oración s*, el par (i*, j*) desaparece al 100%
doc_ext = doc_flip_pct + [
    summary_mexgen['muestra_40_docs_redundantes_20']['flips_1_sent_pct'],
    summary_mexgen['muestra_40_docs_redundantes_20']['flips_2_sents_pct']
]

x_e = np.arange(len(labels_ext))
w_e = 0.36
b_pe = ax.bar(x_e - w_e/2, pair_ext, w_e, label='Desactivación del Par Maximal (i*, j*) Aislado (<0.60)', color='#60a5fa', edgecolor='#1e3a8a')
b_de = ax.bar(x_e + w_e/2, doc_ext, w_e, label='Desactivación de la Redundancia del DOCUMENTO (<0.60)', color='#ec4899', edgecolor='#831843')
ax.set_xticks(x_e)
ax.set_xticklabels(labels_ext, rotation=24, ha='right', fontweight='bold', fontsize=9)
ax.set_ylabel("% de Documentos Redundantes que Cambian de Etiqueta (1 → 0)")
ax.set_title("A. Cambio de Etiqueta de Redundancia (>=0.60 → <0.60) en Muestra N=40", fontweight='bold')
ax.legend(loc='upper left', frameon=True, facecolor='white', fontsize=8.8)
for b in b_pe:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.0f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e3a8a')
for b in b_de:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.0f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#831843')
ax.set_ylim(0, 118)

# Panel B: Hallazgo Forense de MexGen en el Corpus Completo (N=712 con Redundancia Estricta >=0.60):
# Bucle Único (Se desclasifica al retirar s*) vs Red Multi-Bucle Persistente (S_(2) >= 0.60) en Noticias Reales vs Falsas
ax2 = axes3[1]
cats = ['Noticias Reales\n(N = 310)', 'Noticias Falsas\n(N = 402)', 'Total Redundantes\n(N = 712)']
bucle_unico = [
    summary_mexgen['corpus_completo_712_redundantes']['real_310_bucle_unico_flips_pct'],
    summary_mexgen['corpus_completo_712_redundantes']['fake_402_bucle_unico_flips_pct'],
    summary_mexgen['corpus_completo_712_redundantes']['bucle_unico_flips_total_pct']
]
multibucle = [
    summary_mexgen['corpus_completo_712_redundantes']['real_310_multibucle_persist_pct'],
    summary_mexgen['corpus_completo_712_redundantes']['fake_402_multibucle_persist_pct'],
    100.0 - summary_mexgen['corpus_completo_712_redundantes']['bucle_unico_flips_total_pct']
]
xc = np.arange(len(cats))
wc = 0.36
b_bu = ax2.bar(xc - wc/2, bucle_unico, wc, label='Bucle Único Focalizado (Cambia a <0.60 con 1 oración)', color='#10b981', edgecolor='#065f46')
b_mb = ax2.bar(xc + wc/2, multibucle, wc, label='Red Multi-Bucle Persistente (Sigue >=0.60 tras ablación)', color='#ef4444', edgecolor='#991b1b')
ax2.set_xticks(xc)
ax2.set_xticklabels(cats, fontweight='bold', fontsize=10.5)
ax2.set_ylabel("% de Documentos")
ax2.set_title(f"B. Topología MexGen en Corpus (OR={or_fisher:.2f}, p={p_fisher:.2e})", fontweight='bold')
ax2.legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.8)
for b in b_bu:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#065f46')
for b in b_mb:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#991b1b')
ax2.set_ylim(0, 82)

fig3.suptitle("Cambio de Etiqueta de Redundancia y Descubrimiento Topológico Multi-Bucle (MexGen vs. Token XAI)", fontsize=13.5, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(IMG_OUT_DIR, "fig3_cambio_estado_redundancia_y_multibucle_mexgen.png"), bbox_inches="tight")
plt.close(fig3)

# --- FIGURA 4: Ablación No Tautológica sobre la Similitud Media Global (ΔS_bar_doc) y Sanity Check de Adebayo ---
fig4, axes4 = plt.subplots(1, 2, figsize=(15.5, 5.8))

# Panel A: Prueba No Tautológica sobre Similitud Media del Documento (MoRF Hub vs Azar Irrestricto 1/K vs LoRF Min)
ax = axes4[0]
conds = ['MoRF Oracional\n(Retirar Hub s_hub)', 'Control Aleatorio\nIrrestricto (1/K)', 'LoRF Oracional\n(Retirar Periférica s_min)', 'Token XAI (Fast-IG)\n(Borrar 20% par maximal)']
vals_40 = [
    summary_mexgen['muestra_40_docs_redundantes_20']['delta_mean_morf_hub'],
    0.0,
    summary_mexgen['muestra_40_docs_redundantes_20']['delta_mean_lorf_min'],
    summary_docs40_xai['1. Fast-IG (10 pasos)']['doc_delta_mean_tok_del']
]
vals_712 = [
    summary_mexgen['corpus_completo_712_redundantes']['delta_mean_hub_morf'],
    0.0,
    summary_mexgen['corpus_completo_712_redundantes']['delta_mean_min_lorf'],
    summary_docs40_xai['1. Fast-IG (10 pasos)']['doc_delta_mean_tok_del']
]
x_c = np.arange(len(conds))
b_c40 = ax.bar(x_c - 0.18, vals_40, width=0.34, label=f'Muestra Unificada (p={p_val_morf_lorf_40:.2e})', color='#2563eb', edgecolor='#1e3a8a')
b_c712 = ax.bar(x_c + 0.18, vals_712, width=0.34, label=f'Corpus Redundante N=712 (p={p_val_morf_lorf_712:.2e})', color='#ec4899', edgecolor='#831843')
ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_xticks(x_c)
ax.set_xticklabels(conds, fontweight='bold', fontsize=9)
ax.set_ylabel("Cambio en Similitud Media del Documento ΔS̄_doc")
ax.set_title("A. Prueba Causal No Tautológica sobre Similitud Media (ΔS̄_doc)", fontweight='bold')
ax.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9)
for b in b_c40:
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    off = 0.002 if h >= 0 else -0.005
    ax.text(b.get_x() + b.get_width()/2, h + off, f"{h:+.4f}", ha='center', va=va, fontsize=8, fontweight='bold', color='#1e3a8a')
ax.set_ylim(-0.075, 0.062)

# Panel B: Sanity Check de Adebayo en SBERT para los 7 métodos
ax2 = axes4[1]
st_labels = [s[0] for s in stages_adebayo]
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
for idx_m, m in enumerate(methods_red):
    ax2.plot(st_labels, spearman_red_adebayo[m], marker=markers[idx_m], markersize=7, linewidth=2.2, color=colors_red[idx_m], label=short_red[idx_m])
ax2.axhline(0, color='#475569', linestyle=':', linewidth=1.2)
ax2.set_xticklabels(st_labels, rotation=20, ha='right', fontweight='bold', fontsize=9.5)
ax2.set_ylabel("Correlación de Rangos de Spearman (ρ)")
ax2.set_title("B. Control de Sanidad de Adebayo en Cascada (SBERT)", fontweight='bold')
ax2.legend(loc='lower left', fontsize=8.2, frameon=True, facecolor='white', ncol=2)
ax2.set_ylim(-0.65, 1.08)

fig4.suptitle("Validación Metrológica No Tautológica (MexGen) y Sensibilidad a Parámetros (Adebayo et al., 2018)", fontsize=13.5, fontweight='bold', y=1.02)
fig4.tight_layout()
fig4.savefig(os.path.join(IMG_OUT_DIR, "fig4_ablacion_no_tautologica_y_adebayo_sbert.png"), bbox_inches="tight")
plt.close(fig4)

# --- FIGURA 5: Perfil de Latencia Computacional en GPU (Pares y Documentos) ---
fig5, ax5 = plt.subplots(figsize=(10.5, 5.2))
lats_p = [summary_pairs20[m]['latencia_ms_mean'] for m in methods_red]
bars_l = ax5.bar(x_r, lats_p, color=colors_red, edgecolor='#1e293b', width=0.55)
ax5.set_xticks(x_r)
ax5.set_xticklabels(short_red, fontweight='bold', fontsize=10)
ax5.set_yscale('log')
ax5.set_ylabel("Latencia Promedio por Par (ms - Escala Logarítmica)")
ax5.set_title("Eficiencia Computacional en GPU (NVIDIA GTX 1650): 6 Métodos XAI de Literatura vs. Propio (MexGen)", fontweight='bold', pad=12)
for b, v in zip(bars_l, lats_p):
    ax5.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v:.1f} ms", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax5.set_ylim(0.5, max(lats_p) * 3.0)
fig5.tight_layout()
fig5.savefig(os.path.join(IMG_OUT_DIR, "fig5_latencia_computacional_redundancia.png"), bbox_inches="tight")
plt.close(fig5)

print("\n[7/7] ¡Benchmark de Redundancia finalizado con éxito!")
