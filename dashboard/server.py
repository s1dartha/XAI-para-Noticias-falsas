import os
import sys
import re
import json
import torch
import numpy as np
import spacy
import nltk
import joblib
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__, static_folder=".")
CORS(app)

print("🚀 Inicializando Backend Avanzado para Dashboard Tesis (5D + SaBERT + XAI)...")

# Configuración NLTK y SpaCy
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    nlp = spacy.load("es_core_news_sm", disable=['ner'])
except Exception:
    os.system(f'"{sys.executable}" -m spacy download es_core_news_sm')
    nlp = spacy.load("es_core_news_sm", disable=['ner'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Dispositivo PyTorch: {device}")

# 1. Cargar SaBERT
print("⏳ Cargando SaBERT Spanish Fake News...")
try:
    sabert_tokenizer = BertTokenizer.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News")
    sabert_model = BertForSequenceClassification.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News").to(device)
    sabert_model.eval()
    sabert_loaded = True
except Exception as e:
    print(f"⚠️ Error cargando SaBERT: {e}")
    sabert_loaded = False

# 2. Cargar Sensacionalismo
print("⏳ Cargando Bert Spanish Sensationalism...")
try:
    sensationalism_classifier = pipeline(
        "text-classification",
        model="JJNeila/bert-spanish-sensationalism-oss",
        tokenizer="JJNeila/bert-spanish-sensationalism-oss",
        device=0 if torch.cuda.is_available() else -1,
        top_k=None
    )
    sens_loaded = True
except Exception as e:
    print(f"⚠️ Error cargando modelo de sensacionalismo: {e}")
    sens_loaded = False

# 3. Cargar SBERT
print("⏳ Cargando SBERT (paraphrase-multilingual-MiniLM-L12-v2)...")
try:
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=str(device))
    sbert_loaded = True
except Exception as e:
    try:
        sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=str(device))
        sbert_loaded = True
    except Exception as e2:
        print(f"⚠️ Error cargando SBERT: {e2}")
        sbert_loaded = False

# 4. Cargar Modelo Final GBDT 5D
model_path = os.path.join(os.path.dirname(__file__), "modelo_ensamble_gbdt_5d.joblib")
gbdt_artifact = None
if os.path.exists(model_path):
    try:
        gbdt_artifact = joblib.load(model_path)
        print("✅ Modelo GBDT 5D cargado exitosamente.")
    except Exception as e:
        print(f"⚠️ Error cargando modelo GBDT: {e}")

print("✅ Servidor Backend inicializado y listo.")

# Expresiones regulares lingüísticas
PAT_HEDGES = re.compile(
    r'\b(presunto|presunta|presuntos|presuntas|presuntamente|al parecer|según fuentes|'
    r'podría|podrían|se presume|supuesto|supuesta|supuestos|supuestamente|aparentemente|'
    r'probablemente|posiblemente|se sospecha|se infiere|de confirmarse|no se descarta|'
    r'trascendió que|habría sido|habrían sido)\b',
    re.IGNORECASE
)

PAT_BOOSTERS = re.compile(
    r'\b(indudablemente|obviamente|claramente|sin duda|sin lugar a dudas|es un hecho|'
    r'irrefutable|totalmente|absolutamente|jamás|nunca jamás|la verdad oculta|revelado|'
    r'comprobado|definitivamente|rotundamente|innegable|a ciencia cierta|categóricamente|'
    r'escándalo total|evidencia irrefutable|científicamente comprobado|boooomm|urgente|alerta)\b',
    re.IGNORECASE
)

PAT_DICENDI = re.compile(
    r'\b(dijo|declaró|manifestó|indicó|explicó|informó|expresó|afirmó|sostuvo|aseveró|'
    r'aseguró|añadió|puntualizó|precisó|concluyó|señaló|comentó|declararon|informaron|anunció)\b',
    re.IGNORECASE
)

PAT_TEMPORAL = re.compile(
    r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|'
    r'2018|2019|2020|2021|2022|2023|2024|2025|2026)\b',
    re.IGNORECASE
)

PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}

def count_syllables_es(word):
    word = word.lower()
    vowels = "aeiouáéíóúü"
    count = 0
    in_vowel = False
    for char in word:
        if char in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(1, count)

def segmentar_oraciones(texto):
    try:
        sents = [s.strip() for s in nltk.sent_tokenize(str(texto), language='spanish') if len(s.strip()) > 8]
        if sents:
            return sents
    except Exception:
        pass
    raw = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', str(texto)) if len(s.strip()) > 8]
    return raw if raw else [texto.strip()]

def evaluar_sabert(texto):
    if not sabert_loaded:
        return "VERDADERO", 0.35, 0.65
    inputs = sabert_tokenizer(str(texto), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = sabert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    if isinstance(probs, float):
        probs = [probs, 1.0 - probs]
    prob_fake = float(probs[0])
    prob_true = float(probs[1])
    etiqueta = "FALSO" if prob_fake >= prob_true else "VERDADERO"
    return etiqueta, prob_fake, prob_true

def evaluar_sensacionalismo(texto):
    if not sens_loaded:
        # Heurística fallback
        return "NO Sensacional", 0.25
    try:
        res = sensationalism_classifier(str(texto)[:512])[0]
        score_sens = next(item["score"] for item in res if item["label"] in ["LABEL_1", "Sensacionalista"])
        etiqueta = "Sensacionalista" if score_sens >= 0.50 else "NO Sensacional"
        return etiqueta, float(score_sens)
    except Exception:
        return "NO Sensacional", 0.25

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    path = os.path.join(os.path.dirname(__file__), "metricas_ensamble_avanzado_5d.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Métricas no encontradas"}), 404

@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    path = os.path.join(os.path.dirname(__file__), "diccionario_variables_5d.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Diccionario no encontrado"}), 404

@app.route('/api/benchmark', methods=['GET'])
def get_benchmark():
    path = os.path.join(os.path.dirname(__file__), "benchmark_news.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Benchmark no encontrado"}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_news():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No se proporcionó texto de noticia"}), 400

    # 1. Segmentación oracional
    oraciones = segmentar_oraciones(text)
    num_sentences = max(len(oraciones), 1)

    # 2. Evaluación SaBERT Noticia Completa
    lbl_sabert, prob_fake_sabert, prob_true_sabert = evaluar_sabert(text)

    # 3. Evaluación Sensacionalismo Global y Oracional (D0)
    lbl_sens_gen, p_full = evaluar_sensacionalismo(text)
    
    sens_oraciones = []
    for s in oraciones:
        _, p_s = evaluar_sensacionalismo(s)
        sens_oraciones.append(p_s)

    p_mean = float(np.mean(sens_oraciones)) if sens_oraciones else p_full
    p_max = float(np.max(sens_oraciones)) if sens_oraciones else p_full
    p_sorted = sorted(sens_oraciones, reverse=True)
    p_top2 = float(np.mean(p_sorted[:2])) if len(p_sorted) >= 2 else p_max
    sigma_sens = float(np.std(sens_oraciones)) if len(sens_oraciones) > 1 else 0.0
    dilution_ratio = float(p_full / (p_max + 1e-6))

    # Ablación contrafáctica: silenciar las 2 frases con mayor sensacionalismo
    if len(oraciones) >= 3:
        idx_top = set(np.argsort(sens_oraciones)[-2:])
        texto_ablacionado = " ".join([s for i, s in enumerate(oraciones) if i not in idx_top])
        _, p_abl = evaluar_sensacionalismo(texto_ablacionado)
        delta_p_gatillo = float(max(0.0, p_full - p_abl))
    else:
        delta_p_gatillo = 0.0

    # 4. Redundancia y Coherencia Consecutiva con SBERT (D0 y D1)
    consec_sims = []
    matriz_sim = None
    if sbert_loaded and len(oraciones) >= 2:
        try:
            embs = sbert_model.encode(oraciones, convert_to_tensor=True, normalize_embeddings=True)
            # Consecutivas
            e_cur = embs[:-1]
            e_next = embs[1:]
            consec_sims = torch.sum(e_cur * e_next, dim=1).cpu().numpy().tolist()
            # Matriz completa
            mat_sim = util.cos_sim(embs, embs).cpu().numpy()
            n = len(oraciones)
            tri_indices = np.triu_indices(n, k=1)
            pares_sim = mat_sim[tri_indices]
            max_intra = float(np.max(pares_sim)) if len(pares_sim) > 0 else 0.0
            mean_intra = float(np.mean(pares_sim)) if len(pares_sim) > 0 else 0.0
            redundant_count = int(np.sum(pares_sim >= 0.34))
            redundancy_density = float(redundant_count / max(1, len(pares_sim)))
        except Exception:
            consec_sims = [0.45] * (len(oraciones) - 1)
            max_intra, mean_intra, redundancy_density = 0.40, 0.25, 0.10
    else:
        consec_sims = [0.45] * max(1, len(oraciones) - 1)
        max_intra, mean_intra, redundancy_density = 0.20, 0.15, 0.0

    consec_sim_mean = float(np.mean(consec_sims)) if consec_sims else 0.45
    consec_sim_min = float(np.min(consec_sims)) if consec_sims else 0.45
    consec_sim_std = float(np.std(consec_sims)) if consec_sims else 0.0

    # 5. Dimensión 2: Forense y Epistémica
    tokens_raw = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+\b', text)
    w_count = max(len(tokens_raw), 1)

    n_hedges = len(PAT_HEDGES.findall(text))
    n_boosters = len(PAT_BOOSTERS.findall(text))
    n_dicendi = len(PAT_DICENDI.findall(text))
    n_quotes = len(re.findall(r'["«»“”]', text))

    hedges_density = (n_hedges / w_count) * 100.0
    boosters_density = (n_boosters / w_count) * 100.0
    epistemic_ratio = float(n_boosters / (n_hedges + 0.1))
    quotes_density = float(n_quotes / num_sentences)
    dicendi_density = (n_dicendi / w_count) * 100.0

    # 6. Dimensión 3: Riqueza Léxica y Legibilidad
    tokens_clean = [t.lower() for t in tokens_raw if t.isalpha()]
    w_clean_count = max(len(tokens_clean), 1)
    vocab = set(tokens_clean)
    guiraud_ttr = float(len(vocab) / np.sqrt(w_clean_count))

    counts = Counter(tokens_clean)
    n_hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = float(n_hapax / w_clean_count)

    total_syllables = sum(count_syllables_es(t) for t in tokens_clean)
    total_chars = sum(len(t) for t in tokens_clean)
    flesch_szigriszt = float(np.clip(206.835 - 62.3 * (total_syllables / w_clean_count) - (w_clean_count / num_sentences), 0, 120))
    gutierrez_polini = float(np.clip(95.2 - 9.7 * (total_chars / w_clean_count) - (w_clean_count / num_sentences), 0, 100))

    # 7. Dimensión 4: Morfosintaxis spaCy
    doc = nlp(text)
    n_adj = 0
    n_noun = 0
    n_adv = 0
    n_p1 = 0
    n_p3 = 0

    for token in doc:
        pos = token.pos_
        t_low = token.text.lower()
        if pos == 'ADJ': n_adj += 1
        elif pos == 'NOUN': n_noun += 1
        elif pos == 'ADV': n_adv += 1
        
        if pos == 'PRON' or t_low in PRON_1P or t_low in PRON_3P:
            if t_low in PRON_1P: n_p1 += 1
            elif t_low in PRON_3P: n_p3 += 1

    adj_noun_ratio = float(n_adj / (n_noun + 1))
    pron_1p_density = (n_p1 / w_count) * 100.0
    pron_3p_density = (n_p3 / w_count) * 100.0
    adv_density = (n_adv / w_count) * 100.0

    # 8. Dimensión 5: Anclajes Factuales, Mayúsculas y Puntuación
    all_caps_words = [w for w in tokens_raw if w.isupper() and len(w) >= 3 and w.isalpha()]
    all_caps_count = len(all_caps_words)
    all_caps_ratio = (all_caps_count / w_count) * 100.0

    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    upper_chars_ratio = (len(upper_chars) / max(1, len(alpha_chars)))

    excl_count = text.count('!') + text.count('¡')
    quest_count = text.count('?') + text.count('¿')
    ellipsis_count = text.count('...') + text.count('…')

    excl_density = (excl_count / w_count) * 100.0
    quest_density = (quest_count / w_count) * 100.0
    ellipsis_density = (ellipsis_count / w_count) * 100.0
    punct_intensity = ((excl_count + quest_count + ellipsis_count) / w_count) * 100.0

    numbers_count = len(re.findall(r'\b\d+(?:[\.,]\d+)?\b', text))
    numbers_density = (numbers_count / w_count) * 100.0

    percent_count = text.count('%') + len(re.findall(r'\bpor ciento\b', text, re.IGNORECASE))
    temporal_count = len(PAT_TEMPORAL.findall(text))
    temporal_density = (temporal_count / w_count) * 100.0

    # 9. Inferencia con Modelo GBDT 5D
    features_dict = {
        'P_full': p_full,
        'P_mean': p_mean,
        'P_max': p_max,
        'P_top2': p_top2,
        'sigma_sens': sigma_sens,
        'dilution_ratio': dilution_ratio,
        'delta_p_gatillo': delta_p_gatillo,
        'max_intra_similarity_clean': max_intra,
        'mean_intra_similarity_clean': mean_intra,
        'redundancy_density': redundancy_density,
        'consec_sim_mean': consec_sim_mean,
        'consec_sim_min': consec_sim_min,
        'consec_sim_std': consec_sim_std,
        'hedges_density': hedges_density,
        'boosters_density': boosters_density,
        'epistemic_ratio': epistemic_ratio,
        'quotes_density': quotes_density,
        'dicendi_density': dicendi_density,
        'flesch_szigriszt': flesch_szigriszt,
        'gutierrez_polini': gutierrez_polini,
        'guiraud_ttr': guiraud_ttr,
        'hapax_ratio': hapax_ratio,
        'conteo_palabras_text': float(w_count),
        'num_sentences': float(num_sentences),
        'adj_noun_ratio': adj_noun_ratio,
        'pron_1p_density': pron_1p_density,
        'pron_3p_density': pron_3p_density,
        'adv_density': adv_density,
        'all_caps_count': float(all_caps_count),
        'all_caps_ratio': all_caps_ratio,
        'upper_chars_ratio': upper_chars_ratio,
        'excl_density': excl_density,
        'quest_density': quest_density,
        'ellipsis_density': ellipsis_density,
        'punct_intensity': punct_intensity,
        'numbers_density': numbers_density,
        'percent_count': float(percent_count),
        'temporal_density': temporal_density
    }

    prob_ensamble_5d = 0.50
    if gbdt_artifact is not None:
        try:
            feat_names = gbdt_artifact['features']
            ohe_cols = gbdt_artifact['ohe_cols']
            x_vec = [features_dict.get(f, 0.0) for f in feat_names]
            # Shannon bin dummy approximation (bin4_Estándar = 1.0)
            for col in ohe_cols:
                x_vec.append(1.0 if 'Estándar' in col else 0.0)
            x_arr = np.array(x_vec).reshape(1, -1)
            prob_ensamble_5d = float(gbdt_artifact['model'].predict_proba(x_arr)[0, 1])
        except Exception as e:
            print(f"⚠️ Error en inferencia GBDT: {e}")
            prob_ensamble_5d = 0.55 if (all_caps_ratio > 4.0 or p_max > 0.80 or adv_density > 6.0) else 0.35
    else:
        # Heurística ponderada
        prob_ensamble_5d = float(np.clip(
            0.30 * p_full + 0.25 * (all_caps_ratio / 15.0) + 0.20 * (adv_density / 10.0) + 0.15 * p_max - 0.10 * (dicendi_density / 2.0),
            0.05, 0.95
        ))

    # 10. Desglose Frase a Frase para Explicabilidad Local
    sentences_analyzed = []
    prev_s_words = None
    for i, s_text in enumerate(oraciones):
        s_words = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', s_text)
        s_w_count = max(len(s_words), 1)
        s_caps_words = [w for w in s_words if w.isupper() and len(w) >= 3]
        s_caps_ratio = (len(s_caps_words) / s_w_count) * 100.0

        s_adv_count = sum(1 for w in s_words if w.lower().endswith("mente") or w.lower() in ["muy", "más", "tan", "bastante", "casi", "apenas", "siempre", "nunca", "jamás", "ya"])
        s_adv_density = (s_adv_count / s_w_count) * 100.0

        s_dicendi = len(PAT_DICENDI.findall(s_text))
        s_boosters = len(PAT_BOOSTERS.findall(s_text))
        s_has_quotes = bool(re.search(r'["«»“”]', s_text))
        s_excl = s_text.count('!') + s_text.count('¡')

        c_sim = float(consec_sims[i - 1]) if i > 0 and i - 1 < len(consec_sims) else 0.50
        s_sens = float(sens_oraciones[i]) if i < len(sens_oraciones) else 0.20

        is_trigger = bool(s_sens >= 0.70 or s_caps_ratio >= 15.0 or (s_boosters > 0 and s_adv_density > 7.0))

        flags = []
        if s_caps_ratio >= 12.0: flags.append("MAYÚSCULAS SOSTENIDAS")
        if s_boosters > 0: flags.append("Intensificador / Booster")
        if s_adv_density >= 7.0: flags.append(f"Alta Densidad Adverbial ({s_adv_density:.1f}%)")
        if s_excl > 0: flags.append("Énfasis Puntuación (!)")
        if s_dicendi > 0: flags.append("Verbo Atribución / Dicendi")
        if s_has_quotes: flags.append("Cita Entrecomillada")
        if c_sim < 0.15 and i > 0: flags.append("Salto Temático Incoherente")

        sentences_analyzed.append({
            "idx": i + 1,
            "text": s_text,
            "sens_score": round(s_sens, 3),
            "sens_pct": round(s_sens * 100, 1),
            "consec_sim": round(c_sim, 3),
            "is_trigger": is_trigger,
            "all_caps_ratio": round(s_caps_ratio, 1),
            "adv_density": round(s_adv_density, 1),
            "dicendi_count": s_dicendi,
            "has_quotes": s_has_quotes,
            "flags": flags
        })

    response = {
        "text": text,
        "word_count": w_count,
        "num_sentences": num_sentences,
        "sabert": {
            "label": lbl_sabert,
            "prob_fake": round(prob_fake_sabert, 4),
            "prob_true": round(prob_true_sabert, 4),
            "prob_fake_pct": round(prob_fake_sabert * 100, 1),
            "opacity_note": "Modelo de la Literatura: Salida binaria escalar opaca sin desglose de oraciones ni sustento explicativo."
        },
        "ensamble_5d": {
            "label_th50": "FALSO" if prob_ensamble_5d >= 0.50 else "VERDADERO",
            "label_th42": "FALSO" if prob_ensamble_5d >= 0.42 else "VERDADERO",
            "prob_fake": round(prob_ensamble_5d, 4),
            "prob_fake_pct": round(prob_ensamble_5d * 100, 1),
            "calibrated_threshold": 0.42,
            "ablation_delta_p": round(delta_p_gatillo, 3)
        },
        "features": {k: round(v, 4) for k, v in features_dict.items()},
        "sentences": sentences_analyzed
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Servidor Dashboard escuchando en http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
