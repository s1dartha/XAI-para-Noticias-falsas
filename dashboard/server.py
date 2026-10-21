import os
import sys
import re
import json
import time
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Explicabilidad_Propia", "codigos")))
try:
    from arquetipos_flexibles import clasificar_arquetipo
except Exception:
    def clasificar_arquetipo(d1, d2, d3, d4, d5):
        a1 = 0.45 * max(0.0, min(1.0, (d1 - 40) / 35)) + 0.30 * (1.0 - max(0.0, min(1.0, (d3 - 10) / 50))) + 0.25 * max(0.0, min(1.0, (d4 - 35) / 40))
        a2 = 0.45 * max(0.0, min(1.0, (d2 - 35) / 30)) + 0.40 * max(0.0, min(1.0, (d3 - 40) / 35)) + 0.15 * (1.0 - max(0.0, min(1.0, (d1 - 30) / 40)))
        a3 = 0.45 * max(0.0, min(1.0, (d4 - 45) / 35)) + 0.35 * (1.0 - max(0.0, min(1.0, (d1 - 20) / 40))) + 0.20 * (1.0 - max(0.0, min(1.0, (d2 - 25) / 40)))
        a4 = 0.50 * (1.0 - max(0.0, min(1.0, (d5 - 38) / 32))) + 0.35 * (1.0 - max(0.0, min(1.0, (d4 - 25) / 30))) + 0.15 * (1.0 - max(0.0, min(1.0, (d3 - 20) / 40)))
        a5 = 0.35 * max(0.0, min(1.0, (d5 - 45) / 30)) + 0.30 * (1.0 - max(0.0, min(1.0, (d1 - 20) / 35))) + 0.20 * (1.0 - abs(d4 - 45) / 30) + 0.15 * (1.0 - max(0.0, min(1.0, (d2 - 25) / 30)))
        afs = [a1, a2, a3, a4, a5]
        idx = int(np.argmax(afs))
        names = [
            ("ESTILO_I_DESINFO_ESTRIDENTE", "Estilo I: Desinformación Estridente / Cliché Hiperbólico", "Alarma emocional sostenida en todo el texto, sin amortiguación del cuerpo."),
            ("ESTILO_II_CLICKBAIT_CABECERA", "Estilo II: Cebo Comercial / Clickbait de Cabecera", "Titular alarmista pero con cuerpo informativo formal que amortigua."),
            ("ESTILO_III_PROPAGANDA_SOBRIA", "Estilo III: Propaganda Institucional / Astroturfing Encubierto", "Tono sobrio pero con bucles de repetición semántica calculados."),
            ("ESTILO_IV_DESARTICULADO", "Estilo IV: Incoherencia Estructural / Generación Rota", "Texto con baja conectividad proposicional o fragmentación."),
            ("ESTILO_V_PERIODISMO_EQUILIBRADO", "Estilo V: Periodismo Profesional Balanceado", "Redacción equilibrada con variedad léxica y progresión en Banda de Shannon.")
        ]
        return {"id": names[idx][0], "nombre": names[idx][1], "descripcion": names[idx][2], "afinidad_pct": round(afs[idx]*100, 1), "arquetipo_secundario": "", "afinidad_secundaria_pct": 0.0, "afinidades_pct": {}, "condicion_activada": f"Afinidad dominante = {afs[idx]*100:.1f}%"}

app = Flask(__name__, static_folder=".")
CORS(app)

print("🚀 Inicializando Backend para Pipeline Tesis (Auditoría Forense IML + 36 Variables D0-D5 + Arquetipos + Ensamble 5D + SaBERT)...")

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

# 2. Cargar Sensacionalismo (BETO)
print("⏳ Cargando BETO Spanish Sensationalism...")
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

# 3. Cargar SBERT Clásico sobre BETO (hiiamsid/sentence_similarity_spanish_es, d=768)
print("⏳ Cargando BETO SBERT (hiiamsid/sentence_similarity_spanish_es, d=768)...")
try:
    sbert_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es', device=str(device))
    sbert_loaded = True
except Exception as e:
    print(f"⚠️ Error cargando BETO SBERT: {e}")
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

# Expresiones regulares forenses y lingüísticas compiladas para español
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
    r'escándalo total|evidencia irrefutable|científicamente comprobado|boooomm|urgente|alerta|'
    r'impactante|brutal|increíble|asombroso|jamás visto)\b',
    re.IGNORECASE
)

PAT_DICENDI = re.compile(
    r'\b(dijo|declaró|manifestó|indicó|explicó|informó|expresó|afirmó|sostuvo|aseveró|'
    r'aseguró|añadió|puntualizó|precisó|concluyó|señaló|comentó|declararon|informaron|anunció|'
    r'reiteró|enfatizó|subrayó|destacó|detalló)\b',
    re.IGNORECASE
)

PAT_TEMPORAL = re.compile(
    r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|'
    r'2018|2019|2020|2021|2022|2023|2024|2025|2026)\b',
    re.IGNORECASE
)

PAT_QUOTES = re.compile(r'["«»“”]')
PAT_DIGITS = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
PAT_CENSORED_MASKS = re.compile(r'\*(?:NUMBER|URL|PHONE|\d+)\*', re.IGNORECASE)
PAT_CENSORED_NUMBER = re.compile(r'\*(?:NUMBER|\d+)\*', re.IGNORECASE)
PAT_PERCENT = re.compile(r'(?:\d+\s*%|\bpor\s+ciento\b)', re.IGNORECASE)
PAT_ALL_CAPS = re.compile(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
SIGLAS_EXCLUIDAS = {'ONU', 'UE', 'EEUU', 'OMS', 'PIB', 'FMI', 'OTAN', 'VOX', 'PSOE', 'PP', 'INE', 'USA', 'EFE', 'IAC', 'NICA', 'NUMBER', 'URL', 'PHONE'}

PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}
ADVERBIOS_ENFATICOS = {'muy', 'más', 'tan', 'bastante', 'casi', 'apenas', 'siempre', 'nunca', 'jamás', 'ya', 'totalmente', 'absolutamente', 'demasiado'}

TAU_CONTINUIDAD_GMM = 0.3434      # Continuidad temática basal (GMM)
TAU_PARAFRASIS_PAWSX = 0.5340     # Paráfrasis moderada (P90 / PAWS-X)
TAU_REDUNDANCIA_ESTRICTA = 0.5996 # Redundancia semántica estricta (P95 / Shannon = 0.6079)
TAU_HIPER_REDUNDANCIA = 0.7287    # Hiper-redundancia circular (P99)
TAU_REDUNDANCIA = TAU_PARAFRASIS_PAWSX

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
    raw_text = str(texto).strip()
    try:
        sents = [s.strip() for s in nltk.sent_tokenize(raw_text, language='spanish') if len(s.strip()) > 8]
        if sents:
            return sents
    except Exception:
        pass
    raw = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', raw_text) if len(s.strip()) > 8]
    return raw if raw else [raw_text[:200]]

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

@app.route('/api/forense_examples', methods=['GET'])
def get_forense_examples():
    path = os.path.join(os.path.dirname(__file__), "auditoria_completa_4_noticias.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Ejemplos forenses no encontrados"}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_news():
    t_start = time.time()
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    titulo = data.get("title", "").strip()
    if not text:
        return jsonify({"error": "No se proporcionó texto de noticia"}), 400

    # 1. Segmentación oracional
    oraciones = segmentar_oraciones(text)
    num_sentences = max(len(oraciones), 1)
    k = len(oraciones)

    # 2. Evaluación SaBERT Noticia Completa y por oración
    lbl_sabert, prob_fake_sabert, prob_true_sabert = evaluar_sabert(text)
    sabert_oraciones = [evaluar_sabert(s) for s in oraciones]

    # 3. Evaluación Sensacionalismo Global y Oracional (BETO - D0)
    lbl_sens_gen, p_full = evaluar_sensacionalismo(text)
    sens_oraciones = [evaluar_sensacionalismo(s)[1] for s in oraciones]

    p_mean = float(np.mean(sens_oraciones)) if sens_oraciones else p_full
    p_max = float(np.max(sens_oraciones)) if sens_oraciones else p_full
    idx_max = int(np.argmax(sens_oraciones)) if sens_oraciones else 0
    oracion_gatillo = oraciones[idx_max] if oraciones else text

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
        delta_p_gatillo = float(max(0.0, p_full - p_mean))

    # 4. Redundancia y Coherencia Consecutiva con BETO SBERT (D0 y D1)
    consec_sims = []
    sim_matrix = np.eye(k)
    n_pairs = 0
    redundant_count = 0
    second_max_intra = 0.0
    top3_mean_intra = 0.0
    p90_intra = 0.0
    var_intra = 0.0
    mexgen_gap = 0.0
    delta_mean_morf = 0.0
    topic_cont_density = 0.0
    paraphrase_density = 0.0
    strict_red_density = 0.0
    hyper_red_density = 0.0
    if sbert_loaded and k >= 2:
        try:
            embs = sbert_model.encode(oraciones, convert_to_tensor=True, normalize_embeddings=True)
            sim_matrix = util.cos_sim(embs, embs).cpu().numpy()
            for i in range(k - 1):
                consec_sims.append(float(sim_matrix[i, i + 1]))
            tri_indices = np.triu_indices(k, k=1)
            pares_sim = sim_matrix[tri_indices]
            n_pairs = len(pares_sim)
            pares_sorted = np.sort(pares_sim)[::-1]
            max_intra = float(pares_sorted[0]) if n_pairs > 0 else 0.0
            second_max_intra = float(pares_sorted[1]) if n_pairs >= 2 else max_intra
            top3_mean_intra = float(np.mean(pares_sorted[:3])) if n_pairs > 0 else 0.0
            p90_intra = float(np.percentile(pares_sim, 90)) if n_pairs > 0 else 0.0
            var_intra = float(np.var(pares_sim)) if n_pairs > 0 else 0.0
            mean_intra = float(np.mean(pares_sim)) if n_pairs > 0 else 0.0
            mexgen_gap = float(max(0.0, max_intra - second_max_intra))
            if k >= 3:
                row_means = [(np.sum(sim_matrix[r]) - 1.0) / (k - 1) for r in range(k)]
                hub_idx = int(np.argmax(row_means))
                rem_sims = [sim_matrix[r1, r2] for r1 in range(k) for r2 in range(r1 + 1, k) if r1 != hub_idx and r2 != hub_idx]
                delta_mean_morf = float(mean_intra - np.mean(rem_sims)) if rem_sims else 0.0
            topic_cont_density = float(np.sum(pares_sim >= TAU_CONTINUIDAD_GMM) / max(1, n_pairs))
            paraphrase_density = float(np.sum(pares_sim >= TAU_PARAFRASIS_PAWSX) / max(1, n_pairs))
            strict_red_density = float(np.sum(pares_sim >= TAU_REDUNDANCIA_ESTRICTA) / max(1, n_pairs))
            hyper_red_density = float(np.sum(pares_sim >= TAU_HIPER_REDUNDANCIA) / max(1, n_pairs))
            redundant_count = int(np.sum(pares_sim >= TAU_CONTINUIDAD_GMM))
            redundancy_density = strict_red_density
        except Exception:
            consec_sims = [0.35] * (k - 1)
            max_intra, mean_intra, redundancy_density = 0.45, 0.25, 0.05
    else:
        consec_sims = [0.35] * max(0, k - 1)
        max_intra, mean_intra, redundancy_density = 0.20, 0.15, 0.0

    consec_sim_mean = float(np.mean(consec_sims)) if consec_sims else 0.35
    consec_sim_min = float(np.min(consec_sims)) if consec_sims else 0.35
    consec_sim_std = float(np.std(consec_sims)) if consec_sims else 0.0

    # Discretizaciones de Shannon sobre BETO SBERT
    cuts_7 = [-np.inf, 0.3740, 0.4500, 0.5340, 0.6079, 0.6650, 0.7287, np.inf]
    shannon_bin_7 = int(pd.cut([max_intra], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7])[0])
    cuts_4 = [-np.inf, 0.3740, 0.6079, 0.7287, np.inf]
    shannon_bin_4 = int(pd.cut([max_intra], bins=cuts_4, labels=[1, 2, 3, 4])[0])

    top_pair_info = None
    if k >= 2:
        max_pair_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(k) * 999), sim_matrix.shape)
        i_p, j_p = int(max_pair_idx[0]), int(max_pair_idx[1])
        top_pair_info = {
            "oracion_a_idx": i_p + 1,
            "oracion_a_texto": oraciones[i_p],
            "oracion_b_idx": j_p + 1,
            "oracion_b_texto": oraciones[j_p],
            "similitud_coseno": round(float(sim_matrix[i_p, j_p]), 4),
            "supera_umbral_gmm": bool(sim_matrix[i_p, j_p] >= TAU_CONTINUIDAD_GMM),
            "supera_umbral_pawsx": bool(sim_matrix[i_p, j_p] >= TAU_PARAFRASIS_PAWSX),
            "supera_umbral_estricto": bool(sim_matrix[i_p, j_p] >= TAU_REDUNDANCIA_ESTRICTA),
            "supera_umbral_hiper": bool(sim_matrix[i_p, j_p] >= TAU_HIPER_REDUNDANCIA)
        }

    # 5. Las 5 Dimensiones del Índice de Manipulación Lingüística (IML) y Arquetipos (BETO SBERT)
    d1_sensacionalismo = float(np.clip(p_full * 100.0, 0.0, 100.0))
    volatilidad_raw = max(0.0, p_max - p_mean)
    d2_volatilidad_gatillo = float(np.clip(volatilidad_raw * 100.0 * 1.5, 0.0, 100.0))
    if p_max >= 0.50:
        amortiguacion_raw = max(0.0, 1.0 - dilution_ratio)
    else:
        amortiguacion_raw = 1.0
    d3_amortiguacion = float(np.clip(amortiguacion_raw * 100.0, 0.0, 100.0))
    sim_eff = 0.65 * max_intra + 0.35 * second_max_intra
    d4_redundancia = float(np.clip((sim_eff - 0.3434) / (0.7287 - 0.3434) * 100.0, 0.0, 100.0))
    if max_intra < 0.3740:
        cohesion_raw = (max_intra / 0.3740) * 55.0
    elif max_intra <= 0.6079:
        cohesion_raw = 88.0 + (1.0 - abs(max_intra - 0.4910) / 0.1170) * 12.0
    else:
        cohesion_raw = max(25.0, 90.0 - (max_intra - 0.6079) / 0.2500 * 65.0)
    d5_cohesion = float(np.clip(cohesion_raw, 0.0, 100.0))

    iml_score = float(np.clip(
        d1_sensacionalismo * 0.35 +
        d2_volatilidad_gatillo * 0.20 +
        (100.0 - d3_amortiguacion) * 0.15 +
        d4_redundancia * 0.20 +
        (100.0 - d5_cohesion) * 0.10,
        0.0, 100.0
    ))

    # Clasificación Flexible en los 5 Arquetipos Estilométricos por Afinidad
    arq_info = clasificar_arquetipo(
        d1_sensacionalismo,
        d2_volatilidad_gatillo,
        d3_amortiguacion,
        d4_redundancia,
        d5_cohesion
    )
    arquetipo_id = arq_info["id"]
    arquetipo_nombre = arq_info["nombre"]
    arquetipo_desc = arq_info["descripcion"]
    arquetipo_activador = arq_info["condicion_activada"]

    if iml_score <= 35.0:
        riesgo_iml = "Bajo Riesgo de Manipulación Estilométrica (Texto Neutro/Sobrio)"
    elif iml_score <= 60.0:
        riesgo_iml = "Riesgo Moderado de Manipulación (Retórica Comercial o Polarizada)"
    else:
        riesgo_iml = "Alto Riesgo de Manipulación Estilométrica (Anomalía Forense Severa)"

    # 6. Dimensión 2: Forense y Epistémica
    tokens_raw = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+\b', text)
    w_count = max(len(tokens_raw), 1)

    hedges_encontrados = PAT_HEDGES.findall(text)
    boosters_encontrados = PAT_BOOSTERS.findall(text)
    dicendi_encontrados = PAT_DICENDI.findall(text)
    quotes_encontradas = PAT_QUOTES.findall(text)

    hedges_density = (len(hedges_encontrados) / w_count) * 100.0
    boosters_density = (len(boosters_encontrados) / w_count) * 100.0
    epistemic_ratio = float((len(boosters_encontrados) + 0.01) / (len(hedges_encontrados) + 0.1))
    quotes_density = float(len(quotes_encontradas) / num_sentences)
    dicendi_density = (len(dicendi_encontrados) / w_count) * 100.0

    # 7. Dimensión 3: Riqueza Léxica y Legibilidad
    tokens_clean = [t.lower() for t in tokens_raw if t.isalpha()]
    w_clean_count = max(len(tokens_clean), 1)
    vocab = set(tokens_clean)
    guiraud_ttr = float(len(vocab) / np.sqrt(w_clean_count))

    counts = Counter(tokens_clean)
    n_hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = float(n_hapax / w_clean_count)
    hapax_ratio_pct = hapax_ratio * 100.0

    total_syllables = sum(count_syllables_es(t) for t in tokens_clean)
    total_chars = sum(len(t) for t in tokens_clean)
    avg_syllables_word = float(total_syllables / w_clean_count)
    avg_words_sentence = float(w_clean_count / num_sentences)

    flesch_szigriszt = float(np.clip(206.835 - 62.3 * avg_syllables_word - avg_words_sentence, 0, 120))
    gutierrez_polini = float(np.clip(95.2 - 9.7 * (total_chars / w_clean_count) - avg_words_sentence, 0, 100))

    # 8. Dimensión 4: Morfosintaxis spaCy
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

    # 9. Dimensión 5: Anclajes Factuales, Mayúsculas y Puntuación
    texto_str_no_masks = PAT_CENSORED_MASKS.sub(' ', text)
    caps_words_raw = PAT_ALL_CAPS.findall(texto_str_no_masks)
    all_caps_words = [w for w in caps_words_raw if w not in SIGLAS_EXCLUIDAS and len(w) >= 3]
    all_caps_count = len(all_caps_words)
    all_caps_ratio = (all_caps_count / w_count) * 100.0

    alpha_chars = [c for c in texto_str_no_masks if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    upper_chars_ratio_frac = float(len(upper_chars) / max(1, len(alpha_chars)))
    upper_chars_ratio_pct = upper_chars_ratio_frac * 100.0

    excl_count = texto_str_no_masks.count('!') + texto_str_no_masks.count('¡')
    quest_count = texto_str_no_masks.count('?') + texto_str_no_masks.count('¿')
    ellipsis_count = len(re.findall(r'(\.\.\.|…)', texto_str_no_masks))

    excl_density = (excl_count / w_count) * 100.0
    quest_density = (quest_count / w_count) * 100.0
    ellipsis_density = (ellipsis_count / w_count) * 100.0
    punct_intensity = ((excl_count + quest_count + ellipsis_count) / w_count) * 100.0

    numbers_list = PAT_DIGITS.findall(texto_str_no_masks) + PAT_CENSORED_NUMBER.findall(text)
    numbers_density = (len(numbers_list) / w_count) * 100.0

    percent_matches = PAT_PERCENT.findall(text)
    percent_count = len(percent_matches)
    temporal_matches = PAT_TEMPORAL.findall(texto_str_no_masks)
    temporal_density = (len(temporal_matches) / w_count) * 100.0

    # 10. Inferencia con Modelo GBDT 5D (BETO SBERT + MexGen)
    features_dict = {
        'P_full': p_full,
        'P_mean': p_mean,
        'P_max': p_max,
        'P_top2': p_top2,
        'sigma_sens': sigma_sens,
        'dilution_ratio': dilution_ratio,
        'delta_p_gatillo': delta_p_gatillo,
        'max_intra_similarity_clean': max_intra,
        'second_max_intra_similarity_clean': second_max_intra,
        'top3_mean_intra_similarity_clean': top3_mean_intra,
        'p90_intra_similarity_clean': p90_intra,
        'var_intra_similarity_clean': var_intra,
        'mean_intra_similarity_clean': mean_intra,
        'mexgen_persistence_gap': mexgen_gap,
        'delta_mean_sim_morf': delta_mean_morf,
        'topic_continuity_density': topic_cont_density,
        'paraphrase_density_p90': paraphrase_density,
        'strict_redundancy_density_p95': strict_red_density,
        'hyper_redundancy_density': hyper_red_density,
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
        'upper_chars_ratio': upper_chars_ratio_frac,
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
            for col in ohe_cols:
                x_vec.append(1.0 if 'Estándar' in col else 0.0)
            x_arr = np.array(x_vec).reshape(1, -1)
            prob_ensamble_5d = float(gbdt_artifact['model'].predict_proba(x_arr)[0, 1])
        except Exception as e:
            print(f"⚠️ Error en inferencia GBDT: {e}")
            prob_ensamble_5d = 0.55 if (all_caps_ratio > 4.0 or p_max > 0.80 or adv_density > 6.0) else 0.35
    else:
        prob_ensamble_5d = float(np.clip(
            0.30 * p_full + 0.25 * (all_caps_ratio / 15.0) + 0.20 * (adv_density / 10.0) + 0.15 * p_max - 0.10 * (dicendi_density / 2.0),
            0.05, 0.95
        ))

    # 11. Desglose Frase a Frase (Explicabilidad Local Oracional)
    desglose_oraciones = []
    sentences_analyzed = []
    for i, s_text in enumerate(oraciones):
        s_text_no_masks = PAT_CENSORED_MASKS.sub(' ', s_text)
        s_tokens = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', s_text_no_masks)
        s_w_count = max(len(s_tokens), 1)

        s_caps = [w for w in s_tokens if w.isupper() and len(w) >= 3 and w not in SIGLAS_EXCLUIDAS]
        s_caps_ratio = (len(s_caps) / s_w_count) * 100.0

        s_adv_count = sum(1 for w in s_tokens if w.lower().endswith("mente") or w.lower() in ADVERBIOS_ENFATICOS)
        s_adv_density = (s_adv_count / s_w_count) * 100.0

        s_dicendi = PAT_DICENDI.findall(s_text)
        s_boosters = PAT_BOOSTERS.findall(s_text)
        s_hedges = PAT_HEDGES.findall(s_text)
        s_has_quotes = bool(PAT_QUOTES.search(s_text))
        s_excl = s_text.count('!') + s_text.count('¡')
        s_quest = s_text.count('?') + s_text.count('¿')

        c_sim = float(consec_sims[i - 1]) if (i > 0 and i - 1 < len(consec_sims)) else None
        s_sens = float(sens_oraciones[i]) if i < len(sens_oraciones) else 0.20
        lbl_sab_s, p_fake_sab_s, p_true_sab_s = sabert_oraciones[i] if i < len(sabert_oraciones) else ("VERDADERO", 0.35, 0.65)

        if k >= 2:
            sims_otras = [float(sim_matrix[i, j]) for j in range(k) if j != i]
            mean_sim_otras = float(np.mean(sims_otras))
            max_sim_otra_idx = int(np.argmax([sim_matrix[i, j] if j != i else -1 for j in range(k)]))
            max_sim_otra_val = float(sim_matrix[i, max_sim_otra_idx])
            par_mas_cercano = {
                "idx_oracion_cercana": max_sim_otra_idx + 1,
                "similitud": round(max_sim_otra_val, 4),
                "texto_resumido": oraciones[max_sim_otra_idx][:80] + "..." if len(oraciones[max_sim_otra_idx]) > 80 else oraciones[max_sim_otra_idx]
            }
        else:
            mean_sim_otras = 0.0
            par_mas_cercano = None

        is_trigger = bool(s_sens >= 0.70 or s_caps_ratio >= 15.0 or (len(s_boosters) > 0 and s_adv_density >= 6.0))

        flags = []
        if s_sens >= 0.70:
            flags.append(f"🔥 Pico Sensacionalista ({s_sens*100:.1f}%)")
        if s_caps_ratio >= 15.0:
            flags.append(f"📢 MAYÚSCULAS Sostenidas ({s_caps_ratio:.1f}%)")
        if len(s_boosters) > 0:
            flags.append(f"⚡ Intensificador/Booster ({', '.join(set(s_boosters))})")
        if s_adv_density >= 6.0:
            flags.append(f"💬 Densidad Adverbial Alta ({s_adv_density:.1f}%)")
        if s_excl > 0:
            flags.append("❗ Puntuación Enfática (!)")
        if s_quest > 0:
            flags.append("❓ Pregunta Retórica (?)")
        if len(s_dicendi) > 0:
            flags.append(f"🎙️ Atribución Fuente ({', '.join(set(s_dicendi))})")
        if s_has_quotes:
            flags.append("📜 Cita Textual Entrecomillada")
        if len(s_hedges) > 0:
            flags.append(f"🛡️ Cautela Epistémica ({', '.join(set(s_hedges))})")
        if c_sim is not None and c_sim < 0.15:
            flags.append("⚠️ Salto Temático Abrupto (Baja Coherencia)")
        if par_mas_cercano and par_mas_cercano["similitud"] >= TAU_REDUNDANCIA:
            flags.append(f"🔁 Redundante con Frase #{par_mas_cercano['idx_oracion_cercana']} (cos={par_mas_cercano['similitud']:.2f})")

        oracion_obj = {
            "idx": i + 1,
            "texto": s_text,
            "num_palabras": s_w_count,
            "sensacionalismo_prob": round(s_sens, 4),
            "sensacionalismo_pct": round(s_sens * 100.0, 1),
            "es_oracion_gatillo": is_trigger,
            "coherencia_consecutiva": round(c_sim, 4) if c_sim is not None else None,
            "similitud_media_documento": round(mean_sim_otras, 4),
            "par_mas_redundante": par_mas_cercano,
            "mayusculas_ratio": round(s_caps_ratio, 2),
            "palabras_mayusculas": s_caps,
            "adverbios_densidad": round(s_adv_density, 2),
            "verbos_dicendi": list(set(s_dicendi)),
            "boosters": list(set(s_boosters)),
            "hedges": list(set(s_hedges)),
            "tiene_citas": s_has_quotes,
            "exclamaciones": s_excl,
            "interrogaciones": s_quest,
            "banderas_forenses": flags,
            "sabert": {
                "label": lbl_sab_s,
                "prob_fake": round(p_fake_sab_s, 4),
                "score_fake_pct": round(p_fake_sab_s * 100.0, 1)
            }
        }
        desglose_oraciones.append(oracion_obj)

        sentences_analyzed.append({
            "idx": i + 1,
            "text": s_text,
            "sens_score": round(s_sens, 3),
            "sens_pct": round(s_sens * 100, 1),
            "consec_sim": round(c_sim, 3) if c_sim is not None else None,
            "is_trigger": is_trigger,
            "all_caps_ratio": round(s_caps_ratio, 1),
            "adv_density": round(s_adv_density, 1),
            "dicendi_count": len(s_dicendi),
            "has_quotes": s_has_quotes,
            "flags": flags,
            "par_mas_redundante": par_mas_cercano
        })

    t_elapsed = time.time() - t_start

    response = {
        "text": text,
        "word_count": w_count,
        "num_sentences": num_sentences,
        "metadatos": {
            "titulo": titulo if titulo else f"Noticia Analizada ({w_count} palabras, {num_sentences} oraciones)",
            "conteo_palabras": w_count,
            "num_oraciones": num_sentences,
            "tiempo_analisis_segundos": round(t_elapsed, 3),
            "dispositivo": str(device)
        },
        "sensacionalismo_beto": {
            "P_full": round(p_full, 4),
            "P_full_pct": round(p_full * 100.0, 1),
            "P_mean": round(p_mean, 4),
            "P_max": round(p_max, 4),
            "P_top2": round(p_top2, 4),
            "sigma_sens": round(sigma_sens, 4),
            "dilution_ratio": round(dilution_ratio, 4),
            "delta_p_gatillo_causal": round(delta_p_gatillo, 4),
            "oracion_gatillo_principal": {
                "idx": idx_max + 1,
                "texto": oracion_gatillo,
                "sens_prob": round(p_max, 4)
            },
            "diagnostico": "Alarma Sensacionalista Sostenida" if p_full >= 0.65 else (
                "Cebo Sensacionalista Aislado (Gatillo Cabecera)" if (p_max >= 0.70 and p_full < 0.50) else
                "Tono Informativo Sobrio / Desprovisto de Amarillismo"
            )
        },
        "redundancia_sbert": {
            "modelo_sbert": "hiiamsid/sentence_similarity_spanish_es",
            "mean_intra_similarity": round(mean_intra, 4),
            "max_intra_similarity": round(max_intra, 4),
            "second_max_intra_similarity": round(second_max_intra, 4),
            "top3_mean_intra_similarity": round(top3_mean_intra, 4),
            "p90_intra_similarity": round(p90_intra, 4),
            "var_intra_similarity": round(var_intra, 4),
            "mexgen_persistence_gap": round(mexgen_gap, 4),
            "delta_mean_sim_morf": round(delta_mean_morf, 4),
            "redundant_pairs_count": redundant_count,
            "num_pairs_total": n_pairs,
            "redundancy_density_pct": round(redundancy_density * 100.0, 2),
            "topic_continuity_density_pct": round(topic_cont_density * 100.0, 2),
            "paraphrase_density_p90_pct": round(paraphrase_density * 100.0, 2),
            "strict_redundancy_density_p95_pct": round(strict_red_density * 100.0, 2),
            "hyper_redundancy_density_pct": round(hyper_red_density * 100.0, 2),
            "umbral_gmm_tau": TAU_CONTINUIDAD_GMM,
            "umbral_pawsx_tau": TAU_PARAFRASIS_PAWSX,
            "umbral_estricto_tau": TAU_REDUNDANCIA_ESTRICTA,
            "umbral_hiper_tau": TAU_HIPER_REDUNDANCIA,
            "shannon_bin_7": shannon_bin_7,
            "shannon_bin_4": shannon_bin_4,
            "par_maxima_redundancia": top_pair_info,
            "matriz_similitud": [[round(float(val), 4) for val in row] for row in sim_matrix],
            "diagnostico": (
                "Hiper-Redundancia Circular (Bucle de Verdad Ilusoria)" if max_intra >= TAU_HIPER_REDUNDANCIA else (
                    "Redundancia Semántica Estricta / Paráfrasis Reiterativa" if max_intra >= TAU_PARAFRASIS_PAWSX else (
                        "Redundancia Balanceada de Prensa Profesional (Banda Óptima Shannon)" if (0.3740 <= max_intra < TAU_PARAFRASIS_PAWSX) else
                        "Discurso Desarticulado / Párrafos Fragmentados"
                    )
                )
            )
        },
        "las_5_dimensiones_iml": {
            "D1_Carga_Emocional": {
                "valor": round(d1_sensacionalismo, 1),
                "escala": "[0 - 100]",
                "interpretacion": "Densidad de superlativos y dramatismo léxico asignado por BETO.",
                "nivel": "Crítico / Alarmismo Extremo" if d1_sensacionalismo >= 65.0 else (
                    "Moderado / Comercial" if d1_sensacionalismo >= 40.0 else "Sobrio / Formal"
                )
            },
            "D2_Volatilidad_Gatillo": {
                "valor": round(d2_volatilidad_gatillo, 1),
                "escala": "[0 - 100]",
                "interpretacion": "Desbalance entre la frase gatillo y la sobriedad media del artículo.",
                "nivel": "Clickbait Severo / Gatillo Aislado" if d2_volatilidad_gatillo >= 50.0 else (
                    "Desbalance Leve" if d2_volatilidad_gatillo >= 25.0 else "Tono Homogéneo"
                )
            },
            "D3_Amortiguacion_Contextual": {
                "valor": round(d3_amortiguacion, 1),
                "escala": "[0 - 100]",
                "interpretacion": "Capacidad del cuerpo del texto para diluir o mitigar la alarma del titular.",
                "nivel": "Amortiguación Alta (Cuerpo Sobrio Neutraliza Titular)" if d3_amortiguacion >= 60.0 else (
                    "Amortiguación Parcial" if d3_amortiguacion >= 30.0 else "Sin Amortiguación (Pánico Sostenido en Todo el Texto)"
                )
            },
            "D4_Reiteracion_Redundancia": {
                "valor": round(d4_redundancia, 1),
                "escala": "[0 - 100]",
                "interpretacion": "Circularidad semántica intra-documental mediante paráfrasis y reiteración.",
                "nivel": "Bucle Argumental / Hiper-Redundancia" if d4_redundancia >= 65.0 else (
                    "Progresión Normal Temática" if d4_redundancia >= 25.0 else "Diversidad Temática / Sin Repeticiones"
                )
            },
            "D5_Cohesion_Fluidez": {
                "valor": round(d5_cohesion, 1),
                "escala": "[0 - 100]",
                "interpretacion": "Conectividad proposicional y solidez sintáctica en la banda de Shannon.",
                "nivel": "Cohesión Profesional Óptima" if d5_cohesion >= 75.0 else (
                    "Cohesión Regular" if d5_cohesion >= 50.0 else "Texto Desarticulado / Roto"
                )
            },
            "score_global_iml": round(iml_score, 1),
            "nivel_riesgo_estilometrico": riesgo_iml,
            "arquetipo_discursivo": {
                "id": arquetipo_id,
                "nombre": arquetipo_nombre,
                "descripcion": arquetipo_desc,
                "afinidad_pct": arq_info.get("afinidad_pct", 100.0),
                "arquetipo_secundario": arq_info.get("arquetipo_secundario", ""),
                "afinidad_secundaria_pct": arq_info.get("afinidad_secundaria_pct", 0.0),
                "afinidades_pct": arq_info.get("afinidades_pct", {}),
                "condicion_activada": arquetipo_activador
            }
        },
        "variables_ensamble_avanzado_5d": {
            "D1_Dinamica_Discursiva": {
                "consec_sim_mean": round(consec_sim_mean, 4),
                "consec_sim_min": round(consec_sim_min, 4),
                "consec_sim_std": round(consec_sim_std, 4)
            },
            "D2_Linguistica_Forense_Epistemica": {
                "hedges_density": round(hedges_density, 3),
                "boosters_density": round(boosters_density, 3),
                "epistemic_ratio": round(epistemic_ratio, 3),
                "quotes_density": round(quotes_density, 3),
                "dicendi_density": round(dicendi_density, 3),
                "tokens_hedges": list(set(hedges_encontrados)),
                "tokens_boosters": list(set(boosters_encontrados)),
                "tokens_dicendi": list(set(dicendi_encontrados))
            },
            "D3_Riqueza_Lexica_Legibilidad": {
                "flesch_szigriszt": round(flesch_szigriszt, 2),
                "gutierrez_polini": round(gutierrez_polini, 2),
                "guiraud_ttr": round(guiraud_ttr, 3),
                "hapax_ratio_pct": round(hapax_ratio_pct, 2),
                "vocabulario_unico": len(vocab),
                "promedio_silabas_palabra": round(avg_syllables_word, 2),
                "promedio_palabras_oracion": round(avg_words_sentence, 2)
            },
            "D4_Perfilado_Morfosintactico_POS": {
                "adv_density": round(adv_density, 3),
                "pron_1p_density": round(pron_1p_density, 3),
                "pron_3p_density": round(pron_3p_density, 3),
                "adj_noun_ratio": round(adj_noun_ratio, 3),
                "conteo_adjetivos": n_adj,
                "conteo_sustantivos": n_noun,
                "conteo_adverbios": n_adv,
                "conteo_pronombres_1p": n_p1,
                "conteo_pronombres_3p": n_p3
            },
            "D5_Anclajes_Factuales_Puntuacion": {
                "all_caps_count": all_caps_count,
                "all_caps_ratio_pct": round(all_caps_ratio, 2),
                "upper_chars_ratio_pct": round(upper_chars_ratio_pct, 2),
                "excl_density": round(excl_density, 3),
                "quest_density": round(quest_density, 3),
                "ellipsis_density": round(ellipsis_density, 3),
                "punct_intensity": round(punct_intensity, 3),
                "numbers_density": round(numbers_density, 3),
                "percent_count": percent_count,
                "temporal_density": round(temporal_density, 3),
                "palabras_mayusculas_encontradas": all_caps_words[:10],
                "marcadores_temporales_encontrados": list(set(temporal_matches))
            }
        },
        "desglose_oraciones": desglose_oraciones,
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
