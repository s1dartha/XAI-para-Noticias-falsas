import os
import sys
import torch
import numpy as np
import spacy
import nltk
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__, static_folder=".")
CORS(app)

print("🚀 Inicializando Servidor de Inferencia para Pipeline Tesis (MexGen)...")

# Configuración NLTK y SpaCy
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    nlp = spacy.load("es_core_news_sm")
except Exception:
    os.system(f'"{sys.executable}" -m spacy download es_core_news_sm')
    nlp = spacy.load("es_core_news_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Dispositivo PyTorch: {device}")

# 1. SaBERT
print("⏳ Cargando SaBERT Spanish Fake News...")
sabert_tokenizer = BertTokenizer.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News")
sabert_model = BertForSequenceClassification.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News").to(device)
sabert_model.eval()

# 2. Sensacionalismo
print("⏳ Cargando Bert Spanish Sensationalism...")
sensationalism_classifier = pipeline(
    "text-classification",
    model="JJNeila/bert-spanish-sensationalism-oss",
    tokenizer="JJNeila/bert-spanish-sensationalism-oss",
    device=0 if torch.cuda.is_available() else -1,
    top_k=None
)

# 3. SBERT (Redundancia)
print("⏳ Cargando SBERT (paraphrase-multilingual-mpnet-base-v2)...")
sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=str(device))

print("✅ Todos los modelos cargados en memoria. Servidor Listo.")

def mexgen_extraer_parrafos(texto):
    if not isinstance(texto, str) or not texto.strip():
        return []
    lineas = [p.strip() for p in texto.split("\n") if len(p.strip()) > 10]
    if len(lineas) >= 2:
        return lineas
    doc = nlp(texto)
    oraciones = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
    if len(oraciones) >= 2:
        return oraciones
    fragmentos = [frag.strip() for frag in texto.replace('\n', ' ').split('.') if len(frag.strip()) > 15]
    return fragmentos if fragmentos else [texto.strip()]

def evaluar_sabert(texto):
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
    res = sensationalism_classifier(str(texto))[0]
    score_sens = next(item["score"] for item in res if item["label"] in ["LABEL_1", "Sensacionalista"])
    etiqueta = "Sensacionalista" if score_sens >= 0.50 else "NO Sensacional"
    return etiqueta, float(score_sens)

def evaluar_redundancia(lista_parrafos, umbral=0.70):
    if len(lista_parrafos) < 2:
        return "NO Redundante", 0.0, {0: (0.0, "NO Redundante")}
    embeddings = sbert_model.encode(lista_parrafos, convert_to_tensor=True)
    matriz_sim = util.cos_sim(embeddings, embeddings).cpu().numpy()
    n = len(lista_parrafos)
    similitudes_max = []
    detalles = {}
    for i in range(n):
        sims_otros = [float(matriz_sim[i][j]) for j in range(n) if i != j]
        max_sim = float(np.max(sims_otros)) if sims_otros else 0.0
        lbl = "Redundante" if max_sim >= umbral else "NO Redundante"
        detalles[i] = (max_sim, lbl)
        similitudes_max.append(max_sim)
    score_global = float(np.mean(similitudes_max)) if similitudes_max else 0.0
    label_global = "Redundante" if score_global >= umbral else "NO Redundante"
    return label_global, score_global, detalles

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_news():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No se proporcionó texto de noticia"}), 400

    # 1. MexGen Paragraph Extraction
    parrafos_raw = mexgen_extraer_parrafos(text)

    # 2. SaBERT Noticia General
    lbl_gen, prob_fake_gen, prob_true_gen = evaluar_sabert(text)
    
    # 3. SaBERT & Sensacionalismo Por Párrafo
    parrafos_eval = []
    for idx, p in enumerate(parrafos_raw, 1):
        lbl_p, prob_fake_p, prob_true_p = evaluar_sabert(p)
        lbl_sens_p, score_sens_p = evaluar_sensacionalismo(p)
        parrafos_eval.append({
            "id": idx,
            "name": f"Párrafo {idx}",
            "short_name": f"P{idx}",
            "text": p,
            "sabert": {
                "label": lbl_p,
                "prob_fake": prob_fake_p,
                "prob_true": prob_true_p,
                "score_fake_pct": round(prob_fake_p * 100, 1)
            },
            "sensationalism": {
                "label": lbl_sens_p,
                "score": score_sens_p,
                "score_pct": round(score_sens_p * 100, 1)
            }
        })

    # 4. Filter MexGen Top Scores
    parrafos_ordenados = sorted(parrafos_eval, key=lambda x: x["sabert"]["prob_fake"], reverse=True)
    top_k_filtrados = parrafos_ordenados[:2]
    ids_filtrados = set(p["id"] for p in top_k_filtrados)

    # Assign MexGen rank
    for rank, p in enumerate(parrafos_ordenados, 1):
        for orig_p in parrafos_eval:
            if orig_p["id"] == p["id"]:
                orig_p["mexgen_rank"] = rank
                orig_p["is_mexgen_top"] = orig_p["id"] in ids_filtrados

    # 5. Sensacionalismo General
    lbl_sens_gen, score_sens_gen = evaluar_sensacionalismo(text)

    # 6. Redundancia General y Por Párrafo
    lbl_red_gen, score_red_gen, detalles_red = evaluar_redundancia(parrafos_raw)

    for i, p in enumerate(parrafos_eval):
        max_sim_p, lbl_red_p = detalles_red.get(i, (0.0, "NO Redundante"))
        p["redundancy"] = {
            "label": lbl_red_p,
            "score": max_sim_p,
            "score_pct": round(max_sim_p * 100, 1)
        }

    result = {
        "general": {
            "label": lbl_gen,
            "prob_fake": prob_fake_gen,
            "prob_true": prob_true_gen,
            "prob_fake_pct": round(prob_fake_gen * 100, 1)
        },
        "sensationalism": {
            "label": lbl_sens_gen,
            "score": score_sens_gen,
            "score_pct": round(score_sens_gen * 100, 1)
        },
        "redundancy": {
            "label": lbl_red_gen,
            "score": score_red_gen,
            "score_pct": round(score_red_gen * 100, 1)
        },
        "paragraphs": parrafos_eval,
        "filtered_paragraph_ids": list(ids_filtrados)
    }

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Servidor escuchando en http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
EOF
