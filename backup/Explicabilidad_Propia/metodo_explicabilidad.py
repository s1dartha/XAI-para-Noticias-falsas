"""
=============================================================================
MÉTODO EXPLICABLE DE AUDITORÍA FORENSE Y EXTRACCIÓN DE CARACTERÍSTICAS TEXTUALES
=============================================================================
Desarrollado para la Tesis de Maestría / Grado en Inteligencia Artificial y PLN.
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/metodo_explicabilidad.py

Este módulo implementa un marco de explicabilidad intrínseca y auditoría forense
que rechaza el reduccionismo de la clasificación binaria de caja negra ("Verdadero vs Falso").
En su lugar, disecciona cualquier noticia entrante a través de:

1. El Modelo de Sensacionalismo Documental y Oracional (BETO fine-tuned)
   - Probabilidad documental (P_full), media (P_mean), pico gatillo (P_max),
     top-2 oracional (P_top2), dispersión afectiva (sigma_sens),
     ratio de dilución contextual (dilution_ratio) e impacto causal de gatillos (delta_p_gatillo).
2. El Modelo de Redundancia Semántica Intra-Documental (SBERT)
   - Similitud coseno media y máxima, recuento y densidad de pares redundantes
     (tau = 0.34 calibrado con GMM), discretización en 7 y 4 bins de Shannon,
     y desglose de oraciones redundantes/paráfrasis cruzadas.
3. Las 5 Dimensiones del Índice de Manipulación Lingüística (IML):
   - D1: Carga Emocional / Sensacionalismo [0 - 100]
   - D2: Volatilidad de Gatillo / Desbalance Cabecera-Cuerpo [0 - 100]
   - D3: Amortiguación Contextual / Resiliencia [0 - 100]
   - D4: Reiteración y Bucle Argumental / Hiper-Redundancia [0 - 100]
   - D5: Cohesión y Fluidez Discursiva [0 - 100]
   - Índice Global IML [0 - 100] y Diagnóstico en los 5 Arquetipos Estilométricos.
4. Las 5 Dimensiones de Variables Enriquecidas del Ensamble Avanzado:
   - D1: Flujo de Coherencia Secuencial (consec_sim_mean, consec_sim_min, consec_sim_std)
   - D2: Lingüística Forense y Marcadores Epistémicos (hedges, boosters, dicendi, quotes)
   - D3: Riqueza Léxica y Legibilidad (Flesch-Szigriszt, Gutiérrez de Polini, Guiraud TTR, Hapax)
   - D4: Morfosintaxis y Subjetividad spaCy (adverbios, pronombres 1P/3P, ratio adjetivo/sustantivo)
   - D5: Anclajes Factuales y Puntuación (mayúsculas sostenidas, números, fechas, puntuación enfática)
5. Auditoría Frase a Frase:
   - Identificación de oraciones gatillo, saltos temáticos, citas, atribuciones y advertencias estilísticas.
=============================================================================
"""

import os
import sys
import re
import json
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import spacy

# Descargas silenciosas de NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
PAT_PERCENT = re.compile(r'(?:\d+\s*%|\bpor\s+ciento\b)', re.IGNORECASE)
PAT_ALL_CAPS = re.compile(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
SIGLAS_EXCLUIDAS = {'ONU', 'UE', 'EEUU', 'OMS', 'PIB', 'FMI', 'OTAN', 'VOX', 'PSOE', 'PP', 'INE', 'USA', 'EFE', 'IAC', 'NICA'}

PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}
ADVERBIOS_ENFATICOS = {'muy', 'más', 'tan', 'bastante', 'casi', 'apenas', 'siempre', 'nunca', 'jamás', 'ya', 'totalmente', 'absolutamente', 'demasiado'}

TAU_REDUNDANCIA = 0.34  # Umbral bayesiano calibrado con GMM en la tesis

def count_syllables_es(word: str) -> int:
    """Conteo heurístico de núcleos silábicos en español considerando diptongos."""
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


class AuditorExplicabilidadNoticia:
    """
    Auditor Forense de Noticias basado en las 5 Dimensiones del IML,
    Sensacionalismo de BETO, Redundancia de SBERT y Variables Enriquecidas 5D.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"🚀 [AuditorExplicable] Inicializando con dispositivo: {self.device}")

        # 1. Cargar spaCy
        print("⏳ Cargando modelo morfosintáctico spaCy ('es_core_news_sm')...")
        try:
            self.nlp = spacy.load("es_core_news_sm", disable=['ner'])
        except Exception:
            os.system(f'"{sys.executable}" -m spacy download es_core_news_sm')
            self.nlp = spacy.load("es_core_news_sm", disable=['ner'])

        # 2. Cargar BETO Sensacionalismo
        self.sens_model_name = "JJNeila/bert-spanish-sensationalism-oss"
        print(f"⏳ Cargando BETO Sensacionalismo ({self.sens_model_name})...")
        self.sens_tokenizer = AutoTokenizer.from_pretrained(self.sens_model_name)
        self.sens_model = AutoModelForSequenceClassification.from_pretrained(self.sens_model_name).to(self.device)
        self.sens_model.eval()

        # 3. Cargar SBERT Redundancia
        self.sbert_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"⏳ Cargando SBERT ({self.sbert_name})...")
        self.sbert_model = SentenceTransformer(self.sbert_name, device=str(self.device))

        print("✅ [AuditorExplicable] Modelos cargados y listos para auditoría forense.\n")

    def segmentar_oraciones(self, texto: str) -> List[str]:
        """
        Segmentación oracional adaptada a noticias en español,
        respetando titulares con saltos de línea y signos de puntuación.
        """
        raw_text = str(texto).strip()
        # Segmentación estándar NLTK
        sents = [s.strip() for s in nltk.sent_tokenize(raw_text, language='spanish') if len(s.strip()) > 8]
        if not sents:
            # Fallback por saltos de línea o puntuación fuerte
            sents = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', raw_text) if len(s.strip()) > 8]
        return sents if sents else [raw_text[:200]]

    def predecir_sensacionalismo(self, textos: List[str], max_len: int = 256) -> List[float]:
        """Calcula la probabilidad de sensacionalismo [0, 1] para una lista de textos."""
        probs = []
        batch_size = 32
        for i in range(0, len(textos), batch_size):
            batch = [str(t)[:1000] for t in textos[i:i+batch_size]]
            inputs = self.sens_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                logits = self.sens_model(**inputs).logits
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend([float(x) for x in p])
        return probs

    def codificar_oraciones_sbert(self, oraciones: List[str]) -> np.ndarray:
        """Genera embeddings normalizados SBERT para oraciones."""
        embs = self.sbert_model.encode(
            oraciones,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embs.cpu().numpy()

    def auditar_noticia(self, texto: str, titulo: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta la auditoría completa de características explicables sobre una noticia.
        NO EMITE CLASIFICACIÓN BINARIA; entrega la anatomía estilométrica y forense completa.
        """
        t_start = time.time()
        texto_str = str(texto).strip()
        oraciones = self.segmentar_oraciones(texto_str)
        num_sentences = max(len(oraciones), 1)

        # ---------------------------------------------------------------------
        # 1. Sensacionalismo Documental, Oracional y Causal (BETO)
        # ---------------------------------------------------------------------
        # Global
        p_full = self.predecir_sensacionalismo([texto_str], max_len=256)[0]
        # Oracional
        sens_oraciones = self.predecir_sensacionalismo(oraciones, max_len=128)

        p_mean = float(np.mean(sens_oraciones))
        p_max = float(np.max(sens_oraciones))
        idx_max = int(np.argmax(sens_oraciones))
        oracion_gatillo = oraciones[idx_max]

        # Top 2
        p_sorted = sorted(sens_oraciones, reverse=True)
        p_top2 = float(np.mean(p_sorted[:2])) if len(p_sorted) >= 2 else p_max
        sigma_sens = float(np.std(sens_oraciones)) if len(sens_oraciones) > 1 else 0.0

        # Dilution Ratio
        dilution_ratio = float(p_full / (p_max + 1e-6))

        # Ablación causal contrafáctica (silenciar top 2 frases más sensacionalistas)
        if len(oraciones) >= 3:
            indices_top2 = set(np.argsort(sens_oraciones)[-2:])
            oraciones_abladas = [s for i, s in enumerate(oraciones) if i not in indices_top2]
            texto_ablado = " ".join(oraciones_abladas)
            p_ablada = self.predecir_sensacionalismo([texto_ablado], max_len=256)[0]
            delta_p_gatillo = float(max(0.0, p_full - p_ablada))
        else:
            p_ablada = p_mean
            delta_p_gatillo = float(max(0.0, p_full - p_mean))

        # ---------------------------------------------------------------------
        # 2. Redundancia Semántica Intra-Documental y Flujo Secuencial (SBERT)
        # ---------------------------------------------------------------------
        embeddings = self.codificar_oraciones_sbert(oraciones)
        k = len(oraciones)

        # Matriz de similitud coseno completa
        sim_matrix = np.dot(embeddings, embeddings.T)  # (k, k)

        consec_sims = []
        if k >= 2:
            for i in range(k - 1):
                consec_sims.append(float(sim_matrix[i, i + 1]))

            # Pares superiores estrictos
            triu_indices = np.triu_indices(k, k=1)
            pairwise_sims = sim_matrix[triu_indices]
            n_pairs = len(pairwise_sims)

            mean_intra = float(np.mean(pairwise_sims))
            max_intra = float(np.max(pairwise_sims))
            redundant_count = int(np.sum(pairwise_sims >= TAU_REDUNDANCIA))
            redundancy_density = float(redundant_count / max(n_pairs, 1))

            consec_sim_mean = float(np.mean(consec_sims))
            consec_sim_min = float(np.min(consec_sims))
            consec_sim_std = float(np.std(consec_sims))
        else:
            consec_sims = []
            n_pairs = 0
            mean_intra = 0.25
            max_intra = 0.40
            redundant_count = 0
            redundancy_density = 0.0
            consec_sim_mean = 0.45
            consec_sim_min = 0.45
            consec_sim_std = 0.0

        # Discretizaciones de Shannon
        cuts_7 = [-np.inf, 0.5924, 0.6177, 0.6336, 0.8077, 0.8514, 0.9308, np.inf]
        shannon_bin_7 = int(pd.cut([max_intra], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7])[0])

        cuts_4 = [-np.inf, 0.60, 0.81, 0.98, np.inf]
        shannon_bin_4 = int(pd.cut([max_intra], bins=cuts_4, labels=[1, 2, 3, 4])[0])

        # Encontrar el par de oraciones con máxima similitud
        top_pair_info = None
        if k >= 2:
            max_pair_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(k)*999), sim_matrix.shape)
            i_p, j_p = int(max_pair_idx[0]), int(max_pair_idx[1])
            top_pair_info = {
                "oracion_a_idx": i_p + 1,
                "oracion_a_texto": oraciones[i_p],
                "oracion_b_idx": j_p + 1,
                "oracion_b_texto": oraciones[j_p],
                "similitud_coseno": round(float(sim_matrix[i_p, j_p]), 4),
                "supera_umbral_gmm": bool(sim_matrix[i_p, j_p] >= TAU_REDUNDANCIA)
            }

        # ---------------------------------------------------------------------
        # 3. Las 5 Dimensiones del Índice de Manipulación Lingüística (IML)
        # ---------------------------------------------------------------------
        # D1: Carga Emocional / Sensacionalismo (0 - 100)
        d1_sensacionalismo = float(np.clip(p_full * 100.0, 0.0, 100.0))

        # D2: Volatilidad de Gatillo / Desbalance Cabecera-Cuerpo (0 - 100)
        volatilidad_raw = max(0.0, p_max - p_mean)
        d2_volatilidad_gatillo = float(np.clip(volatilidad_raw * 100.0 * 1.5, 0.0, 100.0))

        # D3: Amortiguación Contextual / Resiliencia (0 - 100)
        if p_max >= 0.50:
            amortiguacion_raw = max(0.0, 1.0 - dilution_ratio)
        else:
            amortiguacion_raw = 1.0  # Texto sin picos de alarma
        d3_amortiguacion = float(np.clip(amortiguacion_raw * 100.0, 0.0, 100.0))

        # D4: Reiteración y Bucle Argumental / Redundancia (0 - 100)
        d4_redundancia = float(np.clip((max_intra - 0.34) / (0.90 - 0.34) * 100.0, 0.0, 100.0))

        # D5: Cohesión y Fluidez Discursiva (0 - 100)
        if max_intra <= 0.5924:
            cohesion_raw = (max_intra / 0.5924) * 60.0  # Desarticulado
        elif max_intra <= 0.8077:
            cohesion_raw = 90.0 + (1.0 - abs(max_intra - 0.625) / 0.18) * 10.0  # Óptimo profesional
        else:
            cohesion_raw = max(30.0, 100.0 - (max_intra - 0.8077) / 0.19 * 70.0)  # Bucle
        d5_cohesion = float(np.clip(cohesion_raw, 0.0, 100.0))

        # Score Global IML (0 - 100)
        iml_score = (
            d1_sensacionalismo * 0.35 +
            d2_volatilidad_gatillo * 0.20 +
            (100.0 - d3_amortiguacion) * 0.15 +
            d4_redundancia * 0.20 +
            (100.0 - d5_cohesion) * 0.10
        )
        iml_score = float(np.clip(iml_score, 0.0, 100.0))

        # Clasificación en los 5 Arquetipos Estilométricos
        if d1_sensacionalismo >= 65.0 and d4_redundancia >= 65.0:
            arquetipo_id = "ESTILO_I_DESINFO_ESTRIDENTE"
            arquetipo_nombre = "Estilo I: Desinformación Estridente / Cliché Hiperbólico"
            arquetipo_desc = "Texto saturado homogéneamente de emociones extremas con reiteración circular de premisas."
            arquetipo_activador = "D1 >= 65.0 y D4 >= 65.0"
        elif d2_volatilidad_gatillo >= 50.0 and d3_amortiguacion >= 50.0:
            arquetipo_id = "ESTILO_II_CLICKBAIT_CABECERA"
            arquetipo_nombre = "Estilo II: Cebo Comercial / Clickbait de Cabecera"
            arquetipo_desc = "Titular o apertura alarmista diseñado para capturar clics, pero con cuerpo informativo formal que amortigua la alarma."
            arquetipo_activador = "D2 >= 50.0 y D3 >= 50.0"
        elif d1_sensacionalismo < 45.0 and d4_redundancia >= 65.0:
            arquetipo_id = "ESTILO_III_PROPAGANDA_SOBRIA"
            arquetipo_nombre = "Estilo III: Propaganda Institucional / Astroturfing Encubierto"
            arquetipo_desc = "Tono aparentemente sobrio y académico, pero con bucles de repetición semántica calculados para inducir verdad ilusoria."
            arquetipo_activador = "D1 < 45.0 y D4 >= 65.0"
        elif d5_cohesion <= 55.0 and d4_redundancia < 40.0:
            arquetipo_id = "ESTILO_IV_DESARTICULADO"
            arquetipo_nombre = "Estilo IV: Incoherencia Estructural / Generación Rota"
            arquetipo_desc = "Texto con baja conectividad proposicional, fragmentación sintáctica o párrafos desvinculados."
            arquetipo_activador = "D5 <= 55.0 y D4 < 40.0"
        else:
            arquetipo_id = "ESTILO_V_PERIODISMO_EQUILIBRADO"
            arquetipo_nombre = "Estilo V: Periodismo Profesional Balanceado"
            arquetipo_desc = "Redacción equilibrada con variedad léxica, desarrollo temático progresivo y ausencia de manipulaciones estilísticas."
            arquetipo_activador = "Comportamiento estilométrico equilibrado sin anomalías severas"

        # Nivel de Riesgo Estilométrico IML
        if iml_score <= 35.0:
            riesgo_iml = "Bajo Riesgo de Manipulación Estilométrica (Texto Neutro/Sobrio)"
        elif iml_score <= 60.0:
            riesgo_iml = "Riesgo Moderado de Manipulación (Retórica Comercial o Polarizada)"
        else:
            riesgo_iml = "Alto Riesgo de Manipulación Estilométrica (Anomalía Forense Severa)"

        # ---------------------------------------------------------------------
        # 4. Variables Enriquecidas de las 5 Dimensiones (Ensamble Avanzado)
        # ---------------------------------------------------------------------
        # Tokens y conteos
        tokens_raw = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+\b', texto_str)
        w_count = max(len(tokens_raw), 1)

        # D2: Forense y Epistémica
        hedges_encontrados = PAT_HEDGES.findall(texto_str)
        boosters_encontrados = PAT_BOOSTERS.findall(texto_str)
        dicendi_encontrados = PAT_DICENDI.findall(texto_str)
        quotes_encontradas = PAT_QUOTES.findall(texto_str)

        hedges_density = (len(hedges_encontrados) / w_count) * 100.0
        boosters_density = (len(boosters_encontrados) / w_count) * 100.0
        epistemic_ratio = float((len(boosters_encontrados) + 0.01) / (len(hedges_encontrados) + 0.1))
        quotes_density = float(len(quotes_encontradas) / num_sentences)
        dicendi_density = (len(dicendi_encontrados) / w_count) * 100.0

        # D3: Riqueza Léxica y Legibilidad
        tokens_clean = [t.lower() for t in tokens_raw if t.isalpha()]
        w_clean_count = max(len(tokens_clean), 1)
        vocab = set(tokens_clean)
        guiraud_ttr = float(len(vocab) / np.sqrt(w_clean_count))

        token_counts = Counter(tokens_clean)
        n_hapax = sum(1 for c in token_counts.values() if c == 1)
        hapax_ratio = float(n_hapax / w_clean_count) * 100.0

        total_syllables = sum(count_syllables_es(t) for t in tokens_clean)
        total_chars = sum(len(t) for t in tokens_clean)
        avg_syllables_word = float(total_syllables / w_clean_count)
        avg_words_sentence = float(w_clean_count / num_sentences)

        flesch_szigriszt = float(np.clip(206.835 - 62.3 * avg_syllables_word - avg_words_sentence, 0, 120))
        gutierrez_polini = float(np.clip(95.2 - 9.7 * (total_chars / w_clean_count) - avg_words_sentence, 0, 100))

        # D4: Morfosintaxis spaCy
        doc = self.nlp(texto_str)
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

        # D5: Anclajes Factuales, Mayúsculas y Puntuación
        caps_words_raw = PAT_ALL_CAPS.findall(texto_str)
        caps_words = [w for w in caps_words_raw if w not in SIGLAS_EXCLUIDAS and len(w) >= 3]
        all_caps_count = len(caps_words)
        all_caps_ratio = (all_caps_count / w_count) * 100.0

        alpha_chars = [c for c in texto_str if c.isalpha()]
        upper_chars = [c for c in alpha_chars if c.isupper()]
        upper_chars_ratio = float(len(upper_chars) / max(len(alpha_chars), 1)) * 100.0

        excl_count = texto_str.count('!') + texto_str.count('¡')
        quest_count = texto_str.count('?') + texto_str.count('¿')
        ellipsis_count = len(re.findall(r'(\.\.\.|…)', texto_str))

        excl_density = (excl_count / w_count) * 100.0
        quest_density = (quest_count / w_count) * 100.0
        ellipsis_density = (ellipsis_count / w_count) * 100.0
        punct_intensity = ((excl_count + quest_count + ellipsis_count) / w_count) * 100.0

        numbers_list = PAT_DIGITS.findall(texto_str)
        numbers_density = (len(numbers_list) / w_count) * 100.0

        percent_matches = PAT_PERCENT.findall(texto_str)
        percent_count = len(percent_matches)

        temporal_matches = PAT_TEMPORAL.findall(texto_str)
        temporal_density = (len(temporal_matches) / w_count) * 100.0

        # ---------------------------------------------------------------------
        # 5. Desglose Frase a Frase (Explicabilidad Local Oracional)
        # ---------------------------------------------------------------------
        desglose_oraciones = []
        for i, s_text in enumerate(oraciones):
            s_tokens = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', s_text)
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

            # Métricas relacionales SBERT
            c_sim = float(consec_sims[i - 1]) if (i > 0 and i - 1 < len(consec_sims)) else None
            s_sens = float(sens_oraciones[i])

            # Relación con el resto del documento
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

            # Oración Gatillo
            is_trigger = bool(s_sens >= 0.70 or s_caps_ratio >= 15.0 or (len(s_boosters) > 0 and s_adv_density >= 6.0))

            # Banderas forenses
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

            desglose_oraciones.append({
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
                "banderas_forenses": flags
            })

        t_elapsed = time.time() - t_start

        # ---------------------------------------------------------------------
        # Estructura del Resultado de Auditoría Forense
        # ---------------------------------------------------------------------
        resultado = {
            "metadatos": {
                "titulo": titulo if titulo else f"Noticia ({w_count} palabras, {num_sentences} oraciones)",
                "conteo_palabras": w_count,
                "num_oraciones": num_sentences,
                "tiempo_analisis_segundos": round(t_elapsed, 3),
                "dispositivo": str(self.device),
                "metadata_adicional": metadata or {}
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
                "mean_intra_similarity": round(mean_intra, 4),
                "max_intra_similarity": round(max_intra, 4),
                "redundant_pairs_count": redundant_count,
                "num_pairs_total": n_pairs,
                "redundancy_density_pct": round(redundancy_density * 100.0, 2),
                "umbral_gmm_tau": TAU_REDUNDANCIA,
                "shannon_bin_7": shannon_bin_7,
                "shannon_bin_4": shannon_bin_4,
                "par_maxima_redundancia": top_pair_info,
                "matriz_similitud": [[round(float(val), 4) for val in row] for row in sim_matrix],
                "diagnostico": (
                    "Hiper-Redundancia Circular (Bucle de Verdad Ilusoria)" if max_intra >= 0.80 else (
                        "Redundancia Balanceada de Prensa Profesional (Banda Óptima Shannon)" if (0.59 <= max_intra <= 0.80) else
                        "Discurso Desarticulado / Párrafos Fragmentados"
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
                    "hapax_ratio_pct": round(hapax_ratio, 2),
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
                    "upper_chars_ratio_pct": round(upper_chars_ratio, 2),
                    "excl_density": round(excl_density, 3),
                    "quest_density": round(quest_density, 3),
                    "ellipsis_density": round(ellipsis_density, 3),
                    "punct_intensity": round(punct_intensity, 3),
                    "numbers_density": round(numbers_density, 3),
                    "percent_count": percent_count,
                    "temporal_density": round(temporal_density, 3),
                    "palabras_mayusculas_encontradas": caps_words[:10],
                    "marcadores_temporales_encontrados": list(set(temporal_matches))
                }
            },
            "desglose_oraciones": desglose_oraciones
        }

        return resultado

    def formatear_informe_markdown(self, res: Dict[str, Any]) -> str:
        """Genera un informe forense detallado en formato Markdown."""
        meta = res["metadatos"]
        sens = res["sensacionalismo_beto"]
        red = res["redundancia_sbert"]
        iml = res["las_5_dimensiones_iml"]
        vars_5d = res["variables_ensamble_avanzado_5d"]
        oraciones = res["desglose_oraciones"]

        md = []
        md.append(f"# Auditoría Forense Explicable de Noticia: {meta['titulo']}")
        md.append(f"**Longitud:** {meta['conteo_palabras']} palabras | **Oraciones:** {meta['num_oraciones']} | **Tiempo de Análisis:** {meta['tiempo_analisis_segundos']} s\n")

        md.append("## 1. Perfil Estilométrico y Las 5 Dimensiones del IML")
        md.append(f"**Score Global IML:** `{iml['score_global_iml']} / 100` — *{iml['nivel_riesgo_estilometrico']}*  ")
        md.append(f"**Arquetipo de Estilo Asignado:** **{iml['arquetipo_discursivo']['nombre']}**  ")
        md.append(f"> **Justificación:** {iml['arquetipo_discursivo']['descripcion']} (Regla: `{iml['arquetipo_discursivo']['condicion_activada']}`)\n")

        md.append("| Dimensión IML | Puntuación (0-100) | Nivel Forense | Descripción |")
        md.append("|---|:---:|---|---|")
        for k in ["D1_Carga_Emocional", "D2_Volatilidad_Gatillo", "D3_Amortiguacion_Contextual", "D4_Reiteracion_Redundancia", "D5_Cohesion_Fluidez"]:
            d_obj = iml[k]
            md.append(f"| **{k}** | `{d_obj['valor']}` | {d_obj['nivel']} | {d_obj['interpretacion']} |")

        md.append("\n## 2. Sensacionalismo Documental y Oracional (BETO)")
        md.append(f"- **Sensacionalismo Global ($P_{{full}}$):** `{sens['P_full']}` ({sens['P_full_pct']}%)")
        md.append(f"- **Promedio Oracional ($P_{{mean}}$):** `{sens['P_mean']}` | **Pico Máximo ($P_{{max}}$):** `{sens['P_max']}` | **Top-2 Oracional ($P_{{top2}}$):** `{sens['P_top2']}`")
        md.append(f"- **Desviación Estándar Afectiva ($\\sigma_{{sens}}$):** `{sens['sigma_sens']}`")
        md.append(f"- **Dilution Ratio ($DR$):** `{sens['dilution_ratio']}`")
        md.append(f"- **Impacto Causal Contrafáctico ($\\Delta P_{{gatillo}}$):** `{sens['delta_p_gatillo_causal']}` (Reducción al silenciar los 2 gatillos)")
        gat = sens["oracion_gatillo_principal"]
        md.append(f"- **Oración Gatillo Principal (Frase #{gat['idx']}):**")
        md.append(f"  > *\"{gat['texto']}\"* (Sensacionalismo: `{gat['sens_prob']}`)")
        md.append(f"- **Diagnóstico Forense:** {sens['diagnostico']}\n")

        md.append("## 3. Redundancia Semántica Intra-Documental (SBERT)")
        md.append(f"- **Similitud Coseno Media:** `{red['mean_intra_similarity']}` | **Similitud Coseno Máxima:** `{red['max_intra_similarity']}`")
        md.append(f"- **Pares Redundantes ($\\cos \\ge {red['umbral_gmm_tau']}$):** `{red['redundant_pairs_count']}` de `{red['num_pairs_total']}` pares ({red['redundancy_density_pct']}%)")
        md.append(f"- **Discretización Entrópica de Shannon:** Bin 7: `{red['shannon_bin_7']}` | Bin 4: `{red['shannon_bin_4']}`")
        if red["par_maxima_redundancia"]:
            pmr = red["par_maxima_redundancia"]
            md.append(f"- **Par con Máxima Coincidencia Semántica (cos = {pmr['similitud_coseno']}):**")
            md.append(f"  1. Frase #{pmr['oracion_a_idx']}: *\"{pmr['oracion_a_texto']}\"*")
            md.append(f"  2. Frase #{pmr['oracion_b_idx']}: *\"{pmr['oracion_b_texto']}\"*")
        md.append(f"- **Diagnóstico Forense:** {red['diagnostico']}\n")

        md.append("## 4. Variables Enriquecidas de las 5 Dimensiones")
        md.append("| Categoría | Variable | Valor | Significado Forense |")
        md.append("|---|---|:---:|---|")
        # D1
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_mean` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_mean']}` | Coherencia media entre oraciones contiguas |")
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_min` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_min']}` | Ruptura o salto temático más pronunciado |")
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_std` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_std']}` | Inestabilidad en la transición de ideas |")
        # D2
        md.append(f"| **D2: Forense Epistémica** | `dicendi_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['dicendi_density']}%` | Densidad de verbos de reporte y atribución a fuentes |")
        md.append(f"| **D2: Forense Epistémica** | `quotes_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['quotes_density']}` | Citas directas entrecomilladas por oración |")
        md.append(f"| **D2: Forense Epistémica** | `hedges_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['hedges_density']}%` | Atenuadores de cautela ('presunto', 'al parecer') |")
        md.append(f"| **D2: Forense Epistémica** | `boosters_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['boosters_density']}%` | Intensificadores de certeza ('sin duda', 'obvio') |")
        md.append(f"| **D2: Forense Epistémica** | `epistemic_ratio` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['epistemic_ratio']}` | Ratio Certeza / Cautela |")
        # D3
        md.append(f"| **D3: Riqueza Léxica** | `guiraud_ttr` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['guiraud_ttr']}` | Índice Guiraud de riqueza léxica invariante |")
        md.append(f"| **D3: Riqueza Léxica** | `hapax_ratio_pct` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['hapax_ratio_pct']}%` | Porcentaje de palabras usadas una sola vez |")
        md.append(f"| **D3: Legibilidad** | `flesch_szigriszt` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['flesch_szigriszt']}` | Escala de comprensión Flesch-Szigriszt (0-100) |")
        md.append(f"| **D3: Legibilidad** | `gutierrez_polini` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['gutierrez_polini']}` | Legibilidad Gutiérrez de Polini para español |")
        # D4
        md.append(f"| **D4: Morfosintaxis spaCy** | `adv_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['adv_density']}%` | Densidad de adverbios (Predictor Top 1) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `pron_1p_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['pron_1p_density']}%` | Densidad pronombres 1.ª persona (apelo emocional) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `pron_3p_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['pron_3p_density']}%` | Densidad pronombres 3.ª persona (registro formal) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `adj_noun_ratio` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['adj_noun_ratio']}` | Ratio de adjetivación calificativa |")
        # D5
        md.append(f"| **D5: Anclajes Factuales** | `all_caps_ratio_pct` | `{vars_5d['D5_Anclajes_Factuales_Puntuacion']['all_caps_ratio_pct']}%` | Palabras completas en MAYÚSCULAS sostenidas |")
        md.append(f"| **D5: Anclajes Factuales** | `upper_chars_ratio_pct`| `{vars_5d['D5_Anclajes_Factuales_Puntuacion']['upper_chars_ratio_pct']}%` | Proporción global de letras mayúsculas |")
        md.append(f"| **D5: Anclajes Factuales** | `numbers_density` | `{vars_5d['D5_Anclajes_Factuales_Puntuacion']['numbers_density']}%` | Anclaje cuantitativo numérico |")
        md.append(f"| **D5: Anclajes Factuales** | `temporal_density` | `{vars_5d['D5_Anclajes_Factuales_Puntuacion']['temporal_density']}%` | Marcadores temporales específicos (meses/años) |")
        md.append(f"| **D5: Puntuación Enfática** | `punct_intensity` | `{vars_5d['D5_Anclajes_Factuales_Puntuacion']['punct_intensity']}%` | Signos expresivos agregados (!, ?, ...) |")

        md.append("\n## 5. Auditoría Frase a Frase (Explicabilidad Local)")
        md.append("| # | Texto de la Frase | Sensac. (%) | Coher. Consec. | Gatillo | Mayúsc. | Adverbios | Banderas y Advertencias Forenses |")
        md.append("|---|---|:---:|:---:|:---:|:---:|:---:|---|")
        for s in oraciones:
            gat_icon = "🔥 SÍ" if s["es_oracion_gatillo"] else "No"
            c_val = f"{s['coherencia_consecutiva']:.3f}" if s["coherencia_consecutiva"] is not None else "—"
            flags_str = "<br>".join(s["banderas_forenses"]) if s["banderas_forenses"] else "✅ Tono neutro / Sin anomalías"
            md.append(f"| **{s['idx']}** | *\"{s['texto']}\"* | `{s['sensacionalismo_pct']}%` | `{c_val}` | {gat_icon} | `{s['mayusculas_ratio']}%` | `{s['adverbios_densidad']}%` | {flags_str} |")

        return "\n".join(md)
