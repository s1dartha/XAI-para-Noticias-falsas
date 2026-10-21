"""
=============================================================================
MÉTODO EXPLICABLE DE AUDITORÍA FORENSE Y EXTRACCIÓN DE CARACTERÍSTICAS TEXTUALES
=============================================================================
Desarrollado para la Tesis de Maestría / Grado en Inteligencia Artificial y PLN.
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/metodo_explicabilidad.py

Este módulo implementa un marco de explicabilidad intrínseca y auditoría forense
que rechaza el reduccionismo de la clasificación binaria de caja negra ("Verdadero vs Falso").
En su lugar, disecciona cualquier noticia entrante a través de:

1. El Modelo de Sensacionalismo Documental y Oracional (BETO fine-tuned: JJNeila/bert-spanish-sensationalism-oss)
   - Probabilidad documental (P_full), media (P_mean), pico gatillo (P_max),
     top-2 oracional (P_top2), dispersión afectiva (sigma_sens),
     ratio de dilución contextual (dilution_ratio) e impacto causal de gatillos (delta_p_gatillo).
2. El Modelo Ganador de Redundancia Semántica Intra-Documental (BETO SBERT: hiiamsid/sentence_similarity_spanish_es, d=768)
   - Jerarquía validada de 4 umbrales semánticos:
     * tau_GMM = 0.3434 (Continuidad Temática Basal)
     * tau_PAWSX = 0.5340 (Paráfrasis Semántica Moderada, P90 / Benchmark humano PAWS-X)
     * tau_P95 = 0.5996 / Shannon = 0.6079 (Redundancia Semántica Estricta)
     * tau_Hiper = 0.7287 (Hiper-Redundancia Circular / Bucle Argumental, P99)
   - Estadísticos de orden y Ablación Oracional No Tautológica (MexGen):
     * Primer estadístico de orden S_(1) (max_intra_similarity) y segundo estadístico de orden S_(2) (second_max_intra_similarity),
       top-3 media, percentil 90, varianza, brecha de persistencia multi-bucle MexGen y caída media MoRF (delta_mean_sim_morf).
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
   - Identificación de oraciones gatillo, saltos temáticos, paráfrasis PAWS-X, redundancia estricta, citas y atribuciones.
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
from sentence_transformers import SentenceTransformer
import spacy

# Import de arquetipos flexibles
try:
    from codigos.arquetipos_flexibles import clasificar_arquetipo, calcular_afinidades
except Exception:
    try:
        from arquetipos_flexibles import clasificar_arquetipo, calcular_afinidades
    except Exception:
        # Definición fallback si no se encuentra el módulo
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
PAT_CENSORED_MASKS = re.compile(r'\*(?:NUMBER|URL|PHONE|\d+)\*', re.IGNORECASE)
PAT_CENSORED_NUMBER = re.compile(r'\*(?:NUMBER|\d+)\*', re.IGNORECASE)
PAT_PERCENT = re.compile(r'(?:\d+\s*%|\bpor\s+ciento\b)', re.IGNORECASE)
PAT_ALL_CAPS = re.compile(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
SIGLAS_EXCLUIDAS = {'ONU', 'UE', 'EEUU', 'OMS', 'PIB', 'FMI', 'OTAN', 'VOX', 'PSOE', 'PP', 'INE', 'USA', 'EFE', 'IAC', 'NICA', 'NUMBER', 'URL', 'PHONE'}

PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}
ADVERBIOS_ENFATICOS = {'muy', 'más', 'tan', 'bastante', 'casi', 'apenas', 'siempre', 'nunca', 'jamás', 'ya', 'totalmente', 'absolutamente', 'demasiado'}

# Jerarquía de Umbrales Semánticos Calibrados con BETO SBERT (hiiamsid/sentence_similarity_spanish_es)
TAU_CONTINUIDAD_GMM = 0.3434      # Continuidad temática basal (GMM 2 componentes)
TAU_PARAFRASIS_PAWSX = 0.5340     # Paráfrasis semántica moderada (P90 corpus / PAWS-X = 0.533)
TAU_REDUNDANCIA_ESTRICTA = 0.5996 # Redundancia semántica estricta (P95 corpus / Shannon = 0.6079)
TAU_HIPER_REDUNDANCIA = 0.7287    # Hiper-redundancia circular / Bucle argumental (P99 corpus)
TAU_REDUNDANCIA = TAU_REDUNDANCIA_ESTRICTA


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
    Sensacionalismo de BETO (JJNeila), Redundancia Semántica de BETO SBERT
    (hiiamsid/sentence_similarity_spanish_es, d=768) y Variables Enriquecidas 5D.
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

        # 3. Cargar Modelo Ganador BETO SBERT Redundancia (d=768)
        self.sbert_name = "hiiamsid/sentence_similarity_spanish_es"
        print(f"⏳ Cargando BETO SBERT Redundancia ({self.sbert_name}, d=768)...")
        self.sbert_model = SentenceTransformer(self.sbert_name, device=str(self.device))

        print("✅ [AuditorExplicable] Modelos cargados y listos para auditoría forense.\n")

    def segmentar_oraciones(self, texto: str) -> List[str]:
        """
        Segmentación oracional adaptada a noticias en español,
        respetando titulares con saltos de línea y signos de puntuación.
        """
        raw_text = str(texto).strip()
        sents = [s.strip() for s in nltk.sent_tokenize(raw_text, language='spanish') if len(s.strip()) > 8]
        if not sents:
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
        """Genera embeddings normalizados BETO SBERT (d=768) para oraciones."""
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
        p_full = self.predecir_sensacionalismo([texto_str], max_len=256)[0]
        sens_oraciones = self.predecir_sensacionalismo(oraciones, max_len=128)

        p_mean = float(np.mean(sens_oraciones))
        p_max = float(np.max(sens_oraciones))
        idx_max = int(np.argmax(sens_oraciones))
        oracion_gatillo = oraciones[idx_max]

        p_sorted = sorted(sens_oraciones, reverse=True)
        p_top2 = float(np.mean(p_sorted[:2])) if len(p_sorted) >= 2 else p_max
        sigma_sens = float(np.std(sens_oraciones)) if len(sens_oraciones) > 1 else 0.0

        dilution_ratio = float(p_full / (p_max + 1e-6))

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
        # 2. Redundancia Semántica Intra-Documental, Multi-Umbral y MexGen (BETO SBERT)
        # ---------------------------------------------------------------------
        embeddings = self.codificar_oraciones_sbert(oraciones)
        k = len(oraciones)
        sim_matrix = np.dot(embeddings, embeddings.T)  # (k, k)

        consec_sims = []
        if k >= 2:
            for i in range(k - 1):
                consec_sims.append(float(sim_matrix[i, i + 1]))

            triu_indices = np.triu_indices(k, k=1)
            pairwise_sims = sim_matrix[triu_indices]
            n_pairs = len(pairwise_sims)
            sorted_pairs = np.sort(pairwise_sims)[::-1]

            mean_intra = float(np.mean(pairwise_sims))
            max_intra = float(sorted_pairs[0])
            second_max_intra = float(sorted_pairs[1]) if n_pairs >= 2 else max_intra
            top3_mean_intra = float(np.mean(sorted_pairs[:min(3, n_pairs)]))
            p90_intra = float(np.percentile(pairwise_sims, 90))
            var_intra = float(np.var(pairwise_sims))

            # Conteos y densidades en la jerarquía de 4 umbrales
            topic_cont_count = int(np.sum(pairwise_sims >= TAU_CONTINUIDAD_GMM))
            topic_cont_density = float(topic_cont_count / max(n_pairs, 1))

            paraphrase_p90_count = int(np.sum(pairwise_sims >= TAU_PARAFRASIS_PAWSX))
            paraphrase_p90_density = float(paraphrase_p90_count / max(n_pairs, 1))

            strict_p95_count = int(np.sum(pairwise_sims >= TAU_REDUNDANCIA_ESTRICTA))
            strict_p95_density = float(strict_p95_count / max(n_pairs, 1))

            hyper_count = int(np.sum(pairwise_sims >= TAU_HIPER_REDUNDANCIA))
            hyper_density = float(hyper_count / max(n_pairs, 1))

            # Ablación Oracional No Tautológica (MexGen)
            # Identificar par maximal (i*, j*) y oración crítica s* de mayor centralidad
            max_pair_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(k) * 999.0), sim_matrix.shape)
            i_star, j_star = int(max_pair_idx[0]), int(max_pair_idx[1])
            row_means = np.array([(np.sum(sim_matrix[idx]) - sim_matrix[idx, idx]) / max(k - 1, 1) for idx in range(k)])
            s_star = i_star if row_means[i_star] >= row_means[j_star] else j_star
            s_hub = int(np.argmax(row_means))

            if k >= 3:
                keep_mask = [idx for idx in range(k) if idx != s_star]
                sub_mat = sim_matrix[np.ix_(keep_mask, keep_mask)]
                sub_triu = sub_mat[np.triu_indices(len(keep_mask), k=1)]
                max_post_ablation = float(np.max(sub_triu))
                mexgen_persistence_gap = float(max_intra - max_post_ablation)

                keep_hub = [idx for idx in range(k) if idx != s_hub]
                sub_hub = sim_matrix[np.ix_(keep_hub, keep_hub)]
                mean_post_morf = float(np.mean(sub_hub[np.triu_indices(len(keep_hub), k=1)]))
                delta_mean_sim_morf = float(mean_intra - mean_post_morf)
            else:
                max_post_ablation = second_max_intra
                mexgen_persistence_gap = 0.0
                delta_mean_sim_morf = 0.0

            consec_sim_mean = float(np.mean(consec_sims))
            consec_sim_min = float(np.min(consec_sims))
            consec_sim_std = float(np.std(consec_sims))
        else:
            consec_sims = []
            n_pairs = 0
            mean_intra = 0.3708
            max_intra = 0.5342
            second_max_intra = 0.4400
            top3_mean_intra = 0.4600
            p90_intra = 0.4700
            var_intra = 0.0
            topic_cont_count = 0
            topic_cont_density = 0.0
            paraphrase_p90_count = 0
            paraphrase_p90_density = 0.0
            strict_p95_count = 0
            strict_p95_density = 0.0
            hyper_count = 0
            hyper_density = 0.0
            max_post_ablation = 0.4400
            mexgen_persistence_gap = 0.0
            delta_mean_sim_morf = 0.0
            consec_sim_mean = 0.3800
            consec_sim_min = 0.3800
            consec_sim_std = 0.0

        # Compatibilidad de nombres con reportes y scripts downstream
        redundant_count = strict_p95_count
        redundancy_density = strict_p95_density

        # Discretizaciones de Shannon calibradas In-Fold sobre BETO SBERT
        # 4 Macro-Estratos Funcionales estables: [<=0.3740, (0.3740, 0.6079], (0.6079, 0.7287], >0.7287]
        cuts_4 = [-np.inf, 0.3740, 0.6079, 0.7287, np.inf]
        shannon_bin_4 = int(pd.cut([max_intra], bins=cuts_4, labels=[1, 2, 3, 4])[0])

        cuts_7 = [-np.inf, 0.3740, 0.4500, 0.5340, 0.6079, 0.6650, 0.7287, np.inf]
        shannon_bin_7 = int(pd.cut([max_intra], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7])[0])

        if max_intra >= TAU_REDUNDANCIA_ESTRICTA and max_post_ablation >= TAU_REDUNDANCIA_ESTRICTA:
            topologia_mexgen = "Red Multi-Bucle Persistente (S_post >= 0.60: múltiples pares redundantes)"
        elif max_intra >= TAU_REDUNDANCIA_ESTRICTA and max_post_ablation < TAU_REDUNDANCIA_ESTRICTA:
            topologia_mexgen = "Bucle Único Focalizado (S_post < 0.60: se desactiva al retirar 1 oración)"
        elif max_intra >= TAU_PARAFRASIS_PAWSX:
            topologia_mexgen = "Paráfrasis Moderada Focalizada (PAWS-X >= 0.534)"
        else:
            topologia_mexgen = "Progresión Lineal sin Bucles Redundantes (< 0.534)"

        # Encontrar el par de oraciones con máxima similitud
        top_pair_info = None
        if k >= 2:
            max_pair_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(k) * 999.0), sim_matrix.shape)
            i_p, j_p = int(max_pair_idx[0]), int(max_pair_idx[1])
            sim_val_top = float(sim_matrix[i_p, j_p])
            top_pair_info = {
                "oracion_a_idx": i_p + 1,
                "oracion_a_texto": oraciones[i_p],
                "oracion_b_idx": j_p + 1,
                "oracion_b_texto": oraciones[j_p],
                "similitud_coseno": round(sim_val_top, 4),
                "supera_continuidad_gmm_034": bool(sim_val_top >= TAU_CONTINUIDAD_GMM),
                "supera_parafrasis_pawsx_053": bool(sim_val_top >= TAU_PARAFRASIS_PAWSX),
                "supera_redundancia_estricta_060": bool(sim_val_top >= TAU_REDUNDANCIA_ESTRICTA),
                "supera_hiper_redundancia_073": bool(sim_val_top >= TAU_HIPER_REDUNDANCIA)
            }

        # ---------------------------------------------------------------------
        # 3. Las 5 Dimensiones del Índice de Manipulación Lingüística (IML)
        #    Recalibradas con los umbrales de BETO SBERT (0.3740, 0.5340, 0.6079, 0.7287)
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
            amortiguacion_raw = 1.0
        d3_amortiguacion = float(np.clip(amortiguacion_raw * 100.0, 0.0, 100.0))

        # D4: Reiteración y Bucle Argumental / Redundancia (0 - 100)
        # Calibrado en BETO SBERT:
        # - 0.3434 (Continuidad GMM) -> ~15 pts
        # - 0.5340 (PAWS-X P90) -> ~52 pts
        # - 0.5996 / 0.6079 (Redundancia Estricta P95 / Shannon) -> >= 65 pts (Umbral de Bucle)
        # - 0.7287 (Hiper-Redundancia Circular P99) -> >= 85 pts
        # Además bonifica hasta +8 pts si hay persistencia multi-bucle MexGen (second_max_intra >= 0.5340)
        base_d4 = (max_intra - 0.25) / (0.78 - 0.25) * 100.0
        multi_loop_bonus = max(0.0, (second_max_intra - 0.45) * 35.0) if k >= 3 else 0.0
        d4_redundancia = float(np.clip(base_d4 + multi_loop_bonus, 0.0, 100.0))

        # D5: Cohesión y Fluidez Discursiva (0 - 100)
        # Calibrado con los 4 Macro-Estratos In-Fold de BETO SBERT:
        # - Bin 1 (<= 0.3740): Fragmentación / Baja cohesión (<= 55 pts)
        # - Bin 2 y Bin 3 moderado (0.3740 a 0.68): Banda óptima de progresión lineal (80 a 100 pts)
        # - Bin 4 (> 0.7287): Penalización por bucle circular repetitivo
        if max_intra <= 0.3740:
            cohesion_raw = (max(0.0, max_intra) / 0.3740) * 52.0
        elif max_intra <= 0.6500:
            cohesion_raw = 85.0 + (1.0 - abs(max_intra - 0.5200) / 0.1500) * 15.0
        else:
            cohesion_raw = max(35.0, 95.0 - (max_intra - 0.6500) / 0.2500 * 60.0)
        # Penalizar si hay ruptura secuencial severa y baja similitud media
        if consec_sim_min < 0.08 and mean_intra < 0.28:
            cohesion_raw = min(cohesion_raw, 48.0)
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

        # ---------------------------------------------------------------------
        # 4. Variables Enriquecidas de las 5 Dimensiones (Ensamble Avanzado)
        # ---------------------------------------------------------------------
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
        texto_str_no_masks = PAT_CENSORED_MASKS.sub(' ', texto_str)
        caps_words_raw = PAT_ALL_CAPS.findall(texto_str_no_masks)
        caps_words = [w for w in caps_words_raw if w not in SIGLAS_EXCLUIDAS and len(w) >= 3]
        all_caps_count = len(caps_words)
        all_caps_ratio = (all_caps_count / w_count) * 100.0

        alpha_chars = [c for c in texto_str_no_masks if c.isalpha()]
        upper_chars = [c for c in alpha_chars if c.isupper()]
        upper_chars_ratio = float(len(upper_chars) / max(len(alpha_chars), 1)) * 100.0

        excl_count = texto_str_no_masks.count('!') + texto_str_no_masks.count('¡')
        quest_count = texto_str_no_masks.count('?') + texto_str_no_masks.count('¿')
        ellipsis_count = len(re.findall(r'(\.\.\.|…)', texto_str_no_masks))

        excl_density = (excl_count / w_count) * 100.0
        quest_density = (quest_count / w_count) * 100.0
        ellipsis_density = (ellipsis_count / w_count) * 100.0
        punct_intensity = ((excl_count + quest_count + ellipsis_count) / w_count) * 100.0

        numbers_list = PAT_DIGITS.findall(texto_str_no_masks) + PAT_CENSORED_NUMBER.findall(texto_str)
        numbers_density = (len(numbers_list) / w_count) * 100.0

        percent_matches = PAT_PERCENT.findall(texto_str)
        percent_count = len(percent_matches)

        temporal_matches = PAT_TEMPORAL.findall(texto_str_no_masks)
        temporal_density = (len(temporal_matches) / w_count) * 100.0

        # ---------------------------------------------------------------------
        # 5. Desglose Frase a Frase (Explicabilidad Local Oracional)
        # ---------------------------------------------------------------------
        desglose_oraciones = []
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
            s_sens = float(sens_oraciones[i])

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
            if par_mas_cercano:
                if par_mas_cercano["similitud"] >= TAU_REDUNDANCIA_ESTRICTA:
                    flags.append(f"🔁 Redundancia Estricta con Frase #{par_mas_cercano['idx_oracion_cercana']} (cos={par_mas_cercano['similitud']:.2f} >= 0.60)")
                elif par_mas_cercano["similitud"] >= TAU_PARAFRASIS_PAWSX:
                    flags.append(f"🔄 Paráfrasis Moderada (PAWS-X) con Frase #{par_mas_cercano['idx_oracion_cercana']} (cos={par_mas_cercano['similitud']:.2f} >= 0.53)")

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
                "modelo_redundancia": self.sbert_name,
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
                "second_max_intra_similarity": round(second_max_intra, 4),
                "top3_mean_intra_similarity": round(top3_mean_intra, 4),
                "p90_intra_similarity": round(p90_intra, 4),
                "var_intra_similarity": round(var_intra, 4),
                "max_post_ablation_mexgen": round(max_post_ablation, 4),
                "mexgen_persistence_gap": round(mexgen_persistence_gap, 4),
                "delta_mean_sim_morf": round(delta_mean_sim_morf, 4),
                "topologia_mexgen": topologia_mexgen,
                "topic_continuity_pairs_count": topic_cont_count,
                "topic_continuity_density_pct": round(topic_cont_density * 100.0, 2),
                "paraphrase_pairs_count_p90": paraphrase_p90_count,
                "paraphrase_density_p90_pct": round(paraphrase_p90_density * 100.0, 2),
                "strict_redundant_pairs_count_p95": strict_p95_count,
                "strict_redundancy_density_p95_pct": round(strict_p95_density * 100.0, 2),
                "hyper_redundant_pairs_count": hyper_count,
                "hyper_redundancy_density_pct": round(hyper_density * 100.0, 2),
                "redundant_pairs_count": redundant_count,
                "num_pairs_total": n_pairs,
                "redundancy_density_pct": round(redundancy_density * 100.0, 2),
                "umbral_gmm_tau": TAU_CONTINUIDAD_GMM,
                "umbral_pawsx_tau": TAU_PARAFRASIS_PAWSX,
                "umbral_estricto_tau": TAU_REDUNDANCIA_ESTRICTA,
                "umbral_hiper_tau": TAU_HIPER_REDUNDANCIA,
                "shannon_bin_7": shannon_bin_7,
                "shannon_bin_4": shannon_bin_4,
                "par_maxima_redundancia": top_pair_info,
                "matriz_similitud": [[round(float(val), 4) for val in row] for row in sim_matrix],
                "diagnostico": (
                    "Hiper-Redundancia Circular (Bucle de Verdad Ilusoria, >= 0.73)" if max_intra >= TAU_HIPER_REDUNDANCIA else (
                        "Redundancia Semántica Estricta (Reiteración Proposicional, >= 0.60)" if max_intra >= TAU_REDUNDANCIA_ESTRICTA else (
                            "Cohesión Periodística Estándar en Banda de Shannon (0.37 - 0.60)" if max_intra > 0.3740 else
                            "Discurso Desarticulado / Párrafos Fragmentados (<= 0.37)"
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
                    "interpretacion": "Circularidad semántica intra-documental (S_(1), S_(2) y paráfrasis BETO SBERT).",
                    "nivel": "Bucle Argumental / Redundancia Estricta" if d4_redundancia >= 65.0 else (
                        "Progresión Normal Temática" if d4_redundancia >= 25.0 else "Diversidad Temática / Sin Repeticiones"
                    )
                },
                "D5_Cohesion_Fluidez": {
                    "valor": round(d5_cohesion, 1),
                    "escala": "[0 - 100]",
                    "interpretacion": "Conectividad proposicional y solidez sintáctica en la banda In-Fold de Shannon.",
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

        md.append("## 3. Redundancia Semántica Intra-Documental y Ablación MexGen (BETO SBERT, $d=768$)")
        md.append(f"- **Similitud Coseno Media ($\\bar{{S}}_{{\\text{{doc}}}}$):** `{red['mean_intra_similarity']}` | **Máxima ($S_{{(1)}}$):** `{red['max_intra_similarity']}` | **Segundo Estadístico ($S_{{(2)}}$):** `{red['second_max_intra_similarity']}`")
        md.append(f"- **Jerarquía Multi-Umbral (sobre `{red['num_pairs_total']}` pares):**")
        md.append(f"  - Continuidad Temática ($\\cos \\ge 0.3434$, GMM): `{red['topic_continuity_pairs_count']}` pares (`{red['topic_continuity_density_pct']}%`)")
        md.append(f"  - Paráfrasis Moderada ($\\cos \\ge 0.5340$, PAWS-X $P_{{90}}$): `{red['paraphrase_pairs_count_p90']}` pares (`{red['paraphrase_density_p90_pct']}%`)")
        md.append(f"  - Redundancia Estricta ($\\cos \\ge 0.5996$, $P_{{95}}$ / Shannon): `{red['strict_redundant_pairs_count_p95']}` pares (`{red['strict_redundancy_density_p95_pct']}%`)")
        md.append(f"  - Hiper-Redundancia Circular ($\\cos \\ge 0.7287$, $P_{{99}}$): `{red['hyper_redundant_pairs_count']}` pares (`{red['hyper_redundancy_density_pct']}%`)")
        md.append(f"- **Ablación Oracional No Tautológica (MexGen):** Máximo Post-Ablación $S_{{\\text{{post}}}} =$ `{red['max_post_ablation_mexgen']}` ($\\Delta S_{{\\max}} =$ `{red['mexgen_persistence_gap']}`) | Caída Media MoRF $\\Delta \\bar{{S}}_{{\\text{{MoRF}}}} =$ `{red['delta_mean_sim_morf']}` $\\to$ *{red['topologia_mexgen']}*")
        md.append(f"- **Macro-Estrato de Shannon In-Fold (4 Clases):** Bin `{red['shannon_bin_4']}` (Escala 7 Bins: `{red['shannon_bin_7']}`)")
        if red["par_maxima_redundancia"]:
            pmr = red["par_maxima_redundancia"]
            md.append(f"- **Par con Máxima Coincidencia Semántica (cos = {pmr['similitud_coseno']}):**")
            md.append(f"  1. Frase #{pmr['oracion_a_idx']}: *\"{pmr['oracion_a_texto']}\"*")
            md.append(f"  2. Frase #{pmr['oracion_b_idx']}: *\"{pmr['oracion_b_texto']}\"*")
        md.append(f"- **Diagnóstico Forense:** {red['diagnostico']}\n")

        md.append("## 4. Variables Enriquecidas de las 5 Dimensiones")
        md.append("| Categoría | Variable | Valor | Significado Forense |")
        md.append("|---|---|:---:|---|")
        md.append(f"| **D0: Redundancia MexGen** | `second_max_intra_similarity` | `{red['second_max_intra_similarity']}` | Segundo estadístico de orden $S_{{(2)}}$ (persistencia multi-bucle) |")
        md.append(f"| **D0: Redundancia PAWS-X** | `paraphrase_density_p90` | `{red['paraphrase_density_p90_pct']}%` | Densidad de pares parafrásticos ($\\cos \\ge 0.5340$, PAWS-X) |")
        md.append(f"| **D0: Redundancia Estricta** | `strict_redundancy_density_p95` | `{red['strict_redundancy_density_p95_pct']}%` | Densidad de redundancia estricta ($\\cos \\ge 0.5996$, $P_{{95}}$) |")
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_mean` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_mean']}` | Coherencia media entre oraciones contiguas (BETO SBERT) |")
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_min` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_min']}` | Ruptura o salto temático más pronunciado |")
        md.append(f"| **D1: Flujo Secuencial** | `consec_sim_std` | `{vars_5d['D1_Dinamica_Discursiva']['consec_sim_std']}` | Inestabilidad en la transición de ideas |")
        md.append(f"| **D2: Forense Epistémica** | `dicendi_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['dicendi_density']}%` | Densidad de verbos de reporte y atribución a fuentes |")
        md.append(f"| **D2: Forense Epistémica** | `quotes_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['quotes_density']}` | Citas directas entrecomilladas por oración |")
        md.append(f"| **D2: Forense Epistémica** | `hedges_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['hedges_density']}%` | Atenuadores de cautela ('presunto', 'al parecer') |")
        md.append(f"| **D2: Forense Epistémica** | `boosters_density` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['boosters_density']}%` | Intensificadores de certeza ('sin duda', 'obvio') |")
        md.append(f"| **D2: Forense Epistémica** | `epistemic_ratio` | `{vars_5d['D2_Linguistica_Forense_Epistemica']['epistemic_ratio']}` | Ratio Certeza / Cautela |")
        md.append(f"| **D3: Riqueza Léxica** | `guiraud_ttr` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['guiraud_ttr']}` | Índice Guiraud de riqueza léxica invariante |")
        md.append(f"| **D3: Riqueza Léxica** | `hapax_ratio_pct` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['hapax_ratio_pct']}%` | Porcentaje de palabras usadas una sola vez |")
        md.append(f"| **D3: Legibilidad** | `flesch_szigriszt` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['flesch_szigriszt']}` | Escala de comprensión Flesch-Szigriszt (0-100) |")
        md.append(f"| **D3: Legibilidad** | `gutierrez_polini` | `{vars_5d['D3_Riqueza_Lexica_Legibilidad']['gutierrez_polini']}` | Legibilidad Gutiérrez de Polini para español |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `adv_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['adv_density']}%` | Densidad de adverbios (Predictor Top) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `pron_1p_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['pron_1p_density']}%` | Densidad pronombres 1.ª persona (apelo emocional) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `pron_3p_density` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['pron_3p_density']}%` | Densidad pronombres 3.ª persona (registro formal) |")
        md.append(f"| **D4: Morfosintaxis spaCy** | `adj_noun_ratio` | `{vars_5d['D4_Perfilado_Morfosintactico_POS']['adj_noun_ratio']}` | Ratio de adjetivación calificativa |")
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
