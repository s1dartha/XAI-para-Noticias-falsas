"""
=============================================================================
EJECUTOR DE AUDITORÍA FORENSE EXPLICABLE SOBRE 4 NOTICIAS DE EJEMPLO
=============================================================================
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/ejecutar_auditoria_ejemplos.py

Flujo Metodológico de la Tesis:
1. Generación de Explicabilidad Forense Estructurada:
   - Diagrama Visual en Mermaid y ASCII: "El Mapa Conceptual en 4 Niveles: ¿Dónde Entra Cada Cosa?".
   - 36 Variables Cuantitativas Base (D0 a D5), IML, Oración Gatillo, Redundancia Cíclica vs Progresión Lineal.
2. Auditoría Frase a Frase sobre 4 Noticias Emblemáticas de Control (2 Falsas y 2 Verdaderas).
   - Inclusión de gráficos embebidos directamente en Markdown (![...](...)).
3. Pregunta Científica de Transición: ¿Es posible clasificar desinformación a partir de estas características forenses?
4. El Modelo Ensamble Adaptativo Multidimensional (GBDT 5D) y sus Métricas sobre N=4.418 Noticias (Superando a SaBERT en LatAm).
5. Mapa de Dependencias Documentales Actualizado.
=============================================================================
"""

import os
import sys
import json
import time

# Asegurar importación del módulo local
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from metodo_explicabilidad import AuditorExplicabilidadNoticia
from generar_visualizaciones import generar_graficos_auditoria

# -----------------------------------------------------------------------------
# 4 NOTICIAS DE EJEMPLO DEL CORPUS DE LA TESIS (2 FALSAS Y 2 VERDADERAS)
# -----------------------------------------------------------------------------
NOTICIAS_BENCHMARK = [
    {
        "id": "noticia_1_falsa_salud_mayusculas",
        "tipo_referencia": "Falsa (Ground Truth = 1)",
        "titulo": "Bulo de Salud con Mayúsculas Sostenidas y Alarma Oncológica",
        "origen": "FakeDeS 2021 (Hispanoamérica / Redes Sociales)",
        "texto": (
            "Boooomm\n"
            "MUJERES VACUNADAS DE COVID ESTÁN MOSTRANDO EFECTOS SECUNDARIOS TÍPICOS DE CANCER DE MAMA\n"
            "Los médicos de Intermountain Healthcare’s Breast Care Centre de Utah, USA, anuncian nuevas pautas de mamografías para las mujeres vacunadas contra Covid-19 recientemente.\n"
            "Los médicos han observado inflamación de los ganglios linfáticos en las mamografías de detección de mujeres que se vacunaron recientemente contra COVID-19.\n"
            "\"Siempre que los vemos en una mamografía de detección normal, llamamos a esas pacientes porque puede significar cáncer de mama metastásico que viaja a los ganglios linfáticos o linfoma o leucemia\".\n"
            "“Con la vacuna Moderna están habiendo estos síntomas aproximadamente un 11% después de la primera dosis y un 16% después de la segunda dosis. Creemos que también es comparable para la vacuna Pfizer."
        )
    },
    {
        "id": "noticia_2_falsa_conspiracion_coronacirco",
        "tipo_referencia": "Falsa (Ground Truth = 1)",
        "titulo": "Desinformación Conspiracionista y Negacionista de Redes",
        "origen": "Corpus Fact-Checking Hispanoamérica (Redes / Mensajería)",
        "texto": (
            "Victoria Abril ha dejado a todo el mundo con la boca abierta con su discurso anti-plandemia a la que ha llegado a denominar coronacirco. "
            "Más claro y con más lógica no se pueden decir las cosas, se la ha entendido todo perfectamente. "
            "Entre otras cosas ha dicho: «Ya no son tesis conspiracionistas, llevamos un año de coronacirco y epidemia de miedo y la tele nos bombardea con muertos y enfermos». "
            "Antes iban metiendo miedo para decir que la única solución es la vacuna, pero la vacuna no es la solución, nos están usando como conejillos de indias. "
            "Ponemos el vídeo a continuación, no tiene desperdicio."
        )
    },
    {
        "id": "noticia_3_verdadera_puebla_salud",
        "tipo_referencia": "Verdadera (Ground Truth = 0)",
        "titulo": "Noticia Oficial de Salud Pública y Medidas Epidemiológicas",
        "origen": "Prensa Oficial del Estado de Puebla, México (Secretaría de Salud)",
        "texto": (
            "El Gobierno de Puebla anunció que el confinamiento para evitar que colapse el sistema de salud por la pandemia de coronavirus, se extiende hasta el día 25 de enero. "
            "En rueda de prensa, el gobernador Luis Miguel Barbosa indicó que se busca reducir la curva de contagios, pues el estado se mantiene en semáforo naranja con tendencia al alza. "
            "Indicó que de esta forma se ratifica el llamado de alerta máxima en Puebla hasta el 25 de enero, fecha en que se determinará el comportamiento de la pandemia. "
            "En cuanto a la industria considerada como esencial el aforo permitido es de 30 por ciento con horarios escalonados. "
            "Asimismo se advirtió mayor presencia policíaca en las calles para el cumplimiento de las medidas y exhortó a la prudencia ciudadana. "
            "Jesús Ramírez, subsecretario de transparencia, indicó que todo el estado se encuentra en color naranja con tendencia ascendente. "
            "En la capital y zona metropolitana es rojo. "
            "El secretario de salud, José Antonio Martínez, indicó que este es el segundo día con más casos registrados de coronavirus con 353 casos en 24 horas y 37 defunciones en 72 horas. "
            "De mantenerse la tendencia, al día 14 de enero estaríamos \"a tope\" y el día 18 la capacidad hospitalaria se vería rebasada, por lo que urgió a acatar los decretos."
        )
    },
    {
        "id": "noticia_4_verdadera_ciencia_efe",
        "tipo_referencia": "Verdadera (Ground Truth = 0)",
        "titulo": "Crónica Científica sobre el Megaproyecto NICA de Dubná",
        "origen": "Agencia EFE (Sección Ciencia y Tecnología / Rusia)",
        "texto": (
            "Rusia quiere recrear el comienzo del Universo. Redacción DUBNÁ EFE En Dubná, a unos 100 kilómetros al norte de Moscú, se comienza a vislumbrar lo que será un enorme acelerador de partículas destinado a recrear los primeros instantes del Universo tras el Big Bang. "
            "La construcción del NICA (Nuclotron based Ion Collider Facility)en el Instituto Conjunto para la Investigación Nuclear (JINR, por sus siglas en inglés) de Dubná avanza a pasos agigantados. "
            "El deseo de los aproximadamente mil científicos e ingenieros que trabajan en el megaproyecto es poner en marcha el colisionador en 2022. "
            "El objetivo es estudiar la transición de la materia ordinaria al plasma quark-gluón que existía en los primeros microsegundos después de la gran explosión. "
            "Para ello en Dubná se harán colisionar haces de iones de oro."
        )
    }
]

def main():
    print("=" * 80)
    print(" 🔬 INICIANDO AUDITORÍA FORENSE EXPLICABLE SOBRE 4 NOTICIAS DE CONTROL")
    print("=" * 80)

    # 1. Instanciar Auditor
    auditor = AuditorExplicabilidadNoticia()

    # Carpetas de destino
    dir_reportes = os.path.join(current_dir, "reportes")
    dir_graficos = os.path.join(current_dir, "graficos")
    dir_codigos = os.path.join(current_dir, "codigos")
    os.makedirs(dir_reportes, exist_ok=True)
    os.makedirs(dir_graficos, exist_ok=True)
    os.makedirs(dir_codigos, exist_ok=True)

    resultados_auditorias = []

    # 2. Auditar cada noticia
    for i, item in enumerate(NOTICIAS_BENCHMARK):
        print(f"\n🔍 [Noticia {i+1}/4] Auditando: {item['titulo']}")
        print(f"   Tipo de Referencia: {item['tipo_referencia']} | Longitud: {len(item['texto'].split())} palabras")

        resultado = auditor.auditar_noticia(
            texto=item["texto"],
            titulo=item["titulo"],
            metadata={
                "id": item["id"],
                "tipo_referencia": item["tipo_referencia"],
                "origen": item["origen"]
            }
        )
        resultados_auditorias.append(resultado)

        # Guardar JSON individual
        filename_json = f"{item['id']}.json"
        path_json = os.path.join(dir_reportes, filename_json)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"   💾 Guardado JSON individual: {path_json}")

    # 3. Guardar JSON Consolidado
    path_consolidado = os.path.join(dir_reportes, "auditoria_completa_4_noticias.json")
    with open(path_consolidado, "w", encoding="utf-8") as f:
        json.dump(resultados_auditorias, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Guardado JSON Maestro Consolidado: {path_consolidado}")

    # 4. Generar Figuras y Gráficos Científicos
    print("\n📈 Generando figuras forenses en alta resolución...")
    rutas_graficos = generar_graficos_auditoria(resultados_auditorias, dir_graficos)

    # 5. Generar Gran Informe Forense en Markdown
    path_md = os.path.join(dir_reportes, "Reporte_Explicabilidad_Forense_4_Noticias.md")
    print(f"\n📝 Compilando Gran Informe Forense Markdown: {path_md}")

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("# Informe de Auditoría Forense y Explicabilidad Lingüística Multi-Nivel\n\n")
        f.write("## De la Caracterización Forense Estilométrica al Ensamble Adaptativo de Detección de Desinformación\n\n")
        f.write("---\n\n")
        f.write("**Fecha:** Octubre de 2026  \n")
        f.write("**Módulo Principal:** [`Tesis/Explicabilidad_Propia`](../)  \n")
        f.write("**Modelos Base de Extracción:** BETO Sensacionalismo (`JJNeila/bert-spanish-sensationalism-oss`), SBERT Redundancia (`paraphrase-multilingual-MiniLM-L12-v2`) y spaCy NLP (`es_core_news_sm`)  \n")
        f.write("**Corpus de Validación Forense:** 4 Noticias Emblemáticas de Control (2 Falsas y 2 Verdaderas)  \n")
        f.write("**Corpus de Validación Experimental del Ensamble:** $N = 4.418$ Noticias en Español (España y América Latina)  \n\n")

        # ---------------------------------------------------------------------
        # SECCIÓN 1: EL MAPA CONCEPTUAL EN 4 NIVELES
        # ---------------------------------------------------------------------
        f.write("## 1. El Mapa Conceptual en 4 Niveles: ¿Dónde Entra Cada Cosa?\n\n")
        f.write(
            "Para estructurar con total claridad el flujo metrológico y evitar cualquier confusión entre modelos neuronales, "
            "características numéricas de Machine Learning y perfiles cualitativos forenses, toda la metodología de la tesis se organiza en **4 niveles conceptuales estrictos**:\n\n"
        )

        f.write("```mermaid\n")
        f.write("flowchart TD\n")
        f.write("    subgraph N1[\"NIVEL 1: Modelos Base de Procesamiento del Lenguaje Natural (PLN)\"]\n")
        f.write("        BETO[\"BETO Sensacionalismo Afectivo<br/>(JJNeila/bert-spanish-sensationalism-oss)\"]\n")
        f.write("        SBERT[\"Sentence-BERT Semántica Vectorial<br/>(paraphrase-multilingual-MiniLM-L12-v2)\"]\n")
        f.write("        SPACY[\"spaCy POS Tagging + Regex Compilados<br/>(es_core_news_sm + Minería Léxica)\"]\n")
        f.write("    end\n\n")
        f.write("    subgraph N2[\"NIVEL 2: Las 36 Variables Cuantitativas Base (Dimensiones D0 a D5)\"]\n")
        f.write("        D0[\"D0: Sensacionalismo y Redundancia (10 vars)<br/>P_full, P_max, sigma_sens, dilution_ratio, delta_p, sim_cos\"]\n")
        f.write("        D1[\"D1: Dinámica Discursiva y Flujo Secuencial (3 vars)<br/>consec_sim_mean, consec_sim_min, consec_sim_std\"]\n")
        f.write("        D2[\"D2: Lingüística Forense y Epistémica (5 vars)<br/>dicendi_density, quotes_density, hedges, boosters, epistemic_ratio\"]\n")
        f.write("        D3[\"D3: Riqueza Léxica, Sintaxis y Legibilidad (6 vars)<br/>guiraud_ttr, hapax_ratio, flesch_szigriszt, gutierrez_polini\"]\n")
        f.write("        D4[\"D4: Morfosintaxis spaCy POS Tagging (4 vars)<br/>adv_density (Top 1 Predictor), pron_1p, pron_3p, adj_noun_ratio\"]\n")
        f.write("        D5[\"D5: Anclajes Factuales y Mayúsculas Sostenidas (8 vars)<br/>upper_chars_ratio (Top 2), all_caps_ratio (Top 3), numbers, temporal\"]\n")
        f.write("    end\n\n")
        f.write("    subgraph N3[\"NIVEL 3: El Perfilador Estilométrico (Las 5 Dimensiones del IML: Escala 0 a 100)\"]\n")
        f.write("        IML1[\"D1: Carga Emocional / Sensacionalismo (0 - 100)\"]\n")
        f.write("        IML2[\"D2: Volatilidad de Gatillo / Clickbait (0 - 100)\"]\n")
        f.write("        IML3[\"D3: Amortiguación Contextual / Resiliencia (0 - 100)\"]\n")
        f.write("        IML4[\"D4: Reiteración y Bucle Argumental (0 - 100)\"]\n")
        f.write("        IML5[\"D5: Cohesión y Fluidez Discursiva (0 - 100)\"]\n")
        f.write("        IML_SCORE[\"Score Global IML (0 - 100) & Huella en Radar\"]\n")
        f.write("    end\n\n")
        f.write("    subgraph N4[\"NIVEL 4: Los 5 Arquetipos de Estilo Discursivo\"]\n")
        f.write("        E1[\"Estilo I: Desinformación Estridente<br/>(Grita + Repite: D1 >= 65 y D4 >= 65)\"]\n")
        f.write("        E2[\"Estilo II: Cebo Comercial / Clickbait<br/>(Titular grita, cuerpo informa: D2 >= 50 y D3 >= 50)\"]\n")
        f.write("        E3[\"Estilo III: Propaganda Institucional / Astroturfing<br/>(Sobrio pero con bucle circular: D1 < 45 y D4 >= 65)\"]\n")
        f.write("        E4[\"Estilo IV: Incoherencia Estructural / Roto<br/>(Sintaxis fracturada: D5 <= 55 y D4 < 40)\"]\n")
        f.write("        E5[\"Estilo V: Periodismo Profesional Balanceado<br/>(Sobrio con progresión temática lineal óptima)\"]\n")
        f.write("    end\n\n")
        f.write("    N1 --> N2\n")
        f.write("    N2 --> N3\n")
        f.write("    N3 --> N4\n")
        f.write("```\n\n")

        f.write("```\n")
        f.write("RESUMEN TEXTUAL DEL MAPA JERÁRQUICO EN 4 NIVELES:\n")
        f.write("├── NIVEL 1: Modelos de PLN Subyacentes (BETO, SBERT, spaCy, NLTK) ── [Sensores de Lenguaje]\n")
        f.write("├── NIVEL 2: 36 Variables Cuantitativas Base (Dimensiones D0 a D5) ──── [Datos Matemáticos Crudos]\n")
        f.write("├── NIVEL 3: Perfilador Estilométrico (5 Dimensiones IML: Escala 0-100) [Ficha Forense Humana]\n")
        f.write("└── NIVEL 4: Los 5 Arquetipos Discursivos (Estilos I a V) ─────────────── [Diagnóstico Retórico]\n")
        f.write("```\n\n")

        # ---------------------------------------------------------------------
        # SECCIÓN 2: FUNDAMENTO EPISTEMOLÓGICO
        # ---------------------------------------------------------------------
        f.write("## 2. Fundamento Epistemológico: ¿Por qué Explicabilidad Forense Estilométrica Primero?\n\n")
        f.write(
            "La investigación tradicional en detección de desinformación comete una falacia metodológica fundacional: "
            "forzar a una red neuronal de caja negra a emitir un veredicto binario omnisciente ($0 = \\text{Verdadero}, 1 = \\text{Falso}$) "
            "a partir de una simple secuencia de texto plano:\n\n"
            "$$\\text{Texto} \\xrightarrow{\\text{Caja Negra}} \\{0: \\text{Verdadero}, \\ 1: \\text{Falso}\\}$$\n\n"
            "Como se demostró en el Capítulo 1 del Plan Maestro de Tesis ([`Plan_tesis.md`](../../../Plan_tesis.md)), esta formulación es insostenible por dos razones ontológicas:\n"
            "1. **La Verdad Factual es Extrínseca al Texto:** Que un evento haya ocurrido depende del mundo físico y de los hechos empíricos, no de la disposición léxica o sintáctica de las palabras.\n"
            "2. **Las Dos Paradojas que Destruyen los Modelos Binarios:**\n"
            "   - **Paradoja 1 (Propaganda Sobria / Astroturfing):** Campañas estatales o desinformación deliberada redactadas en un tono frío, formal, institucional, sin insultos ni mayúsculas. Un modelo binario superficial las clasifica ingenuamente como verdaderas.\n"
            "   - **Paradoja 2 (Periodismo de Choque Legítimo):** Crónicas periodísticas 100% verídicas sobre desastres naturales, pandemias o denuncias judiciales que usan titulares alarmistas, signos exclamativos y vocabulario dramático. Un modelo binario las censura como falsas.\n\n"
            "Por esta razón, **el punto de partida de esta tesis no fue construir un clasificador binario ciego**, sino diseñar un **Método de Explicabilidad Forense Estilométrica** capaz de radiografiar con precisión matemática qué artificios de manipulación retórica, sesgos de encuadre (*framing*) y anomalías estilísticas contiene cualquier texto noticioso.\n\n"
        )

        # ---------------------------------------------------------------------
        # SECCIÓN 3: ARQUITECTURA DE LAS 36 VARIABLES Y PIPELINE DE EXTRACCIÓN
        # ---------------------------------------------------------------------
        f.write("## 3. Arquitectura Forense: Desglose de las 36 Variables y Pipeline de Extracción\n\n")
        f.write(
            "El sistema procesa y extrae un total de **36 variables cuantitativas base** distribuidas en 6 bloques temáticos (D0 a D5), "
            "sintetiza **5 dimensiones de manipulación estilométrica (IML)** y calcula **14 banderas forenses locales por cada oración**:\n\n"
            "### A. Desglose de las 36 Variables Cuantitativas Base (Dimensiones D0 a D5)\n"
            "1. **D0: Sensacionalismo y Redundancia Base (10 variables):** `P_full`, `P_mean`, `P_max`, `P_top2`, `sigma_sens`, `dilution_ratio`, `delta_p_gatillo`, `max_intra_similarity_clean`, `mean_intra_similarity_clean`, `redundancy_density`.\n"
            "2. **D1: Coherencia y Flujo Secuencial (3 variables):** `consec_sim_mean`, `consec_sim_min`, `consec_sim_std`.\n"
            "3. **D2: Lingüística Forense y Marcadores Epistémicos (5 variables):** `dicendi_density`, `quotes_density`, `hedges_density`, `boosters_density`, `epistemic_ratio`.\n"
            "4. **D3: Riqueza Léxica, Complejidad Sintáctica y Legibilidad (6 variables):** `hapax_ratio`, `guiraud_ttr`, `conteo_palabras_text`, `num_sentences`, `gutierrez_polini`, `flesch_szigriszt`.\n"
            "5. **D4: Morfosintaxis spaCy POS Tagging (4 variables):** `adv_density` (Top 1 Predictor), `pron_1p_density`, `adj_noun_ratio`, `pron_3p_density`.\n"
            "6. **D5: Anclajes Factuales, Mayúsculas y Puntuación (8 variables):** `upper_chars_ratio` (Top 2), `all_caps_ratio` (Top 3), `numbers_density`, `temporal_density`, `punct_intensity`, `excl_density`, `percent_count`, `all_caps_count` (más `ellipsis_density` y `quest_density`).\n\n"
            "### B. Pipeline de Extracción Paso a Paso (¿Cómo se Extrae de la Noticia?)\n"
            "Cuando entra el texto en bruto de una noticia, el pipeline opera en 5 etapas secuenciales:\n"
            "1. **Segmentación y Normalización:** NLTK (`sent_tokenize` adaptado a español) divide el documento en oraciones $S = \\{s_1, \\dots, s_K\\}$. Expresiones regulares limpian los tokens, detectan mayúsculas y computan núcleos silábicos según diptongos del español.\n"
            "2. **Inferencia Afectiva con BETO (`JJNeila/bert-spanish-sensationalism-oss`):** Se evalúa la probabilidad de sensacionalismo del documento completo ($P_{full}$) y de cada oración ($P(s_i)$). Se identifica la oración gatillo ($P_{max}$), se calcula la dispersión emocional $\\sigma_{sens}$ y se ejecuta la ablación causal contrafáctica (silenciar las dos frases extremas) para obtener $\\Delta P_{gatillo}$.\n"
            "3. **Inferencia Semántica con Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`):** Cada oración se proyecta en un vector denso de 384 dimensiones. Se calcula la coherencia consecutiva adyacente ($e_i \\cdot e_{i+1}$) y la matriz completa de similitud coseno $\\mathbf{S} \\in \\mathbb{R}^{K \\times K}$. Se aplica el umbral bayesiano calibrado con Modelos de Mezcla Gaussiana ($\\tau = 0.34$) para computar la densidad de redundancia y los bines de entropía de Shannon (7 y 4).\n"
            "4. **Perfilado Morfosintáctico con spaCy (`es_core_news_sm`):** POS Tagging cataloga adjetivos (ADJ), sustantivos (NOUN), adverbios (ADV) y pronombres (1.ª y 3.ª persona), normalizando densidades por 100 palabras.\n"
            "5. **Minería de Marcadores Lingüísticos (Regex Compilados):** Escaneo de verbos de reporte (*dicendi*), intensificadores de certeza (*boosters*), atenuadores de cautela (*hedges*), comillas tipográficas, números, porcentajes y fechas verificables.\n"
            "6. **Síntesis en el Perfilador Estilométrico (IML):** Conversión a escala $[0, 100]$ para $D_1, D_2, D_3, D_4, D_5$, Score IML ponderado y diagnóstico en uno de los 5 Arquetipos Discursivos.\n\n"
        )

        # ---------------------------------------------------------------------
        # SECCIÓN 4: INTERACCIÓN CRUZADA SENSACIONALISMO VS REDUNDANCIA
        # ---------------------------------------------------------------------
        f.write("## 4. Interacción Cruzada: Sensacionalismo (BETO) frente a Redundancia Semántica (SBERT)\n\n")
        f.write(
            "Una de las fuentes habituales de confusión en el análisis de desinformación es mezclar dos fenómenos lingüísticos que operan en dimensiones ortogonales e independientes:\n\n"
            "| Eje de Análisis | Modelo de PLN | ¿Qué Mide en el Texto? | Casos Típicos Posibles |\n"
            "|---|---|---|---|\n"
            "| **Sensacionalismo** | **BETO** (`bert-spanish-sensationalism-oss`) | **La intensidad afectiva y visceral:** si el texto vocifera, alarma, manipula mediante superlativos o induce pánico. | • **Alarma Sostenida:** $P_{full} > 0.70$ y todas las frases son alarmistas.<br>• **Cebo Aislado / Gatillo:** Titular alarmista ($P_{max} > 0.80$) pero cuerpo sobrio.<br>• **Sobriedad Informativa:** Texto neutro ($P_{full} < 0.25$). |\n"
            "| **Redundancia Semántica** | **Sentence-BERT** (`paraphrase-multilingual-MiniLM-L12-v2`) | **La topología y geometría de ideas:** si las oraciones aportan datos nuevos o si giran en círculos repitiendo lo mismo. | • **Redundancia Cíclica (Bucle):** $S_{max} \\ge 0.80$ (Shannon bins 5-7). Paráfrasis repetitiva.<br>• **Progresión Lineal Óptima:** $S_{max} \\in [0.59, 0.80]$ (Shannon bins 2-4). Cada frase aporta hechos nuevos.<br>• **Incoherencia / Dispersión:** $S_{max} < 0.50$ (Shannon bin 1). Párrafos desconectados. |\n\n"
            "### A. ¿Qué es la Redundancia Cíclica frente a la Redundancia No Cíclica (Progresión Lineal)?\n"
            "- **Redundancia Cíclica (Bucle Argumental / Hiper-Redundancia):** Ocurre cuando un artículo reitera la misma premisa central a lo largo de varias oraciones, disfrazándola con sinónimos. "
            "En la psicología cognitiva de la desinformación, este mecanismo se conoce como **Sesgo de Verdad Ilusoria** (*Illusory Truth Effect*): repetir una mentira múltiples veces hace que el cerebro humano la procese con mayor fluidez cognitiva y tienda a aceptarla como verdadera.\n"
            "- **Redundancia No Cíclica (Progresión Temática Lineal):** Es el estándar del periodismo profesional de calidad. Las oraciones mantienen un hilo conductor coherente "
            "(se sitúan en la **Banda Óptima de Shannon**, $0.618 \\le S_{max} \\le 0.808$), pero **cada oración añade información nueva**: un dato cuantitativo, una cita de un testigo, un antecedente histórico o una declaración oficial.\n\n"
            "### B. El Cruce de Sensacionalismo y Redundancia en los 5 Arquetipos Discursivos\n"
            "Al cruzar la carga de sensacionalismo con el tipo de redundancia, emergen con total claridad los **5 Arquetipos Discursivos** formalizados en la tesis:\n\n"
        )

        f.write("```mermaid\n")
        f.write("flowchart TD\n")
        f.write("    SENS[\"Sensacionalismo Afectivo (BETO)\"]\n")
        f.write("    RED[\"Redundancia Semántica (SBERT)\"]\n\n")
        f.write("    SENS_ALTO[\"Alto Sensacionalismo (P_full > 0.70)\"]\n")
        f.write("    SENS_BAJO[\"Bajo Sensacionalismo (Sobrio P_full < 0.35)\"]\n\n")
        f.write("    RED_CICLICA[\"Bucle Cíclico (cos >= 0.80)<br/>Sesgo de Verdad Ilusoria\"]\n")
        f.write("    RED_LINEAL[\"Progresión Lineal Óptima (cos 0.60-0.80)<br/>Banda de Shannon\"]\n")
        f.write("    RED_ROTA[\"Incoherencia Rota (cos < 0.50)<br/>Desconexión Proposicional\"]\n\n")
        f.write("    SENS --> SENS_ALTO\n")
        f.write("    SENS --> SENS_BAJO\n\n")
        f.write("    SENS_ALTO -->|Bucle Cíclico| ESTILO1[\"Estilo I: Desinformación Estridente<br/>(Grita + Repite Consigna)\"]\n")
        f.write("    SENS_ALTO -->|Progresión Lineal| ESTILO2[\"Estilo II: Cebo Comercial (Clickbait)<br/>(Titular Grita, Cuerpo Informa)\"]\n\n")
        f.write("    SENS_BAJO -->|Bucle Cíclico| ESTILO3[\"Estilo III: Propaganda Institucional<br/>(Tono Frío + Bucle Calculado)\"]\n")
        f.write("    SENS_BAJO -->|Progresión Lineal| ESTILO5[\"Estilo V: Periodismo Profesional<br/>(Sobriedad + Datos Nuevos)\"]\n\n")
        f.write("    RED --> RED_ROTA --> ESTILO4[\"Estilo IV: Incoherencia Estructural<br/>(Sintaxis Fragmentada)\"]\n")
        f.write("```\n\n")

        f.write("```\n")
        f.write("                              MATRIZ DE CRUCE: SENSACIONALISMO × REDUNDANCIA\n")
        f.write("                                                    │\n")
        f.write("                        ┌───────────────────────────┴───────────────────────────┐\n")
        f.write("                        ▼                                                       ▼\n")
        f.write("            [ALTO SENSACIONALISMO]                                  [BAJO SENSACIONALISMO (SOBRIO)]\n")
        f.write("                        │                                                       │\n")
        f.write("         ┌──────────────┴──────────────┐                         ┌──────────────┴──────────────┐\n")
        f.write("         ▼                             ▼                         ▼                             ▼\n")
        f.write("  + Bucle Cíclico              + Progresión Lineal        + Bucle Cíclico              + Progresión Lineal\n")
        f.write("         │                             │                         │                             │\n")
        f.write("         ▼                             ▼                         ▼                             ▼\n")
        f.write("┌─────────────────────┐       ┌─────────────────────┐   ┌─────────────────────┐       ┌─────────────────────┐\n")
        f.write("│      ESTILO I       │       │      ESTILO II      │   │     ESTILO III      │       │      ESTILO V       │\n")
        f.write("│   Desinformación    │       │    Cebo Comercial   │   │     Propaganda      │       │     Periodismo      │\n")
        f.write("│     Estridente      │       │      (Clickbait)    │   │     Institucional   │       │     Profesional     │\n")
        f.write("│(Grita + Repite)     │       │(Titular grita, pero │   │(Tono formal, pero   │       │(Sobrio + Progresión │\n")
        f.write("│                     │       │ cuerpo informa)     │   │ repite consigna)    │       │ informativa nueva)  │\n")
        f.write("└─────────────────────┘       └─────────────────────┘   └─────────────────────┘       └─────────────────────┘\n")
        f.write("                                                  │\n")
        f.write("                                                  ▼\n")
        f.write("                                       ┌─────────────────────┐\n")
        f.write("                                       │      ESTILO IV      │\n")
        f.write("                                       │    Incoherencia     │\n")
        f.write("                                       │    Estructural      │\n")
        f.write("                                       │(Sintaxis rota, sin  │\n")
        f.write("                                       │ cohesión temática)  │\n")
        f.write("                                       └─────────────────────┘\n")
        f.write("```\n\n")

        f.write(
            "### C. Aclaración Fundamental: ¿Por qué en la Tabla Comparativa los Estilos no Aparecen en Orden I a V?\n"
            "En el informe de investigación teórico ([`Reporte_Estilos_Manipulacion_Textual.md`](Reporte_Estilos_Manipulacion_Textual.md)), "
            "los 5 Arquetipos se definen de forma abstracta en orden conceptual (del Estilo I al Estilo V). Sin embargo, en la **Matriz Comparativa de las 4 Noticias**, "
            "las columnas corresponden a **noticias individuales concretas** (Noticia 1, Noticia 2, Noticia 3, Noticia 4). Cada una de ellas es clasificada de forma independiente según sus métricas:\n"
            "- **Noticia 1 (Bulo COVID) $\\to$ Estilo V:** ¿Por qué no fue Estilo I? Porque el Estilo I exige simultáneamente $D_1 \\ge 65$ (Carga Emocional) **Y** $D_4 \\ge 65$ (Hiper-Redundancia). La Noticia 1 tiene una carga emocional altísima ($D_1 = 97.2$), pero su redundancia fue moderada ($D_4 = 54.2 < 65$). Al no entrar en Estilo I, II, III o IV, el clasificador le asigna Estilo V por exclusión (árbol de decisión estricto). Esto revela un hallazgo empírico sobre la necesidad de calibrar reglas de frontera para textos de alta emoción sin bucle.\n"
            "- **Noticia 2 (Negacionismo) $\\to$ Estilo IV:** Presenta sintaxis fracturada y párrafos inconexos ($D_5 = 41.0 \\le 55$ y $D_4 = 11.6 < 40$).\n"
            "- **Noticia 3 (Oficial Puebla) $\\to$ Estilo II:** El semáforo de alerta genera un pico aislado ($P_{max} = 0.537$), pero el cuerpo técnico formal amortigua la alarma en un 99.4% ($D_2 \\ge 50 \\land D_3 \\ge 50$).\n"
            "- **Noticia 4 (Ciencia EFE) $\\to$ Estilo II:** El titular sobre el Big Bang dispara $P_{max} = 0.968$, pero el cuerpo técnico riguroso lo amortigua en un 78.8% ($D_2 \\ge 50 \\land D_3 \\ge 50$).\n\n"
        )

        # ---------------------------------------------------------------------
        # SECCIÓN 5: GATILLO Y ABLACIÓN CAUSAL
        # ---------------------------------------------------------------------
        f.write("## 5. El Mecanismo de la Oración Gatillo y la Prueba de Causalidad Contrafáctica\n\n")
        f.write(
            "Uno de los aportes más novedosos del método es la cuantificación rigurosa del **efecto gatillo** (*trigger effect*):\n"
            "- **¿Qué es la Oración Gatillo ($P_{max}$)?** Es la frase individual dentro del artículo que concentra de forma desmedida la mayor carga dramática, visceral o conspirativa: $P_{max} = \\max_{s \\in S} P(s)$. En la desinformación viral de redes, se sitúa estratégicamente en el titular o la primera línea para capturar la atención inmediata del lector.\n"
            "- **Volatilidad de Gatillo ($D_2$):** Mide la asimetría emocional: $\\text{volatilidad} = 1.5 \\times \\max(0, P_{max} - P_{mean}) \\times 100$. Un valor alto indica que el dramatismo no es representativo de todo el texto, sino un cebo señuelo aislado.\n"
            "- **Dilution Ratio ($DR$):** Evalúa si el cuerpo documental sostiene o diluye la alarma del titular: $DR = \\frac{P_{full}}{P_{max} + 10^{-6}}$. Si $DR \\approx 1.0$, el sensacionalismo domina homogéneamente todo el texto; si $DR \\ll 1.0$, el cuerpo mitiga la alarma inicial.\n"
            "- **Ablación Causal Contrafáctica ($\\Delta P_{gatillo}$):** Es una prueba contrafáctica de XAI: responde a la pregunta *¿cuánto cae el sensacionalismo del documento si silenciamos las dos oraciones gatillo?*\n"
            "  $$\\Delta P_{gatillo} = P(D) - P(D \\setminus \\{s_{top1}, s_{top2}\\})$$\n"
            "  - Un $\\Delta P_{gatillo}$ elevado (> 0.30, como el 0.933 registrado en el Bulo COVID) demuestra que la noticia completa fue inflada artificialmente por dos frases señuelo.\n"
            "  - Un $\\Delta P_{gatillo} \\approx 0.0$ señala un texto de tono uniforme (sea enteramente panfletario o enteramente sobrio).\n\n"
        )

        # ---------------------------------------------------------------------
        # SECCIÓN 6: CASOS DE ESTUDIO EMPÍRICOS (AUDITORÍA 4 NOTICIAS)
        # ---------------------------------------------------------------------
        f.write("## 6. Casos de Estudio Empíricos: Auditoría Forense de las 4 Noticias de Control\n\n")
        f.write("### Figuras Forenses Científicas en Alta Resolución\n\n")
        f.write("A continuación se presentan las 4 radiografías forenses generadas por el sistema en alta resolución:\n\n")
        
        # EMBEBIDO VISUAL DIRECTO DE LAS FIGURAS
        f.write("#### 1. Huella Estilométrica en las 5 Dimensiones del IML (Radar Pentagonal)\n")
        f.write("![Huella Estilométrica en las 5 Dimensiones IML - Radar Pentagonal](../graficos/fig_radar_iml_4_noticias.png)\n\n")
        
        f.write("#### 2. Trayectoria Oracional de Carga Sensacionalista y Detección de Gatillos\n")
        f.write("![Trayectoria Oracional de Carga Sensacionalista y Gatillos](../graficos/fig_trayectoria_sensacionalismo.png)\n\n")
        
        f.write("#### 3. Mapas de Calor de Redundancia Semántica Intra-Documental (SBERT)\n")
        f.write("![Mapas de Calor de Redundancia Intra-Documental SBERT](../graficos/fig_mapas_calor_redundancia.png)\n\n")
        
        f.write("#### 4. Radiografía Comparativa de Variables Clave del Perfilado Forense\n")
        f.write("![Radiografía Comparativa de Variables Clave](../graficos/fig_comparativa_variables_clave.png)\n\n")

        f.write("---\n\n")
        f.write("### 6.1 Matriz de Síntesis Forense: Las 5 Dimensiones del IML y Arquetipos (Escala 0 - 100)\n\n")

        def get_dim(idx, d):
            return f"{resultados_auditorias[idx]['las_5_dimensiones_iml'][d]['valor']} ({resultados_auditorias[idx]['las_5_dimensiones_iml'][d]['nivel'].split()[0]})"

        f.write(
            "| Dimensión del IML | Noticia 1 (Falsa Salud) | Noticia 2 (Falsa Conspir.) | Noticia 3 (Real Puebla) | Noticia 4 (Real Ciencia EFE) | Interpretación Forense |\n"
            "|---|:---:|:---:|:---:|:---:|---|\n"
            f"| **D1: Carga Emocional** | `{get_dim(0, 'D1_Carga_Emocional')}` | `{get_dim(1, 'D1_Carga_Emocional')}` | `{get_dim(2, 'D1_Carga_Emocional')}` | `{get_dim(3, 'D1_Carga_Emocional')}` | Intensidad global de adjetivos, drama y superlativos ($P_{{full}} \\times 100$) |\n"
            f"| **D2: Volatilidad de Gatillo** | `{get_dim(0, 'D2_Volatilidad_Gatillo')}` | `{get_dim(1, 'D2_Volatilidad_Gatillo')}` | `{get_dim(2, 'D2_Volatilidad_Gatillo')}` | `{get_dim(3, 'D2_Volatilidad_Gatillo')}` | Desbalance entre la oración gatillo y el tono promedio del cuerpo |\n"
            f"| **D3: Amortiguación Contextual** | `{get_dim(0, 'D3_Amortiguacion_Contextual')}` | `{get_dim(1, 'D3_Amortiguacion_Contextual')}` | `{get_dim(2, 'D3_Amortiguacion_Contextual')}` | `{get_dim(3, 'D3_Amortiguacion_Contextual')}` | Capacidad del cuerpo para diluir el impacto inicial mediante datos técnicos |\n"
            f"| **D4: Reiteración / Redundancia** | `{get_dim(0, 'D4_Reiteracion_Redundancia')}` | `{get_dim(1, 'D4_Reiteracion_Redundancia')}` | `{get_dim(2, 'D4_Reiteracion_Redundancia')}` | `{get_dim(3, 'D4_Reiteracion_Redundancia')}` | Grado de circularidad semántica y bucle argumental ($S_{{max}}$ SBERT) |\n"
            f"| **D5: Cohesión y Fluidez** | `{get_dim(0, 'D5_Cohesion_Fluidez')}` | `{get_dim(1, 'D5_Cohesion_Fluidez')}` | `{get_dim(2, 'D5_Cohesion_Fluidez')}` | `{get_dim(3, 'D5_Cohesion_Fluidez')}` | Continuidad temática natural en la Banda Óptima de Shannon |\n"
            f"| **Score Global IML (/100)** | **`{resultados_auditorias[0]['las_5_dimensiones_iml']['score_global_iml']}`** | **`{resultados_auditorias[1]['las_5_dimensiones_iml']['score_global_iml']}`** | **`{resultados_auditorias[2]['las_5_dimensiones_iml']['score_global_iml']}`** | **`{resultados_auditorias[3]['las_5_dimensiones_iml']['score_global_iml']}`** | Índice ponderado: $\\le 35$ Neutro, $35-60$ Comercial, $>60$ Anomalía Severa |\n"
            f"| **Arquetipo Asignado** | **{resultados_auditorias[0]['las_5_dimensiones_iml']['arquetipo_discursivo']['nombre'].split(':')[0]}** | **{resultados_auditorias[1]['las_5_dimensiones_iml']['arquetipo_discursivo']['nombre'].split(':')[0]}** | **{resultados_auditorias[2]['las_5_dimensiones_iml']['arquetipo_discursivo']['nombre'].split(':')[0]}** | **{resultados_auditorias[3]['las_5_dimensiones_iml']['arquetipo_discursivo']['nombre'].split(':')[0]}** | Clasificación cualitativa de estilo retórico |\n\n"
        )

        f.write("### 6.2 Matriz Exhaustiva y Completa de las 36 Variables Cuantitativas Base (D0 a D5)\n\n")
        f.write(
            "Esta tabla detalla las **36 variables cuantitativas base** extraídas directamente del texto de las 4 noticias de control, junto con su rango, su bloque temático y su interpretación periodística:\n\n"
            "| # | Dimensión | Variable | Noticia 1 (Falsa Salud) | Noticia 2 (Falsa Conspir.) | Noticia 3 (Real Puebla) | Noticia 4 (Real Ciencia EFE) | Rango / Unidad | Interpretación Forense y Periodística |\n"
            "|---|---|---|:---:|:---:|:---:|:---:|:---:|---|\n"
        )

        vars_def = [
            # D0: Sensacionalismo y Redundancia Base (10)
            ("D0", "P_full", lambda aud: f"{aud['sensacionalismo_beto']['P_full']:.4f}", "[0, 1]", "Sensacionalismo documental global asignado por BETO fine-tuned"),
            ("D0", "P_mean", lambda aud: f"{aud['sensacionalismo_beto']['P_mean']:.4f}", "[0, 1]", "Promedio del sensacionalismo en todas las oraciones del documento"),
            ("D0", "P_max", lambda aud: f"{aud['sensacionalismo_beto']['P_max']:.4f}", "[0, 1]", "Máximo sensacionalismo oracional (identifica la Oración Gatillo)"),
            ("D0", "P_top2", lambda aud: f"{aud['sensacionalismo_beto']['P_top2']:.4f}", "[0, 1]", "Media de las dos oraciones con mayor carga sensacionalista"),
            ("D0", "sigma_sens", lambda aud: f"{aud['sensacionalismo_beto']['sigma_sens']:.4f}", "[0, 0.5]", "Desviación estándar oracional: dispersión afectiva interna"),
            ("D0", "dilution_ratio", lambda aud: f"{aud['sensacionalismo_beto']['dilution_ratio']:.4f}", "[0, 1+]", "Dilution Ratio: P_full / P_max (mide si el cuerpo diluye el titular)"),
            ("D0", "delta_p_gatillo", lambda aud: f"{aud['sensacionalismo_beto']['delta_p_gatillo_causal']:.4f}", "[-1, 1]", "Impacto causal contrafáctico: caída de alarma al silenciar 2 gatillos"),
            ("D0", "max_intra_similarity", lambda aud: f"{aud['redundancia_sbert']['max_intra_similarity']:.4f}", "[0, 1]", "Máxima similitud coseno SBERT entre dos frases del documento"),
            ("D0", "mean_intra_similarity", lambda aud: f"{aud['redundancia_sbert']['mean_intra_similarity']:.4f}", "[0, 1]", "Media de similitudes coseno intra-documentales"),
            ("D0", "redundancy_density", lambda aud: f"{aud['redundancia_sbert']['redundancy_density_pct']:.1f}%", "[0, 100%]", "Densidad de pares de oraciones con similitud cos >= 0.34 (GMM)"),
            # D1: Coherencia y Flujo Secuencial (3)
            ("D1", "consec_sim_mean", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D1_Dinamica_Discursiva']['consec_sim_mean']:.4f}", "[0, 1]", "Coherencia temática promedio entre oraciones adyacentes consecutivas"),
            ("D1", "consec_sim_min", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D1_Dinamica_Discursiva']['consec_sim_min']:.4f}", "[0, 1]", "Salto temático más abrupto o ruptura narrativa entre oraciones contiguas"),
            ("D1", "consec_sim_std", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D1_Dinamica_Discursiva']['consec_sim_std']:.4f}", "[0, 0.5]", "Inestabilidad en la transición discursiva entre frases sucesivas"),
            # D2: Lingüística Forense y Epistémica (5)
            ("D2", "dicendi_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D2_Linguistica_Forense_Epistemica']['dicendi_density']:.2f}%", "[0, 100%]", "Densidad de verbos de reporte y atribución ('declaró', 'indicó', 'afirmó')"),
            ("D2", "quotes_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D2_Linguistica_Forense_Epistemica']['quotes_density']:.2f}", "Ratio / orac", "Frecuencia de citas textuales entrecomilladas por oración"),
            ("D2", "hedges_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D2_Linguistica_Forense_Epistemica']['hedges_density']:.2f}%", "[0, 100%]", "Atenuadores de prudencia epistémica ('presunto', 'al parecer', 'podría')"),
            ("D2", "boosters_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D2_Linguistica_Forense_Epistemica']['boosters_density']:.2f}%", "[0, 100%]", "Intensificadores dogmáticos de certeza ('sin duda', 'obvio', 'es un hecho')"),
            ("D2", "epistemic_ratio", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D2_Linguistica_Forense_Epistemica']['epistemic_ratio']:.2f}", "[0, inf)", "Ratio Asertividad / Cautela: (Boosters + 0.01) / (Hedges + 0.1)"),
            # D3: Riqueza Léxica y Legibilidad (6)
            ("D3", "hapax_ratio", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D3_Riqueza_Lexica_Legibilidad']['hapax_ratio_pct']:.1f}%", "[0, 100%]", "Hapax Legomena: porcentaje de palabras empleadas una sola vez"),
            ("D3", "guiraud_ttr", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D3_Riqueza_Lexica_Legibilidad']['guiraud_ttr']:.2f}", "[1, 20]", "Índice de Guiraud R = Vocabulario / sqrt(Palabras) (invariante a longitud)"),
            ("D3", "conteo_palabras_text", lambda aud: f"{aud['metadatos']['conteo_palabras']}", "Entero", "Longitud total en palabras del documento"),
            ("D3", "num_sentences", lambda aud: f"{aud['metadatos']['num_oraciones']}", "Entero", "Número total de oraciones segmentadas en el texto"),
            ("D3", "gutierrez_polini", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D3_Riqueza_Lexica_Legibilidad']['gutierrez_polini']:.1f}", "[0, 100]", "Score de legibilidad de Gutiérrez de Polini adaptado al español"),
            ("D3", "flesch_szigriszt", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D3_Riqueza_Lexica_Legibilidad']['flesch_szigriszt']:.1f}", "[0, 100]", "Índice de perspicuidad de Flesch-Szigriszt (facilidad lectora)"),
            # D4: Morfosintaxis spaCy POS Tagging (4)
            ("D4", "adv_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D4_Perfilado_Morfosintactico_POS']['adv_density']:.2f}%", "[0, 100%]", "TOP 1 PREDICTOR: Densidad de adverbios modales y enfáticos (subjetividad)"),
            ("D4", "pron_1p_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D4_Perfilado_Morfosintactico_POS']['pron_1p_density']:.2f}%", "[0, 100%]", "Pronombres de 1.ª persona ('nosotros', 'me', 'nos'): apelo emocional"),
            ("D4", "pron_3p_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D4_Perfilado_Morfosintactico_POS']['pron_3p_density']:.2f}%", "[0, 100%]", "Pronombres de 3.ª persona ('él', 'se', 'su'): registro formal e impersonal"),
            ("D4", "adj_noun_ratio", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D4_Perfilado_Morfosintactico_POS']['adj_noun_ratio']:.3f}", "[0, inf)", "Ratio de adjetivación: Adjetivos / (Sustantivos + 1)"),
            # D5: Anclajes Factuales, Mayúsculas y Puntuación (8)
            ("D5", "upper_chars_ratio", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['upper_chars_ratio_pct']:.2f}%", "[0, 100%]", "TOP 2 PREDICTOR: Proporción de caracteres en mayúsculas sobre letras"),
            ("D5", "all_caps_ratio", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['all_caps_ratio_pct']:.2f}%", "[0, 100%]", "TOP 3 PREDICTOR: Palabras completas en MAYÚSCULAS sostenidas (grito digital)"),
            ("D5", "numbers_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['numbers_density']:.2f}%", "[0, 100%]", "Densidad de cifras numéricas: anclaje factual cuantitativo verificable"),
            ("D5", "temporal_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['temporal_density']:.2f}%", "[0, 100%]", "Densidad de fechas precisas (meses y años específicos)"),
            ("D5", "punct_intensity", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['punct_intensity']:.2f}%", "[0, 100%]", "Intensidad agregada de signos expresivos no neutros (!, ?, ...)"),
            ("D5", "excl_density", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['excl_density']:.2f}%", "[0, 100%]", "Densidad de signos de exclamación (! / ¡) por 100 palabras"),
            ("D5", "percent_count", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['percent_count']}", "Conteo", "Total de referencias porcentuales (%) o menciones 'por ciento'"),
            ("D5", "all_caps_count", lambda aud: f"{aud['variables_ensamble_avanzado_5d']['D5_Anclajes_Factuales_Puntuacion']['all_caps_count']}", "Conteo", "Conteo absoluto de palabras vociferadas en mayúsculas sostenidas")
        ]

        for num_v, (dim, name, func, rango, desc) in enumerate(vars_def, 1):
            val0 = func(resultados_auditorias[0])
            val1 = func(resultados_auditorias[1])
            val2 = func(resultados_auditorias[2])
            val3 = func(resultados_auditorias[3])
            f.write(f"| **{num_v}** | `{dim}` | `{name}` | `{val0}` | `{val1}` | `{val2}` | `{val3}` | {rango} | {desc} |\n")

        f.write("\n---\n\n")

        # Auditorías detalladas noticia por noticia
        for i, aud in enumerate(resultados_auditorias):
            f.write(f"### 6.{i+3} Auditoría Detallada Frase a Frase: Noticia {i+1} — {aud['metadatos']['titulo']}\n\n")
            f.write(f"**Referencia:** {NOTICIAS_BENCHMARK[i]['tipo_referencia']} | **Origen:** {NOTICIAS_BENCHMARK[i]['origen']}\n\n")
            f.write(f"#### Texto Completo Analizado\n")
            f.write(f"> {NOTICIAS_BENCHMARK[i]['texto']}\n\n")

            sub_md = auditor.formatear_informe_markdown(aud)
            f.write(sub_md)
            f.write("\n\n---\n\n")

        # Conclusiones forenses inmediatas
        f.write("### 6.7 Conclusiones Forenses Inmediatas sobre los Casos de Control\n\n")
        f.write("1. **La Asimetría Radical de las Mayúsculas Sostenidas (D5):** En el Bulo de Salud con Mayúsculas se constata un 8.73% de palabras completas en mayúsculas (y un 26.3% en la primera frase), frente a un 0.0% estricto en la nota oficial de salud pública y la crónica de EFE.\n")
        f.write("2. **La Presencia de Verbos Dicendi como Huella Profesional (D2):** La nota de salud de Puebla muestra un 2.31% de verbos de reporte y atribución formal ('indicó', 'anunció'), anclando la información en autoridades verificables, mientras que los bulos presentan un 0.0% estricto.\n")
        f.write("3. **El Mecanismo Causal del Gatillo Sensacionalista (D2 y Ablación):** En la Noticia 1, silenciar las dos oraciones gatillo produce una caída del sensacionalismo documental de $\\Delta P_{gatillo} = 0.933$ (caída del 93.3%), demostrando que todo el artículo fue un cebo artificial sostenido por dos premisas hiperbólicas.\n")
        f.write("4. **Coherencia en la Banda de Shannon vs. Incoherencia Rota (D4 y D5):** La Noticia 4 (EFE) opera dentro de la Banda Óptima de Shannon con una cohesión discursiva perfecta ($D_5 = 99.8$), mientras que el panfleto negacionista exhibe fragmentación proposicional ($D_5 = 41.0$).\n\n")

        # ---------------------------------------------------------------------
        # SECCIÓN 7: EL PUNTO DE INFLEXIÓN CIENTÍFICO Y LA PREGUNTA DE TRANSICIÓN
        # ---------------------------------------------------------------------
        f.write("## 7. El Punto de Inflexión Científico: ¿Se Puede Clasificar la Desinformación a partir de esta Explicabilidad Forense?\n\n")
        f.write(
            "Al culminar esta caracterización forense de 36 variables cuantitativas y 5 dimensiones estilométricas, la investigación "
            "llega a un hallazgo de enorme trascendencia científica:\n\n"
            "> **Las diferencias entre desinformación y periodismo legítimo NO son arbitrarias ni dependen del vocabulario político coyuntural: "
            "quedan fielmente reflejadas en la morfología sintáctica, la saturación adverbial, el abuso de mayúsculas, la ausencia de verbos de reporte "
            "y la presencia de bucles redundantes.**\n\n"
            "A partir de este resultado, surge la **pregunta de investigación central** que conecta la explicabilidad forense con la inteligencia artificial aplicada:\n\n"
        )

        f.write("```mermaid\n")
        f.write("flowchart LR\n")
        f.write("    subgraph EXP[\"Explicabilidad Forense Estilométrica (Fase 1)\"]\n")
        f.write("        VARS[\"36 Variables Cuantitativas Base<br/>(D0 a D5: Morfosintaxis, Mayúsculas, Dicendi, etc.)\"]\n")
        f.write("        IML[\"Perfilador IML (5 Dimensiones 0-100)<br/>& Arquetipos Discursivos\"]\n")
        f.write("    end\n\n")
        f.write("    subgraph PREGUNTA[\"Punto de Inflexión Científico\"]\n")
        f.write("        P[\"¿Es posible clasificar noticias<br/>con estas 36 variables forenses<br/>superando el Domain Shift?\"]\n")
        f.write("    end\n\n")
        f.write("    subgraph ENSAMBLE[\"Modelo Ensamble Adaptativo 5D (Fase 2)\"]\n")
        f.write("        GBDT[\"GBDT (140 árboles, profundidad 3)<br/>Umbral Bayesiano θ* = 0.42\"]\n")
        f.write("        LATAM[\"Supera a SaBERT en América Latina<br/>Exactitud: 68.36% vs 65.89%<br/>Recall: 62.16% vs 31.30%\"]\n")
        f.write("    end\n\n")
        f.write("    EXP --> PREGUNTA\n")
        f.write("    PREGUNTA --> ENSAMBLE\n")
        f.write("```\n\n")

        f.write("```\n")
        f.write("┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
        f.write("│                                           PREGUNTA CENTRAL DE INVESTIGACIÓN                                           │\n")
        f.write("├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤\n")
        f.write("│ «¿Es posible aprovechar todo este espacio de 36 características forenses explicables (Dimensiones D0 a D5)            │\n")
        f.write("│  para entrenar un modelo alternativo y adaptativo capaz de clasificar noticias falsas vs. verdaderas, superando       │\n")
        f.write("│  la opacidad, la memorización espuria de nombres propios y el colapso por Domain Shift de los modelos tradicionales   │\n")
        f.write("│  de caja negra (como SaBERT o BERT plano)?»                                                                           │\n")
        f.write("└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")
        f.write("```\n\n")
        f.write("La respuesta a esta pregunta científica es **afirmativa**: aquí es donde entra formalmente en la tesis el **Modelo Ensamble Adaptativo Multidimensional**.\n\n")

        # ---------------------------------------------------------------------
        # SECCIÓN 8: EL MODELO ENSAMBLE ADAPTATIVO Y SUS MÉTRICAS EXPERIMENTALES
        # ---------------------------------------------------------------------
        f.write("## 8. La Respuesta de la Investigación: El Modelo Ensamble Adaptativo Multidimensional y sus Métricas Experimentales\n\n")
        f.write(
            "Para responder a la pregunta de investigación, se diseñó e implementó el **Modelo Ensamble Morfológico Avanzado 5D** "
            "([`Reporte_Ensamble_Adaptativo_FakeNews.md`](../../Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Ensamble_Adaptativo_FakeNews.md)).\n\n"
            "### 8.1 Arquitectura del Ensamble: Superando las Cajas Negras mediante Características Estilométricas\n"
            "En lugar de alimentar una red neuronal con texto en bruto para que memorice atajos dialectales (como nombres de ministros o leyes locales), "
            "el Ensamble opera **exclusivamente sobre el vector numérico de las variables forenses extraídas**:\n"
            "- **Algoritmo Base:** Gradient Boosting Decision Trees (GBDT / Scikit-Learn Ensemble) con 140 estimadores, profundidad máxima de árbol de 3 y submuestreo de características (0.85).\n"
            "- **Espacio de Entrada:** 42 variables numéricas consolidadas (las 36 variables base más codificaciones OHE de bins de Shannon).\n"
            "- **Validación Cruzada Estratificada:** 5-Fold Stratified Cross-Validation sobre el corpus completo garantizando independencia de particiones.\n"
            "- **Calibración Adaptativa Bayesiana (Índice de Youden):** En lugar de usar el umbral ingenuo $\\theta = 0.50$, se calculó el umbral bayesiano óptimo $\\theta^* = 0.42$, "
            "diseñado específicamente para maximizar la sensibilidad (*recall*) ante el engaño sin degradar la precisión.\n\n"
            "### 8.2 Desempeño Experimental en el Corpus Ampliado ($N = 4.418$ Noticias)\n"
            "El modelo fue evaluado sobre el corpus ampliado de la tesis ($2.411$ verdaderas y $2.007$ falsas), dividido entre **España/Europa** ($N = 2.354$) "
            "y **América Latina** ($N = 2.064$), confrontándolo contra el clasificador SaBERT oficial del estado del arte:\n\n"
            "```\n"
            "+===================================================================================================================================================+\n"
            "|               TABLA 8.1: COMPARATIVA METROLÓGICA FORMAL: ENSAMBLE FORENSE 5D vs. SaBERT (N = 4.418 NOTICIAS)                                      |\n"
            "+=========================================+=====================================+=====================================+=============================+\n"
            "| Métrica Evaluada                        | Ensamble Base (12 Vars)             | Ensamble Avanzado 5D (42 Vars)      | SaBERT Oficial              |\n"
            "|                                         | θ = 0,50        │ θ* = 0,42 (Calib.)│ θ = 0,50        │ θ* = 0,42 (Calib.)│ (Colapso en LatAm)          |\n"
            "+=========================================+=================+===================+=================+===================+=============================+\n"
            "| **Capacidad de Separación (ROC-AUC)**   | 0,6568          | 0,6568            | **0,7178 (+6,10)**| **0,7178 (+6,10)**| 0,8519 (En España) / Colapso|\n"
            "| **Área Precision-Recall (PR-AUC)**      | 0,6213          | 0,6213            | **0,7038 (+8,25)**| **0,7038 (+8,25)**| 0,7821                      |\n"
            "| **Exactitud en América Latina**         | 63,52 %         | 61,87 %           | **68,36 %**     | **65,70 %**       | **65,89 % (Cae -23,8 pts)** |\n"
            "| **ROC-AUC en América Latina**           | 0,6608          | 0,6608            | **0,7162**      | **0,7162**        | **0,6800 (Colapso severo)** |\n"
            "| **Recall Fake News LatAm (Sensibilidad)**| 43,95 %        | 61,83 %           | 50,05 %         | **62,16 %**       | **31,30 % (Omite 68,7% bulos|\n"
            "| **Bulos Omitidos en LatAm (FN)**        | 514             | 350               | 458             | **347 (Rescatados)**| **630 bulos omitidos**      |\n"
            "| Recall Global Fake News (N=4.418)       | 43,10 %         | 72,60 %           | 50,62 %         | **70,05 %**       | 57,75 %                     |\n"
            "| Bulos Capturados Globales (TP)          | 865 bulos       | 1.457 bulos       | 1.016 bulos     | **1.406 bulos**   | 1.159 bulos                 |\n"
            "| Matriz de Confusión Global [TN, FP]     | TN=1836, FP=575 │ TN=1170, FP=1241  | TN=1886, FP=525 │ **TN=1414, FP=997** | TN=2313, FP=98              |\n"
            "|                            [FN, TP]     | FN=1142, TP=865 │ FN=550,  TP=1457  | FN=991,  TP=1016│ **FN=601,  TP=1406**| FN=848,  TP=1159            |\n"
            "+=========================================+=================+===================+=================+===================+=============================+\n"
            "```\n\n"
            "### 8.3 El Fenómeno del Domain Shift: Por qué el Ensamble Forense Supera a SaBERT en América Latina\n"
            "1. **Superación Formal de SaBERT en América Latina:**\n"
            "   - En exactitud pura, el Ensamble Avanzado 5D alcanza un **$68,36\\%$**, superando los $65,89\\%$ de SaBERT.\n"
            "   - En ROC-AUC intrínseco, el Ensamble alcanza **$0,7162$**, superando con holgura el **$0,6800$** de SaBERT.\n"
            "2. **Duplicación del Recall y Rescate de Bulos:**\n"
            "   - SaBERT sufrió un colapso dramático al cruzar a América Latina: su tasa de detección cayó al **$31,30\\%$**, dejando escapar **$630$ bulos** (un alarmante $68,7\\%$ de falsos negativos) debido a que sobreajustó con nombres políticos de España (Vox, Pedro Sánchez, leyes peninsulares).\n"
            "   - El Ensamble Forense Calibrado ($\\theta^* = 0,42$) alcanza un **$Recall = 62,16\\%$** en América Latina, **duplicando la detección de SaBERT** y reduciendo los bulos omitidos a $347$.\n\n"
            "### 8.4 Importancia de Características (SHAP / Gini): ¿Qué Variables Mandan en la Clasificación?\n"
            "El análisis de ganancia de información reveló qué variables extraídas por el método forense son las más decisivas para clasificar:\n\n"
            "```\n"
            "DISTRIBUCIÓN DEL PESO PREDICTIVO POR DIMENSIÓN EN EL ENSAMBLE:\n"
            "├── D0: Sensacionalismo y Redundancia Base (31,10%) ── [Pilar Semántico]\n"
            "├── D5: Anclajes Factuales y Mayúsculas (24,11%) ──── [Pilar Enfático]\n"
            "├── D3: Riqueza Léxica y Legibilidad (19,34%) ─────── [Pilar Estilístico]\n"
            "├── D4: Morfosintaxis y Subjetividad POS (14,78%) ─── [Pilar Gramatical]\n"
            "├── D2: Lingüística Forense y Citas (6,56%) ───────── [Pilar Epistémico]\n"
            "└── D1: Coherencia y Flujo Secuencial (4,10%) ─────── [Pilar Discursivo]\n"
            "```\n\n"
            "- **Top 1 Individual: Densidad de Adverbios (`adv_density` = $8,41\\%$):** Fue la característica individual con mayor ganancia de Gini de todo el modelo. Los generadores de bulos saturan el texto de adverbios valorativos (*«claramente», «obviamente», «jamás»*), traicionando la sobriedad informativa.\n"
            "- **Top 2 y Top 3 Global: Mayúsculas Sostenidas (`upper_chars_ratio` = $7,49\\%$ y `all_caps_ratio` = $6,96\\%$):** Juntas representan casi un **$15\\%$ de todo el poder predictivo del clasificador**. La prensa seria prohíbe el uso de mayúsculas sostenidas, mientras que la desinformación viral recurre a ellas de forma sistemática para inducir urgencia visual.\n\n"
            "> **Conclusión Epistemológica de la Tesis:** La clasificación binaria no precede a la explicabilidad. **La clasificación robusta, generalizable y resistente al Domain Shift es el fruto de haber construido primero una sólida explicabilidad estilométrica de 36 dimensiones.**\n\n"
        )

        # ---------------------------------------------------------------------
        # SECCIÓN 9: MAPA DE DEPENDENCIAS DOCUMENTALES
        # ---------------------------------------------------------------------
        f.write("## 9. Mapa de Dependencias Documentales: ¿Qué Otros Archivos o Reportes se Necesitan para Entender Este y Por Qué?\n\n")
        f.write(
            "Para comprender en su totalidad el sustento teórico, la calibración matemática y los modelos subyacentes de este reporte, "
            "es indispensable consultar los siguientes documentos y códigos del repositorio de tesis:\n\n"
        )

        f.write("### 1. Marco Teórico y Replanteamiento Epistemológico\n")
        f.write("- **Archivo:** [`Plan_tesis.md`](../../../Plan_tesis.md)\n")
        f.write("- **¿Por qué se necesita?** Establece la fundamentación epistemológica de toda la investigación. Explica por qué la verdad fáctica es extrínseca al texto y demuestra las limitaciones insalvables de los clasificadores binarios de caja negra (BETO, RoBERTa, SaBERT) frente a la desinformación sofisticada.\n\n")

        f.write("### 2. Formulación Matemática del IML y los 5 Arquetipos Discursivos (Fase 1 Forense)\n")
        f.write("- **Archivo:** [`Reporte_Estilos_Manipulacion_Textual.md`](Reporte_Estilos_Manipulacion_Textual.md)\n")
        f.write("- **¿Por qué se necesita?** Contiene la deducción teórica de las **5 Dimensiones del IML** ($D_1$ a $D_5$), sus rangos de calibración y los umbrales de activación de los **5 Arquetipos Discursivos**. Además, incluye el estudio empírico masivo sobre 2.604 noticias donde se demostró la Paradoja 1 (el Estilo III tiene IML de 21.2 pero 70.2% de fake news) y las Figuras 4 (radar) y 5 (tasas empíricas).\n\n")

        f.write("### 3. Modelo de Ensamble Adaptativo e Importancia Global de Variables (Fase 2 de Clasificación)\n")
        f.write("- **Archivo:** [`Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Ensamble_Adaptativo_FakeNews.md`](../../Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Ensamble_Adaptativo_FakeNews.md)\n")
        f.write("- **¿Por qué se necesita?** Expone el rendimiento predictivo del ensamble GBDT, la calibración bayesiana del umbral adaptativo ($\\theta = 0.42$) y el ranking de importancia de características mediante valores SHAP. Este reporte demostró que las variables morfosintácticas (`adv_density`, `all_caps_ratio`, `upper_chars_ratio`) superan en poder discriminativo a la semántica pura.\n\n")

        f.write("### 4. Diccionario Ontológico y Matemático de las 36 Variables\n")
        f.write("- **Archivo:** [`dashboard/diccionario_variables_5d.json`](../../../dashboard/diccionario_variables_5d.json)\n")
        f.write("- **¿Por qué se necesita?** Es el catálogo exhaustivo de las 36 variables cuantitativas del Ensamble Avanzado. Para cada variable define su fórmula matemática, su hipótesis periodística, su rango de valores, su ranking de importancia y el impacto explicativo que genera en el texto.\n\n")

        f.write("### 5. Códigos del Perfilador Forense y Extracción Masiva\n")
        f.write("- **Archivos:** [`codigos/perfilador_estilo_manipulacion.py`](../codigos/perfilador_estilo_manipulacion.py), [`metodo_explicabilidad.py`](../metodo_explicabilidad.py) y [`extraer_features_avanzadas_5d.py`](../../Clasificar_fake/Modelo_Ensamble/Codigos/extraer_features_avanzadas_5d.py)\n")
        f.write("- **¿Por qué se necesitan?** Contienen la implementación computacional donde se realiza el perfilado estilométrico de 5 dimensiones y la extracción masiva en GPU para el corpus completo de noticias.\n\n")

        f.write("### 6. Calibración del Umbral de Redundancia y Bins de Shannon\n")
        f.write("- **Carpeta:** [`Modelos_Individuales/Redundancia/Reportes/`](../../Redundancia/Reportes)\n")
        f.write("- **¿Por qué se necesita?** Detalla los experimentos con Sentence-BERT y la justificación probabilística del umbral $\\tau = 0.34$, obtenido mediante Modelos de Mezcla Gaussiana (GMM) para separar la similitud temática natural de la paráfrasis redundante, así como los cortes de los árboles de decisión de entropía (bins 7 y 4 de Shannon).\n\n")

        f.write("### 7. Validación de Estabilidad y Sensibilidad de BETO\n")
        f.write("- **Carpeta:** [`Modelos_Individuales/Sensacionalismo/Pruebas_Frases/`](../../Sensacionalismo/Pruebas_Frases)\n")
        f.write("- **¿Por qué se necesita?** Documenta los experimentos de perturbación sintáctica y ablación oracional que demostraron la sensibilidad de BETO ante variaciones locales y validaron el cálculo causal del $\\Delta P_{gatillo}$.\n")

    print(f"🎉 ¡Auditoría forense de las 4 noticias y compilación de informe completada con éxito!")

if __name__ == "__main__":
    main()
