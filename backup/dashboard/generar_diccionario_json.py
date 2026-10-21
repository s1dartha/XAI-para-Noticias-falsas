import json

# Full dictionary of variables with technical details, formulas, and hypotheses
variables = [
    # D0
    {
        "id": "P_full",
        "name": "P_full",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 5.83,
        "importance_rank": 4,
        "tipo": "Probabilidad continua [0, 1]",
        "formula": "P(Sensacionalista | D) vía BETO fine-tuned",
        "hipotesis": "Los bulos despliegan una intensidad emocional desproporcionada en el texto global.",
        "efecto": "Valores altos (>0.70) aumentan fuertemente la probabilidad de bulo."
    },
    {
        "id": "P_mean",
        "name": "P_mean",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 5.49,
        "importance_rank": 5,
        "tipo": "Media continua [0, 1]",
        "formula": "\\frac{1}{|S|} \\sum_{s \\in S} P(Sens | s)",
        "hipotesis": "En textos engañosos, el tono sensacionalista permea uniformemente la mayoría de las oraciones.",
        "efecto": "Valores altos señalan agresividad o tono alarmista sostenido."
    },
    {
        "id": "P_max",
        "name": "P_max",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 5.23,
        "importance_rank": 6,
        "tipo": "Máximo continuo [0, 1]",
        "formula": "\\max_{s \\in S} P(Sens | s)",
        "hipotesis": "Identifica la 'oración gatillo' que concentra la mayor carga dramática o conspirativa.",
        "efecto": "Basta una sola oración hiperbólica (>0.90) para que el artículo actúe como cebo viral."
    },
    {
        "id": "P_top2",
        "name": "P_top2",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 3.65,
        "importance_rank": 13,
        "tipo": "Media top-2 continuo [0, 1]",
        "formula": "\\frac{P_{(1)} + P_{(2)}}{2}",
        "hipotesis": "Amortigua anomalías aisladas exigiendo confirmación de un segundo pico afectivo.",
        "efecto": "Robustez contra falsas alarmas provocadas por una cita sensacionalista en noticia seria."
    },
    {
        "id": "sigma_sens",
        "name": "sigma_sens",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 1.25,
        "importance_rank": 22,
        "tipo": "Desviación estándar [0, 0.5]",
        "formula": "\\sqrt{\\frac{1}{|S|} \\sum (P(s) - \\bar{P})^2}",
        "hipotesis": "La varianza afectiva mide la alternancia entre párrafos descriptivos y ataques viscerales.",
        "efecto": "Los bulos polarizados exhiben picos erráticos de alta varianza."
    },
    {
        "id": "dilution_ratio",
        "name": "dilution_ratio",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 3.19,
        "importance_rank": 14,
        "tipo": "Ratio continuo [0, 1+]",
        "formula": "\\frac{P_{full}}{P_{max} + 10^{-6}}",
        "hipotesis": "Mide si el cuerpo documental diluye o amplifica el titular/párrafo gatillo.",
        "efecto": "Ratios cercanos a 1.0 demuestran que el sensacionalismo domina todo el documento."
    },
    {
        "id": "delta_p_gatillo",
        "name": "delta_p_gatillo",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 2.35,
        "importance_rank": 16,
        "tipo": "Diferencial de probabilidad [-1, 1]",
        "formula": "P(D) - P(D \\setminus \\{s_{top1}, s_{top2}\\})",
        "hipotesis": "Impacto causal contrafáctico: ¿cuánto cae el sensacionalismo si silenciamos los dos gatillos?",
        "efecto": "Delta grande (>0.30) delata una noticia inflada artificialmente por 1 o 2 frases señuelo."
    },
    {
        "id": "max_intra_similarity_clean",
        "name": "max_intra_similarity_clean",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 2.29,
        "importance_rank": 17,
        "tipo": "Coseno continuo [0, 1]",
        "formula": "\\max_{i<j} \\cos(e_i, e_j) \\text{ vía Sentence-BERT}",
        "hipotesis": "Los bulos tienden a parafrasear y repetir la misma consigna sin aportar datos nuevos.",
        "efecto": "Similitud >0.65 entre oraciones distantes delata bucles retóricos."
    },
    {
        "id": "mean_intra_similarity_clean",
        "name": "mean_intra_similarity_clean",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 0.94,
        "importance_rank": 26,
        "tipo": "Coseno medio [0, 1]",
        "formula": "\\text{Media del triángulo superior de matriz coseno}",
        "hipotesis": "Mide la cohesión semántica global del texto periodístico.",
        "efecto": "Noticias legítimas balancean cohesión temática con progresión informativa nueva."
    },
    {
        "id": "redundancy_density",
        "name": "redundancy_density",
        "dim": "D0",
        "dim_name": "D0: Sensacionalismo y Redundancia Base",
        "importance": 0.88,
        "importance_rank": 27,
        "tipo": "Ratio continuo [0, 1]",
        "formula": "\\frac{\\text{Pares con } \\cos \\ge 0.34}{\\text{Total de pares}}",
        "hipotesis": "Densidad de oraciones duplicadas según umbral calibrado por Modelo de Mezcla Gaussiana (GMM).",
        "efecto": "Proporciones altas indican circularidad argumental típica de desinformación."
    },

    # D1
    {
        "id": "consec_sim_mean",
        "name": "consec_sim_mean",
        "dim": "D1",
        "dim_name": "D1: Coherencia y Flujo Secuencial",
        "importance": 1.28,
        "importance_rank": 21,
        "tipo": "Coseno medio adyacente [0, 1]",
        "formula": "\\frac{1}{|S|-1} \\sum_{i=1}^{|S|-1} \\cos(e_i, e_{i+1})",
        "hipotesis": "El periodismo profesional mantiene una progresión temática secuencial coherente.",
        "efecto": "Los bulos ensamblados a partir de retazos inconexos muestran caídas en la coherencia adyacente."
    },
    {
        "id": "consec_sim_min",
        "name": "consec_sim_min",
        "dim": "D1",
        "dim_name": "D1: Coherencia y Flujo Secuencial",
        "importance": 1.25,
        "importance_rank": 23,
        "tipo": "Coseno mínimo adyacente [0, 1]",
        "formula": "\\min_i \\cos(e_i, e_{i+1})",
        "hipotesis": "Detecta baches y rupturas lógicas abruptas (saltos temáticos injustificados).",
        "efecto": "Un mínimo muy bajo (<0.08) evidencia yuxtaposición artificial de párrafos sin ilación."
    },
    {
        "id": "consec_sim_std",
        "name": "consec_sim_std",
        "dim": "D1",
        "dim_name": "D1: Coherencia y Flujo Secuencial",
        "importance": 1.57,
        "importance_rank": 20,
        "tipo": "Desviación estándar de similitud",
        "formula": "\\text{std}(\\cos(e_i, e_{i+1}))",
        "hipotesis": "La inestabilidad en la transición de ideas delata textos no editados profesionalmente.",
        "efecto": "Alta variabilidad revela falta de hilo conductor narrativo."
    },

    # D2
    {
        "id": "dicendi_density",
        "name": "dicendi_density",
        "dim": "D2",
        "dim_name": "D2: Lingüística Forense y Epistémica",
        "importance": 4.88,
        "importance_rank": 8,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Verbos Dicendi}}{\\text{Palabras}} \\times 100",
        "hipotesis": "Las noticias verídicas citan fuentes mediante verbos de reporte ('declaró', 'señaló', 'afirmó').",
        "efecto": "Densidad nula o ínfima es uno de los síntomas más característicos de bulos sin fuente."
    },
    {
        "id": "quotes_density",
        "name": "quotes_density",
        "dim": "D2",
        "dim_name": "D2: Lingüística Forense y Epistémica",
        "importance": 0.81,
        "importance_rank": 28,
        "tipo": "Densidad por oración",
        "formula": "\\frac{\\sum \\text{Citas entrecomilladas}}{\\text{Oraciones}}",
        "hipotesis": "El uso de testimonios directos en comillas delimita la responsabilidad editorial del periodista.",
        "efecto": "La desinformación prefiere aserciones impersonales absolutas sin citas directas."
    },
    {
        "id": "hedges_density",
        "name": "hedges_density",
        "dim": "D2",
        "dim_name": "D2: Lingüística Forense y Epistémica",
        "importance": 0.42,
        "importance_rank": 33,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Atenuadores (presunto, al parecer)}}{\\text{Palabras}} \\times 100",
        "hipotesis": "El periodismo riguroso emplea atenuadores de prudencia epistémica ante hechos no sentenciados.",
        "efecto": "Baja presencia de hedges indica dogmatismo y falta de rigor periodístico."
    },
    {
        "id": "boosters_density",
        "name": "boosters_density",
        "dim": "D2",
        "dim_name": "D2: Lingüística Forense y Epistémica",
        "importance": 0.38,
        "importance_rank": 34,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Intensificadores (indudablemente, obvio)}}{\\text{Palabras}} \\times 100",
        "hipotesis": "La desinformación impone certezas artificiales agresivas para persuadir rápidamente.",
        "efecto": "Altas densidades de boosters correlacionan con bulos ideológicos y teorías conspirativas."
    },
    {
        "id": "epistemic_ratio",
        "name": "epistemic_ratio",
        "dim": "D2",
        "dim_name": "D2: Lingüística Forense y Epistémica",
        "importance": 0.07,
        "importance_rank": 38,
        "tipo": "Ratio asertividad / cautela",
        "formula": "\\frac{\\text{Boosters} + 0.01}{\\text{Hedges} + 0.1}",
        "hipotesis": "Compara directamente la fuerza de aserción frente a la cautela deontológica.",
        "efecto": "Valores muy superiores a 1.0 revelan retórica dogmática no periodística."
    },

    # D3
    {
        "id": "hapax_ratio",
        "name": "hapax_ratio",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 5.21,
        "importance_rank": 7,
        "tipo": "Ratio porcentual [0, 100]",
        "formula": "\\frac{|\\{w : \\text{frec}(w) = 1\\}|}{\\text{Total Palabras}} \\times 100",
        "hipotesis": "Proporción de términos utilizados exactamente una sola vez (Hapax Legomena).",
        "efecto": "Redactores profesionales presentan alta diversidad léxica; los bulos reutilizan un vocabulario restringido."
    },
    {
        "id": "guiraud_ttr",
        "name": "guiraud_ttr",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 4.67,
        "importance_rank": 9,
        "tipo": "Índice continuo Guiraud R",
        "formula": "R = \\frac{|V|}{\\sqrt{N}} \\text{ (Invariante a longitud)}",
        "hipotesis": "Mide la riqueza del vocabulario V compensando el efecto distorsionador del tamaño N.",
        "efecto": "Valores bajos (<6.0) revelan pobreza y reiteración de términos clave en fake news."
    },
    {
        "id": "conteo_palabras_text",
        "name": "conteo_palabras_text",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 4.32,
        "importance_rank": 10,
        "tipo": "Conteo entero",
        "formula": "\\text{Total de tokens de palabras en el artículo}",
        "hipotesis": "Los bulos virales de redes suelen ser breves (70-130 palabras) para facilitar su consumo rápido.",
        "efecto": "Textos cortos sin desarrollo contextual aumentan la sospecha de bulo."
    },
    {
        "id": "num_sentences",
        "name": "num_sentences",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 1.74,
        "importance_rank": 19,
        "tipo": "Conteo entero",
        "formula": "\\text{Total de oraciones detectadas por NLTK}",
        "hipotesis": "Estructura sintáctica y segmentación compositiva del documento.",
        "efecto": "Noticias con pocas oraciones largas o demasiadas oraciones truncadas indican redacción informal."
    },
    {
        "id": "gutierrez_polini",
        "name": "gutierrez_polini",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 2.24,
        "importance_rank": 18,
        "tipo": "Score de legibilidad [0, 100]",
        "formula": "95.2 - 9.7 \\frac{\\text{Chars}}{\\text{Words}} - \\frac{\\text{Words}}{\\text{Sents}}",
        "hipotesis": "Fórmula de legibilidad diseñada específicamente para la estructura silábica del español.",
        "efecto": "Scores anómalos reflejan textos traducidos automáticamente o sintaxis desordenada."
    },
    {
        "id": "flesch_szigriszt",
        "name": "flesch_szigriszt",
        "dim": "D3",
        "dim_name": "D3: Riqueza Léxica y Legibilidad",
        "importance": 1.16,
        "importance_rank": 24,
        "tipo": "Score de perspicuidad [0, 100]",
        "formula": "206.835 - 62.3 \\frac{\\text{Sílabas}}{\\text{Words}} - \\frac{\\text{Words}}{\\text{Sents}}",
        "hipotesis": "Adaptación española de Flesch para evaluar dificultad de comprensión lectora.",
        "efecto": "Diferencia la prosa periodística elaborada de la prosa elemental de bulos virales."
    },

    # D4
    {
        "id": "adv_density",
        "name": "adv_density",
        "dim": "D4",
        "dim_name": "D4: Morfosintaxis y Subjetividad (POS)",
        "importance": 8.41,
        "importance_rank": 1,
        "tipo": "Densidad porcentual [0, 20%]",
        "formula": "\\frac{\\sum \\text{Adverbios (spaCy)}}{\\text{Palabras}} \\times 100",
        "hipotesis": "¡TOP 1 INDIVIDUAL! Los adverbios modales y enfáticos ('brutalmente', 'totalmente', 'muy') transmiten valoración subjetiva en lugar de hechos puros.",
        "efecto": "Densidad elevada (>6.5%) es el predictor morfosintáctico individual más contundente de manipulación."
    },
    {
        "id": "pron_1p_density",
        "name": "pron_1p_density",
        "dim": "D4",
        "dim_name": "D4: Morfosintaxis y Subjetividad (POS)",
        "importance": 3.69,
        "importance_rank": 12,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Pronombres 1.ª persona (yo, nosotros, nos)}}{\\text{Palabras}} \\times 100",
        "hipotesis": "El periodismo serio utiliza 3.ª persona impersonal; los bulos recurren a la 1.ª persona para conectar emocionalmente o infundir miedo colectivo.",
        "efecto": "Presencia de apelaciones en 1.ª persona multiplica la probabilidad de desinformación."
    },
    {
        "id": "adj_noun_ratio",
        "name": "adj_noun_ratio",
        "dim": "D4",
        "dim_name": "D4: Morfosintaxis y Subjetividad (POS)",
        "importance": 1.48,
        "importance_rank": 21,
        "tipo": "Ratio de adjetivación",
        "formula": "\\frac{\\text{Adjetivos}}{\\text{Sustantivos} + 1}",
        "hipotesis": "Sobrecarga de adjetivos calificativos para moldear el juicio del lector antes que reportar los hechos.",
        "efecto": "Ratios >0.40 evidencian estilo retórico y opinativo."
    },
    {
        "id": "pron_3p_density",
        "name": "pron_3p_density",
        "dim": "D4",
        "dim_name": "D4: Morfosintaxis y Subjetividad (POS)",
        "importance": 1.22,
        "importance_rank": 25,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Pronombres 3.ª persona}}{\\text{Palabras}} \\times 100",
        "hipotesis": "Control de estilo formal e impersonal característico de crónicas informativas estándar.",
        "efecto": "Normalidad de la estructura referencial del texto."
    },

    # D5
    {
        "id": "upper_chars_ratio",
        "name": "upper_chars_ratio",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 7.49,
        "importance_rank": 2,
        "tipo": "Ratio de caracteres mayúsculos [0, 1]",
        "formula": "\\frac{\\sum \\text{Letras mayúsculas}}{\\text{Total caracteres alfabéticos}}",
        "hipotesis": "¡TOP 2 INDIVIDUAL! Refleja el 'grito digital' y la urgencia inducida mediante tipografía alterada.",
        "efecto": "Valores superiores al 5.0% son rarísimos en prensa formal y predominan en bulos virales."
    },
    {
        "id": "all_caps_ratio",
        "name": "all_caps_ratio",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 6.96,
        "importance_rank": 3,
        "tipo": "Porcentaje de palabras [0, 100]",
        "formula": "\\frac{\\text{Palabras en MAYÚSCULAS sostenidas}}{\\text{Total Palabras}} \\times 100",
        "hipotesis": "¡TOP 3 INDIVIDUAL! Palabras completas en mayúsculas ('¡¡URGENTE!!', 'ALERTA', 'DIFUNDIR').",
        "efecto": "Aporta casi un 7% del poder explicativo por sí sola; rasgo invariante a fronteras geográficas."
    },
    {
        "id": "numbers_density",
        "name": "numbers_density",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 4.04,
        "importance_rank": 11,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Tokens numéricos}}{\\text{Palabras}} \\times 100",
        "hipotesis": "El anclaje cuantitativo (precios, balances, porcentajes, censos) abunda en noticias legítimas.",
        "efecto": "Ausencia absoluta de cifras verificables delata vaguedad característica de fábulas de desinformación."
    },
    {
        "id": "temporal_density",
        "name": "temporal_density",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 2.89,
        "importance_rank": 15,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum \\text{Marcadores temporales (meses, años)}}{\\text{Palabras}} \\times 100",
        "hipotesis": "Las noticias reales se anclan temporalmente en fechas precisas; los bulos usan vaguedades atemporales ('ayer', 'hace poco').",
        "efecto": "Falta de fechas exactas incrementa la probabilidad de engaño."
    },
    {
        "id": "punct_intensity",
        "name": "punct_intensity",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.77,
        "importance_rank": 29,
        "tipo": "Intensidad por 100 palabras",
        "formula": "\\frac{\\sum (!, ?, \\dots)}{\\text{Palabras}} \\times 100",
        "hipotesis": "Carga agregada de signos de puntuación expresivos no neutros.",
        "efecto": "El periodismo objetivo evita la saturación de signos exclamativos y suspensivos."
    },
    {
        "id": "excl_density",
        "name": "excl_density",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.61,
        "importance_rank": 30,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum (\\text{!} + \\text{¡})}{\\text{Palabras}} \\times 100",
        "hipotesis": "Exclamaciones enfáticas para suscitar alarma o indignación inmediata.",
        "efecto": "Cualquier titular periodístico serio excluye signos de admiración."
    },
    {
        "id": "percent_count",
        "name": "percent_count",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.58,
        "importance_rank": 31,
        "tipo": "Conteo entero",
        "formula": "\\sum \\text{Símbolos '%' o frases 'por ciento'}",
        "hipotesis": "Rigor estadístico verificable presente en periodismo de datos.",
        "efecto": "Disminuye la probabilidad de desinformación."
    },
    {
        "id": "all_caps_count",
        "name": "all_caps_count",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.49,
        "importance_rank": 32,
        "tipo": "Conteo absoluto",
        "formula": "\\sum \\text{Tokens en mayúsculas de } \\ge 3 \\text{ caracteres}",
        "hipotesis": "Volumen bruto de palabras vociferadas en el texto.",
        "efecto": "Señal fuerte de cebo de clics (clickbait)."
    },
    {
        "id": "ellipsis_density",
        "name": "ellipsis_density",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.31,
        "importance_rank": 35,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum (\\dots)}{\\text{Palabras}} \\times 100",
        "hipotesis": "Generación de suspenso artificial o insinuaciones sin sustento.",
        "efecto": "Uso típico de conspiraciones que invitan a 'atar cabos'."
    },
    {
        "id": "quest_density",
        "name": "quest_density",
        "dim": "D5",
        "dim_name": "D5: Anclajes Factuales, Puntuación y Mayúsculas",
        "importance": 0.28,
        "importance_rank": 36,
        "tipo": "Densidad por 100 palabras",
        "formula": "\\frac{\\sum (? + ¿)}{\\text{Palabras}} \\times 100",
        "hipotesis": "Preguntas retóricas que eluden afirmaciones comprobables legalmente.",
        "efecto": "Técnica estándar de tabloides sensacionalistas."
    }
]

with open('/home/ubuntu/Documentos/Tesis/dashboard/diccionario_variables_5d.json', 'w', encoding='utf-8') as f:
    json.dump(variables, f, ensure_ascii=False, indent=2)

print(f"✅ Diccionario generado exitosamente con {len(variables)} variables documentadas.")
