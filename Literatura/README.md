# 📚 Acervo Bibliográfico y Marco Teórico de la Tesis

Este directorio compila y organiza las fuentes académicas, artículos científicos indexados, revisiones sistemáticas y marcos conceptuales utilizados para la concepción, fundamentación metodológica y desarrollo del proyecto de grado: **"Inteligencia Artificial Explicable (XAI) en Detección de Noticias Falsas y Sensacionalismo en Español"**.

---

## 🗂️ Estructura del Directorio

```text
Literatura/
├── README.md                                          # Catálogo temático y estado del arte
│
├── Antecedentes/                                      # Trabajos previos, benchmarks y antecedentes
│   ├── Andes arbol de decision fake news.pdf
│   ├── deteccion de fake news colombia 2022.pdf
│   ├── Deteccion de Noticias Falsas.pdf
│   ├── Exploiting Content Characteristics for Explainable Detection of Fake News.pdf
│   ├── Performance_Analysis_of_Transformer_Based_Models_B.pdf
│   └── Sensational stories narrative characterstics.pdf
│
├── Explicabilidad_No/                                 # Posturas críticas y fundamentos teóricos de XAI
│   ├── Imposibilitty theorems.pdf
│   └── Stop explaining.pdf (Cynthia Rudin - Black Box Models)
│
├── Introducción/                                      # Contexto, infodemia y revisiones sistemáticas
│   ├── deep_learning_fake_news.pdf
│   ├── fake news review.pdf
│   ├── Fakecracia_ la desinfodemia de las noticias falsas en América Latina.pdf
│   ├── impacto de fake news latinoamerica.pdf
│   └── roozenbeek-et-al-2023-countering-misinformation.pdf
│
└── patrones o caracteristicas/                        # Análisis estilométrico y variables de modelado
    ├── Redundancia/                                   # Modelos siameses, similitud semántica y reiteración
    │   ├── Sci-Net_ Siamese BERT Architecture Model with attention mechanism...pdf
    │   ├── Sentence representations for semantic textual similarity_ A systematic review.pdf
    │   ├── caracteristicas de fake news.pdf
    │   ├── DEFINING "FAKE NEWS" A typology of scholarly definitions.pdf
    │   ├── Fake_news_repetitive_content.pdf
    │   ├── impacto de fake news latinoamerica.pdf
    │   ├── Investigación sobre desinformación y fake news en revistas de comunicación...pdf
    │   └── Populismo, desinformación y polarización política en la comunicación...pdf
    │
    └── Sensacionalismo/                               # Volatilidad emocional, amarillismo y sintaxis
        ├── Hamby_2023_Volatilidad_Emocional_y_Tragedia.pdf
        ├── Horne_2017_Titulares_Cargados_y_Cuerpo_Simple.pdf
        ├── Munoz_2024_Alta_Excitacion_y_Diversidad_Lexica.pdf
        ├── pseudo caracteristicas.pdf
        └── Roozenbeek_2023_Sesgos_Cognitivos_y_Juicio_Critico.pdf
```

---

## 📋 Mapeo de Literatura por Temática de Investigación

| Área de Investigación | Subdirectorio | Documentos Clave | Aporte al Proyecto de Tesis |
| :--- | :--- | :--- | :--- |
| **Antecedentes & Modelos Base** | [`Antecedentes/`](Antecedentes/) | *Exploiting Content Characteristics...*<br>*Performance Analysis of Transformer Models*<br>*Andes árbol de decisión* | Establece la línea base de arquitecturas Transformer (BETO, RoBERTa) y árboles de decisión en contextos de noticias en español y Colombia. |
| **Fundamentos de XAI & Explicabilidad** | [`Explicabilidad_No/`](Explicabilidad_No/) | *Stop Explaining Black Box Machine Learning Models* (C. Rudin)<br>*Impossibility Theorems in XAI* | Justifica por qué las explicaciones post-hoc (LIME/SHAP) deben complementarse con interpretabilidad mecanicista y auditorías formales de fidelidad (Adebayo, MoRF/LoRF). |
| **Contexto e Infodemia** | [`Introducción/`](Introducción/) | *Fakecracia: la desinfodemia en AL*<br>*Deep Learning Fake News Review*<br>*Countering Misinformation* (Roozenbeek) | Provee la contextualización sociopolítica del impacto de la desinformación en medios hispanohablantes y los fundamentos de detección automatizada. |
| **Redundancia Semántica** | [`patrones o caracteristicas/Redundancia/`](patrones%20o%20caracteristicas/Redundancia/) | *Sentence Representations for STS: Systematic Review*<br>*Sci-Net Siamese BERT*<br>*Fake News Repetitive Content* | Respalda metodológicamente el uso de Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`), cálculo de umbrales con GMM y la hipótesis de reiteración argumental. |
| **Sensacionalismo & Estilometría** | [`patrones o caracteristicas/Sensacionalismo/`](patrones%20o%20caracteristicas/Sensacionalismo/) | *Hamby (2023) Volatilidad Emocional*<br>*Horne (2017) Titulares Cargados*<br>*Muñoz (2024) Alta Excitación y Diversidad Léxica* | Fundamenta las métricas de carga emocional, volatilidad entre párrafos y disociación frase-documento en el módulo de Sensacionalismo. |
