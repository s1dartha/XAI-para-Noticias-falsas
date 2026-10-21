# 📊 Estructura y Mapa de Trazabilidad de Reportes e Imágenes: Redundancia Semántica

Este directorio consolida todos los reportes formales, métricas numéricas, conjuntos de datos experimentales y artefactos visuales correspondientes al módulo de **Redundancia Semántica** dentro del proyecto de investigación y desarrollo de Inteligencia Artificial Explicable (XAI) para la detección de noticias falsas, unificados íntegramente bajo el modelo **SBERT Clásico sobre BETO (`hiiamsid/sentence_similarity_spanish_es`, $d=768$)**.

---

## 📁 1. Organización Temática de Reportes

Los reportes se encuentran categorizados y agrupados según su área técnica de investigación:

```text
Modelos_Individuales/Redundancia/
├── Dataset/                                           # Datasets del módulo (similitudes BETO, métricas, corpus crudo)
│   ├── dataset_con_similitudes.csv
│   ├── metricas_redundancia_dataset_completo.csv
│   ├── Noticias_entre_70_y_370_palabras (1).xlsx
│   └── README.md
│
└── Reportes/
    ├── README.md                                      # Mapa de trazabilidad y catálogo general
    │
    ├── Seleccion_Modelos/                             # 1. Selección y Comparativa de Arquitecturas SBERT
    │   ├── Reporte_Redundancia.md                     # Evaluación exhaustiva de 4 modelos siameses
    │   ├── full_metrics_evaluation.json               # Métricas cuantitativas (STS-B, PAWS-X, Corpus)
    │   └── imagenes/                                  # 6 Gráficos de benchmarks y correlación
    │       ├── benchmark_performance_sts.png
    │       ├── model_selection_summary.png
    │       ├── paws_x_performance.png
    │       ├── redundancy_correlation_matrix.png
    │       ├── redundancy_feature_distributions.png
    │       └── xnli_model_comparison.png
    │
    ├── Umbral_Redundancia/                            # 2. Jerarquía de Umbrales, Sensibilidad y Ablación Oracional
    │   ├── Reporte_Umbral_Redundancia.md              # Sensibilidad multi-umbral (0.34, 0.53, 0.60, 0.73) y pruebas por doc
    │   ├── reporte_explicabilidad_mexgen_redundancia.md # Ablación oracional no tautológica y topología S_(2)
    │   ├── validacion_explicabilidad_mexgen.csv       # Métricas por documento de ablación oracional (N=1,820)
    │   └── imagenes/                                  # Gráficos de distribución, sensibilidad y ablación
    │       ├── distribucion_similitud_corpus.png
    │       └── validacion_explicabilidad_mexgen.png
    │
    ├── Discretizacion_Redundancia/                    # 3. Estratificación y Discretización Sin Fuga (In-Fold)
    │   ├── reporte_clases_redundancia.md              # Cortes In-Fold, control de duplicados e IC Bootstrap 95%
    │   ├── reporte_comparativo_estrategias.md         # Evaluación 5-fold In-Fold y StratifiedGroupKFold
    │   └── imagenes/                                  # Gráficos de curvas ROC-AUC In-Fold y particiones
    │       ├── metricas_estrategias_comparacion.png
    │       └── redundancy_splits_visualization.png
    │
    └── XAI/                                           # 4. Inteligencia Artificial Explicable (Unificada en BETO)
        ├── Reporte_XAI_Redundancia.md                 # Batería XAI completa sobre 10 pares curados (BETO)
        ├── Reporte_XAI_Aleatorio.md                   # Validación de robustez en 10 pares aleatorios (BETO)
        ├── xai_experimental_results.json              # Resultados numéricos de fidelidad y latencia
        └── imagenes/                                  # 14 Gráficos: Heatmaps, MoRF/LoRF, Adebayo, Latencia
            ├── xai_cascading_parameter_randomization_sanity_check.png
            ├── xai_cascading_parameter_randomization_sanity_check_aleatorio.png
            ├── xai_cross_attention_alignment.png
            ├── xai_cross_attention_alignment_aleatorio.png
            ├── xai_faithfulness_comprehensiveness_sufficiency.png
            ├── xai_faithfulness_comprehensiveness_sufficiency_aleatorio.png
            ├── xai_method_execution_latency.png
            ├── xai_method_execution_latency_aleatorio.png
            ├── xai_morf_vs_lorf_ablation_curves.png
            ├── xai_morf_vs_lorf_ablation_curves_aleatorio.png
            ├── xai_token_attribution_heatmaps_non_redundant.png
            ├── xai_token_attribution_heatmaps_non_redundant_aleatorio.png
            ├── xai_token_attribution_heatmaps_redundant.png
            └── xai_token_attribution_heatmaps_redundant_aleatorio.png
```

---

## 📋 2. Mapa General de Trazabilidad

| Carpeta Temática | Reporte / Archivo | Tipo | Datasets de Entrada | Temática | Descripción Científica |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`Seleccion_Modelos/`** | [`Reporte_Redundancia.md`](Seleccion_Modelos/Reporte_Redundancia.md) | Reporte `.md` | [`metricas_redundancia_dataset_completo.csv`](../Dataset/metricas_redundancia_dataset_completo.csv)<br>STS-B & PAWS-X | Selección de Modelos | Comparativa rigurosa de 4 arquitecturas siamesas SBERT (Clásico Mean-Pooling, CNN, BLSTM, Attention) en STS-B, PAWS-X y corpus de tesis. |
| | [`full_metrics_evaluation.json`](Seleccion_Modelos/full_metrics_evaluation.json) | Métricas `.json` | Benchmarks estándar | Métricas Base | Registro numérico completo de correlaciones Pearson/Spearman, MSE y exactitud en PAWS-X. |
| **`Umbral_Redundancia/`** | [`Reporte_Umbral_Redundancia.md`](Umbral_Redundancia/Reporte_Umbral_Redundancia.md) | Reporte `.md` | [`dataset_con_similitudes.csv`](../Dataset/dataset_con_similitudes.csv) | Jerarquía de Umbrales y Sensibilidad | Distinción empírica entre continuidad temática ($\tau=0.34$), paráfrasis PAWS-X ($\tau=0.53$), redundancia estricta ($\tau=0.60$) e hiper-redundancia ($\tau=0.73$) con pruebas a nivel de documento. |
| | [`reporte_explicabilidad_mexgen_redundancia.md`](Umbral_Redundancia/reporte_explicabilidad_mexgen_redundancia.md) | Reporte `.md` | [`validacion_explicabilidad_mexgen.csv`](Umbral_Redundancia/validacion_explicabilidad_mexgen.csv) | Ablación Oracional No Tautológica | Prueba causal sobre similitud media global $\bar{S}_{\text{doc}}$ y análisis del segundo estadístico de orden $S_{(2)}$ (bucle único vs. persistencia multi-bucle). |
| **`Discretizacion_Redundancia/`** | [`reporte_clases_redundancia.md`](Discretizacion_Redundancia/reporte_clases_redundancia.md) | Reporte `.md` | [`dataset_con_similitudes.csv`](../Dataset/dataset_con_similitudes.csv) (2,471 válidos / 2,152 únicos) | Discretización Sin Fuga | Análisis de estabilidad de cortes `In-Fold`, intervalos de confianza Bootstrap al 95% y control de duplicados. |
| | [`reporte_comparativo_estrategias.md`](Discretizacion_Redundancia/reporte_comparativo_estrategias.md) | Reporte `.md` | [`dataset_con_similitudes.csv`](../Dataset/dataset_con_similitudes.csv) | Validación Cruzada `In-Fold` y `GroupKFold` | Comparación libre de fugas (`In-Fold`) y deduplicada por texto (`StratifiedGroupKFold`) entre variable continua, umbrales GMM/$P_{95}$, cuartiles y árboles de Shannon. |
| **`XAI/`** | [`Reporte_XAI_Redundancia.md`](XAI/Reporte_XAI_Redundancia.md) | Reporte `.md` | [`dataset_con_similitudes.csv`](../Dataset/dataset_con_similitudes.csv) (10 pares curados) | Interpretabilidad Curada (BETO) | Batería completa de explicabilidad: Fast-IG, Input×Grad, Saliency, LIME, KernelSHAP, fidelidad (Comprehensiveness/Sufficiency), MoRF/LoRF y Sanity Check de Adebayo. |
| | [`Reporte_XAI_Aleatorio.md`](XAI/Reporte_XAI_Aleatorio.md) | Reporte `.md` | [`dataset_con_similitudes.csv`](../Dataset/dataset_con_similitudes.csv) (10 pares aleatorios) | Robustez XAI (BETO) | Validación de estabilidad y transferibilidad de la suite XAI sobre `hiiamsid/sentence_similarity_spanish_es` en pares oracionales seleccionados al azar. |
| | [`xai_experimental_results.json`](XAI/xai_experimental_results.json) | Métricas `.json` | Suite experimental XAI | Métricas XAI | Resultados cuantitativos de comprensividad, suficiencia, correlación inter-métodos y tiempos de latencia por método explicativo. |
