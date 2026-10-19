# 📁 Datasets del Módulo de Redundancia Semántica

Este directorio contiene los conjuntos de datos y matrices de características utilizados en los experimentos, evaluaciones sin fuga de información y reportes científicos del módulo de **Redundancia Semántica** para la detección de noticias falsas (*Fake News*).

---

## 📊 Catálogo de Datos

| Archivo | Tipo / Formato | Muestras / Filas | Columnas Clave | Descripción |
| :--- | :---: | :---: | :--- | :--- |
| **[`dataset_con_similitudes.csv`](dataset_con_similitudes.csv)** | CSV (`UTF-8`) | 2,604 artículos (2,471 válidos con $\ge 2$ oraciones; 2,152 textos únicos deduplicados) | `class`, `Text`, `num_sentences`, `num_pairs`, `pairwise_similarities`, `mean_intra_similarity`, `max_intra_similarity`, `second_max_intra_similarity`, `top3_mean_intra_similarity`, `p90_intra_similarity`, `var_intra_similarity`, `has_redundancy`, `redundant_pairs_count`, `has_paraphrase_p90`, `paraphrase_pairs_count_p90`, `has_strict_redundancy_p95`, `strict_redundant_pairs_count_p95`, `prop_strict_redundant_p95`, `is_unique_text` | Dataset principal procesado con similitudes coseno intra-documentales calculadas entre todos los pares oracionales (47,693 pares) mediante el modelo unificado **SBERT Clásico sobre BETO** (`hiiamsid/sentence_similarity_spanish_es`, $d=768$). Incluye indicadores explícitos para continuidad temática ($\tau=0.3434$), paráfrasis validada en PAWS-X ($\tau=0.5340$), redundancia estricta ($\tau=0.5996$), segundo estadístico de orden $S_{(2)}$ y bandera de deduplicación (`is_unique_text`). |
| **[`metricas_redundancia_dataset_completo.csv`](metricas_redundancia_dataset_completo.csv)** | CSV (`UTF-8`) | 2,604 registros | `Etiqueta`, `SBERT_*`, `CNN-SBERT_*`, `BLSTM-SBERT_*`, `Attention-SBERT_*` | Matriz consolidada de agregaciones estadísticas (`mean`, `max`, `p90`, `var`) para las 4 arquitecturas siamesas evaluadas en el benchmark comparativo de selección de modelos. |
| **[`Noticias_entre_70_y_370_palabras (1).xlsx`](Noticias_entre_70_y_370_palabras%20(1).xlsx)** | Excel (`.xlsx`) | 2,604 artículos (1,259 Reales [0] y 1,345 Falsas [1]) | `class`, `Text`, `Clean_Text`, `Title` | Corpus crudo de noticias en español filtrado por rango de longitud (70 a 370 palabras) a partir de fuentes verificadas y repositorios de desinformación. |

---

## 🔬 Relación con los Reportes Técnicos

- **[`../Reportes/Seleccion_Modelos/Reporte_Redundancia.md`](../Reportes/Seleccion_Modelos/Reporte_Redundancia.md):** Utiliza `metricas_redundancia_dataset_completo.csv` para contrastar la correlación de características y separabilidad entre las 4 arquitecturas siamesas, seleccionando a **SBERT Clásico (`hiiamsid/sentence_similarity_spanish_es`)** como el modelo óptimo.
- **[`../Reportes/Umbral_Redundancia/Reporte_Umbral_Redundancia.md`](../Reportes/Umbral_Redundancia/Reporte_Umbral_Redundancia.md):** Evalúa la jerarquía de umbrales ($\tau_{\text{GMM}} = 0.3434$ continuidad temática, $\tau_{\text{PAWS-X}} = 0.5340$ paráfrasis moderada, $\tau_{P_{95}} = 0.5996$ redundancia estricta, y $\tau_{\text{Hiper}} = 0.7287$ hiper-redundancia circular) mediante análisis de sensibilidad multi-umbral y pruebas estadísticas a nivel de documento con intervalos de confianza Bootstrap al 95%.
- **[`../Reportes/Umbral_Redundancia/reporte_explicabilidad_mexgen_redundancia.md`](../Reportes/Umbral_Redundancia/reporte_explicabilidad_mexgen_redundancia.md):** Ejecuta las pruebas de ablación oracional no tautológica sobre $\bar{S}_{\text{doc}}$ y analiza la topología del segundo estadístico de orden $S_{(2)}$ (bucle único vs. red multi-bucle) sobre BETO.
- **[`../Reportes/Discretizacion_Redundancia/`](../Reportes/Discretizacion_Redundancia/):** Evalúa las particiones de `max_intra_similarity` bajo validación cruzada estricta sin fuga de información (`In-Fold`) y control de textos duplicados (`StratifiedGroupKFold`).
- **[`../Reportes/XAI/`](../Reportes/XAI/):** Extrae pares oracionales curados y aleatorios directamente de `dataset_con_similitudes.csv` evaluados de forma unificada con `hiiamsid/sentence_similarity_spanish_es`.
