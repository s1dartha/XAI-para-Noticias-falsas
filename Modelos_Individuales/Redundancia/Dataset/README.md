# 📁 Datasets del Módulo de Redundancia Semántica

Este directorio contiene los conjuntos de datos y matrices de características utilizados en los experimentos, evaluaciones y reportes científicos del módulo de **Redundancia Semántica** para la detección de noticias falsas (*Fake News*).

---

## 📊 Catálogo de Datos

| Archivo | Tipo / Formato | Muestras / Filas | Columnas Clave | Descripción |
| :--- | :---: | :---: | :--- | :--- |
| **[`dataset_con_similitudes.csv`](dataset_con_similitudes.csv)** | CSV (`UTF-8`) | 2,604 artículos (2,471 válidos con $\ge 2$ oraciones) | `class`, `Text`, `num_sentences`, `num_pairs`, `pairwise_similarities`, `mean_intra_similarity`, `max_intra_similarity`, `p90_intra_similarity`, `var_intra_similarity`, `has_redundancy`, `redundant_pairs_count` | Dataset principal procesado con similitudes coseno intra-documentales calculadas entre todos los pares oracionales de cada artículo mediante el modelo ganador **SBERT Clásico sobre BETO** (`hiiamsid/sentence_similarity_spanish_es`, $d=768$). Base para la determinación del umbral ($\tau = 0.34$), discretización supervisada de Shannon y experimentos XAI. |
| **[`metricas_redundancia_dataset_completo.csv`](metricas_redundancia_dataset_completo.csv)** | CSV (`UTF-8`) | 2,604 registros | `Etiqueta`, `SBERT_*`, `CNN-SBERT_*`, `BLSTM-SBERT_*`, `Attention-SBERT_*` | Matriz consolidada de agregaciones estadísticas (`mean`, `max`, `p90`, `var`) para las 4 arquitecturas siamesas evaluadas en el benchmark comparativo de selección de modelos. |
| **[`Noticias_entre_70_y_370_palabras (1).xlsx`](Noticias_entre_70_y_370_palabras%20(1).xlsx)** | Excel (`.xlsx`) | 2,604 artículos | `class`, `Text`, `Clean_Text`, `Title` | Corpus crudo de noticias en español filtrado por rango de longitud (70 a 370 palabras) a partir de fuentes verificadas y repositorios de desinformación. |

---

## 🔬 Relación con los Reportes Técnicos

- **[`../Reportes/Seleccion_Modelos/Reporte_Redundancia.md`](../Reportes/Seleccion_Modelos/Reporte_Redundancia.md):** Utiliza `metricas_redundancia_dataset_completo.csv` para contrastar la correlación de características y separabilidad entre las 4 arquitecturas siamesas, coronando a **SBERT Clásico (`hiiamsid/sentence_similarity_spanish_es`)** como el modelo óptimo.
- **[`../Reportes/Umbral_Redundancia/Reporte_Umbral_Redundancia.md`](../Reportes/Umbral_Redundancia/Reporte_Umbral_Redundancia.md):** Modela la distribución de los 47,693 pares oracionales extraídos de `dataset_con_similitudes.csv` (con BETO SBERT) para calibrar el umbral bayesiano mediante GMM ($\tau = 0.3434 \approx 0.34$).
- **[`../Reportes/Discretizacion_Redundancia/`](../Reportes/Discretizacion_Redundancia/):** Utiliza la columna `max_intra_similarity` de `dataset_con_similitudes.csv` (BETO SBERT) para las particiones de entropía de Shannon y la validación cruzada estratificada de 5 folds.
- **[`../Reportes/XAI/`](../Reportes/XAI/):** Extrae pares oracionales curados y aleatorios directamente de `dataset_con_similitudes.csv` para la batería de explicabilidad e interpretabilidad.
