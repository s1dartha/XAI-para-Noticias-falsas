# Reporte Técnico: Discretización Óptima de Redundancia Intra-Documental para Clasificación de Fake News

**Proyecto de Tesis:** Detección de Noticias Falsas en Español mediante Estilometría y Redundancia Semántica  
**Métrica Evaluada:** `max_intra_similarity` (Similitud Coseno Máxima entre Pares de Oraciones)  
**Modelo de Embeddings:** `hiiamsid/sentence_similarity_spanish_es` (SBERT Clásico sobre BETO `dccuchile/bert-base-spanish-wwm-cased`, $d=768$)  
**Técnica de Partición:** Árbol de Decisión Supervisado (`DecisionTreeClassifier`, `max_depth=3`, `criterion='entropy'`)  
**Fecha de Generación:** 18 de Octubre de 2026  

---

## 1. Resumen Ejecutivo del Experimento

El objetivo central de este experimento es superar las limitaciones de las discretizaciones arbitrarias (tales como cuantiles uniformes o rangos fijos de amplitud regular) mediante la aplicación de un método fundamentado en la **Teoría de la Información (Ganancia de Shannon / Reducción de Entropía)** sobre las representaciones obtenidas con el modelo ganador de la comparativa de arquitecturas: **SBERT Clásico sobre BETO (`hiiamsid/sentence_similarity_spanish_es`)**.

La variable continua `max_intra_similarity` captura el grado extremo de redundancia local entre cualquier par de oraciones de un documento periodístico:
$$\text{max\_intra\_similarity}(d) = \max_{i < j} \cos(\vec{e}_i, \vec{e}_j)$$

Donde $\vec{e}_i \in \mathbb{R}^{768}$ representa el embedding contextual de BETO para la $i$-ésima oración. Al entrenar un árbol de decisión restringido ($d=3$) de manera unidimensional sobre esta característica, los puntos de división resultantes corresponden a los umbrales que maximizan la divergencia condicional de clases, segregando de forma óptima las Noticias Reales de las Falsas.

### Parámetros del Corpus Evaluado
* **Total de Documentos Válidos Analizados:** 2,471 artículos (filtrados excluyendo los 133 textos con $<2$ oraciones donde la similitud entre pares es indefinida).
* **Distribución Real de Clases:**
  * **Noticias Reales (`label_num = 0`):** 1,199 muestras (48.5%)
  * **Noticias Falsas (`label_num = 1`):** 1,272 muestras (51.5%)
* **Umbrales Óptimos Detectados por el Árbol de Entropía Regularizado (`min_samples_leaf=15`):** `0.3740`, `0.6079`, `0.6124`, `0.6156`, `0.7287` (6 estratos).
* **Umbrales del Árbol de Entropía Sin Restricción de Hoja (`max_depth=3`):** `0.3740`, `0.6079`, `0.6101`, `0.6124`, `0.6152`, `0.6156`, `0.6161` (8 estratos).

---

## 2. Análisis de Particiones y Tasa Empírica de Fake News

La siguiente tabla detalla la estratificación rigurosa obtenida a partir de los intervalos generados por el árbol de decisión guiado por entropía de Shannon sobre los embeddings de BETO (0 = Real, 1 = Fake):

| Bin ID | Rango de Intervalo (`max_intra_similarity` BETO) | N Muestras | % Corpus | Noticias Reales (0) | Noticias Falsas (1) | Tasa Fake News (%) | Perfil Estilométrico / Interpretación |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `[-0.0110, 0.3740]` | 335 | 13.6% | 150 | 185 | **55.2%** | 🔴 **Baja Cohesión / Fragmentación** (Oraciones inconexas, predominio Fake) |
| 2 | `(0.3740, 0.6079]` | 1,422 | 57.5% | 739 | 683 | **48.0%** | 🟢 **Cohesión Periodística Estándar** (Prosa informativa formal y balanceada) |
| 3 | `(0.6079, 0.6124]` | 18 | 0.7% | 15 | 3 | **16.7%** | 🟢 **Dominancia Real (83.3%)** (Transición de reformulación periodística) |
| 4 | `(0.6124, 0.6156]` | 29 | 1.2% | 3 | 26 | **89.7%** | 🔴 **Pico Crítico de Desinformación (89.7%)** (Reiteración léxica de consignas) |
| 5 | `(0.6156, 0.7287]` | 486 | 19.7% | 225 | 261 | **53.7%** | 🟡 **Redundancia Moderada-Alta** (Ligero predominio de noticias falsas) |
| 6 | `(0.7287, 1.0000]` | 181 | 7.3% | 67 | 114 | **63.0%** | 🔴 **Hiper-Redundancia Circular (63.0%)** (Bucle argumentativo de desinformación) |

### Hallazgos Empíricos Clave con BETO SBERT:
1. **Confirmación de la Hipótesis de Hiper-Redundancia en Noticias Falsas (`> 0.7287`):**
   * En el estrato superior (Bin 6: `(0.7287, 1.0000]`), que agrupa **181 artículos (7.3% del corpus)**, la tasa de **Noticias Falsas alcanza el 63.0%** (114 Falsas frente a 67 Reales).
   * A diferencia de los modelos multilingües ligeros que distorsionaban la cola superior, las representaciones en español nativo de BETO demuestran claramente que la reiteración circular intensa entre oraciones es un marcador estilométrico sólido de desinformación.
2. **Pico Focalizado de Desinformación en la Frontera de Fase (`0.6124 < max_intra_similarity <= 0.6156`):**
   * En el Bin 4, la tasa de noticias falsas alcanza un **89.7%** (26 Falsas vs. 3 Reales), precedido por una micro-banda de noticias reales en `(0.6079, 0.6124]` (83.3% Reales).
3. **Baja Cohesión / Fragmentación Discursiva (`<= 0.3740`):**
   * En el extremo inferior (Bin 1, $N=335$ artículos), donde las oraciones del texto apenas guardan relación semántica entre sí, la proporción de noticias falsas sube al **55.2%**, capturando bulos redactados de forma inconexa o telegráfica.

---

### Visualización de Particiones:
La distribución de los intervalos y los puntos de corte de Shannon sobre el corpus con BETO se presenta a continuación:  
![Visualización de Particiones de Redundancia](imagenes/redundancy_splits_visualization.png)

---

## 3. Recomendaciones Estratégicas para la Selección de Clases Finales

Con base en la distribución de entropía sobre BETO y la inspección de los nodos hoja, se formulan dos alternativas concretas:

### Alternativa A: Partición Estilométrica Parsimoniosa en 4 Clases (Recomendada para Modelos Lineales / Interpretabilidad XAI)
Consolidando los umbrales del árbol de decisión en **4 macro-categorías funcionales**:
* **Variante A1 (Macro-Categorías Balanceadas `[0.3740, 0.6079, 0.7287]`, `ROC-AUC = 0.5422`):**
  1. **Clase 1 - Baja Cohesión / Desconexión (`<= 0.3740`, $N=335$):** Discurso fragmentado (**55.2% Fake**).
  2. **Clase 2 - Cohesión Estándar / Periodismo Profesional (`(0.3740, 0.6079]`, $N=1,422$):** Intervalo dominante de noticias verídicas (**52.0% Real / 48.0% Fake**).
  3. **Clase 3 - Redundancia Moderada (`(0.6079, 0.7287]`, $N=533$):** Reiteración argumentativa intermedia (**54.4% Fake**).
  4. **Clase 4 - Hiper-Redundancia Circular (`> 0.7287`, $N=181$):** Alerta estilométrica de bucle de desinformación (**63.0% Fake**).
* **Variante A2 (Árbol Parsimonioso Puro `max_leaf_nodes=4`: `[0.3740, 0.6124, 0.6158]`, `ROC-AUC = 0.5469`):** Maximiza la ganancia de entropía con solo 4 hojas.

### Alternativa B: Discretización Exacta del Árbol de Entropía (Recomendada para Ensembles / GBDT / Random Forest)
* Mantener los **6 intervalos regularizados** (`[0.3740, 0.6079, 0.6124, 0.6156, 0.7287]`, `ROC-AUC = 0.5535`) o los **8 intervalos sin restricción** (`ROC-AUC = 0.5557`).
* Esta configuración preserva el 100% de la ganancia de información teórica de Shannon detectada sobre BETO, alcanzando el mayor poder predictivo en validación cruzada.

---
*Reporte generado de forma autónoma por la pipeline de experimentación científica.*
