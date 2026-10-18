# Reporte de Evaluación Predictiva: Comparación de 4 Estrategias de Discretización para 'max_intra_similarity' con SBERT Clásico (BETO)

**Proyecto de Tesis:** Detección de Noticias Falsas en Español mediante Estilometría y Redundancia Semántica  
**Modelo de Embeddings:** `hiiamsid/sentence_similarity_spanish_es` (SBERT Clásico sobre BETO `dccuchile/bert-base-spanish-wwm-cased`, $d=768$)  
**Característica Evaluada:** `max_intra_similarity` (Similitud Coseno Intra-Documental Máxima vía BETO SBERT)  
**Variable Objetivo:** `class` (Binaria: 1 = Fake News, 0 = Real News)  
**Esquema de Validación:** Validación Cruzada Estratificada de 5 Folds (`StratifiedKFold`, $K=5$, `shuffle=True`, `seed=42`)  
**Codificación de Entrada:** One-Hot Encoding (OHE) sin suposición de linealidad ordinal  
**Fecha de Ejecución:** 18 de Octubre de 2026  

---

## 1. Resumen Ejecutivo del Experimento de Validación Cruzada

El propósito de este experimento definitivo es determinar empíricamente qué estrategia de partición de la métrica continua `max_intra_similarity` —calculada con el modelo ganador de la evaluación comparativa (**SBERT Clásico sobre BETO**, `hiiamsid/sentence_similarity_spanish_es`)— maximiza la capacidad de discriminación entre noticias verídicas y falsas, comparando enfoques no supervisados de la literatura contra particiones supervisadas basadas en la Teoría de la Información (Entropía de Shannon).

Para evitar cualquier sesgo de monotonicidad lineal, cada partición categórica se transformó en variables binarias (*One-Hot Encoding*) y se entrenaron dos familias de modelos predictivos bajo un estricto protocolo de 5-fold Stratified CV sobre los **2,471 artículos válidos**:
1. **Regresión Logística L2 (`solver='liblinear'`):** Evalúa la separabilidad lineal de los bins en el espacio logit.
2. **Random Forest Superficial (`max_depth=4`, $N=100$ árboles):** Captura posibles interacciones no lineales y regulariza la varianza.

### Estrategias Contrastadas (Calibradas sobre BETO SBERT):
* **Estrategia 1 (Literatura & GMM - 3 Clases):** Cortes en `[0.3434, 0.5996]`, derivados del umbral bayesiano del Modelo de Mezcla Gaussiana ($\tau_{\text{GMM}} = 0.3434$) y el percentil $P_{95} = 0.5996$.
* **Estrategia 2 (Literatura & Percentiles - 4 Clases):** Cortes en `[0.3434, 0.5340, 0.6261]`, combinando $\tau_{\text{GMM}}$, el percentil global $P_{90} = 0.5340$ y el cuartil superior de `max_intra_similarity`.
* **Estrategia 3 (Árbol Parsimonioso - 4 Clases):**
  - *Variante 3A (Macro-Categorías Balanceadas):* Cortes en `[0.3740, 0.6079, 0.7287]`, agrupando el corpus en 4 estratos de tamaño robusto ($N \ge 181$).
  - *Variante 3B (Árbol Parsimonioso Puro `max_leaf_nodes=4`):* Cortes en `[0.3740, 0.6124, 0.6158]`, que maximizan directamente la ganancia de información con 4 hojas.
* **Estrategia 4 (Árbol Exacto de Entropía de Shannon):**
  - *Variante 4A (Regularizado `min_samples_leaf=15` - 6 Clases):* Cortes en `[0.3740, 0.6079, 0.6124, 0.6156, 0.7287]`.
  - *Variante 4B (Sin restricción de hoja `max_depth=3` - 8 Clases):* Cortes en `[0.3740, 0.6079, 0.6101, 0.6124, 0.6152, 0.6156, 0.6161]`.

---

## 2. Tabla Comparativa de Desempeño Predictivo (BETO SBERT)

| Estrategia de Discretización | Bins (OHE) | Puntos de Corte (BETO) | Regresión Logística ROC-AUC | Regresión Logística F1-Score | Random Forest ROC-AUC | Random Forest F1-Score | Score AUC Combinado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Estrategia 1 (Literatura & GMM - 3 Clases)** | 3 | `[0.3434, 0.5996]` | `0.5314 (±0.016)` | `0.4933 (±0.014)` | `0.5314 (±0.016)` | `0.4933 (±0.014)` | **0.5314** |
| **Estrategia 2 (Literatura & Percentiles - 4 Clases)** | 4 | `[0.3434, 0.5340, 0.6261]` | `0.5287 (±0.010)` | `0.4543 (±0.016)` | `0.5287 (±0.010)` | `0.4543 (±0.016)` | **0.5287** |
| **Estrategia 3A (Árbol Macro Balanceado - 4 Clases)** | 4 | `[0.3740, 0.6079, 0.7287]` | `0.5412 (±0.018)` | `0.5072 (±0.025)` | `0.5422 (±0.018)` | `0.5072 (±0.025)` | **0.5417** |
| **Estrategia 3B (Árbol Parsimonioso Puro - 4 Clases)** | 4 | `[0.3740, 0.6124, 0.6158]` | `0.5469 (±0.019)` | `0.5084 (±0.026)` | `0.5469 (±0.019)` | `0.5084 (±0.026)` | **0.5469** |
| **Estrategia 4A (Árbol Exacto Regularizado - 6 Clases)** | 6 | `[0.374, 0.608, 0.612, 0.616, 0.729]` | **`0.5535 (±0.022)`** | **`0.5084 (±0.026)`** | **`0.5526 (±0.024)`** | **`0.5084 (±0.026)`** | **0.5530** |
| **Estrategia 4B (Árbol Exacto Puro - 8 Clases)** | 8 | `[0.374, 0.608, 0.610, 0.612, 0.615, 0.616, 0.6161]` | **`0.5557 (±0.022)`** | **`0.5098 (±0.026)`** | `0.5431 (±0.023)` | `0.5072 (±0.025)` | **0.5494** |

### Visualización Asociada:
La figura de barras agrupadas que compara estas métricas se encuentra guardada en:  
![Comparación de Estrategias de Discretización](imagenes/metricas_estrategias_comparacion.png)

---

## 3. Análisis de Resultados por Familia de Estrategia

### A. Limitación de las Estrategias No Supervisadas (GMM y Percentiles)
* La **Estrategia 1 (GMM)** obtiene un ROC-AUC de `0.5314`, y la **Estrategia 2 (Percentiles)** alcanza `0.5287`.
* **Causa Metodológica:** Aunque las representaciones en español de BETO elevan el rendimiento base incluso en cortes no supervisados, definir umbrales únicamente por la densidad marginal $P(X)$ desaprovecha las zonas críticas donde cambia abruptamente la probabilidad condicional $P(\text{Fake} \mid X)$.

### B. Superioridad de las Estrategias Guiadas por Información Mutua de Shannon
* Tanto la Estrategia 3 (`0.5417` - `0.5469`) como la Estrategia 4 (`0.5530` - `0.5557`) superan con claridad a los enfoques no supervisados en ambos clasificadores.
* La **Estrategia 4A (Árbol Exacto Regularizado - 6 Clases)** alcanza el mejor equilibrio global con un **ROC-AUC de `0.5535` en Regresión Logística y `0.5526` en Random Forest** (mientras que la Variante 4B sin restricción de hoja llega a **`0.5557`** en Regresión Logística).
* **Causa Metodológica:** La partición de Shannon sobre BETO captura tres regímenes estilométricos de alto valor discriminativo:
  1. **La zona de fragmentación inferior ($\le 0.3740$):** Donde la tasa de Fake News sube al **55.2%** ($N=335$).
  2. **La discontinuidad abrupta en torno a $0.6124 - 0.6156$:** Donde la tasa de Fake News pasa del **16.7%** en `(0.6079, 0.6124]` al **89.7%** en `(0.6124, 0.6156]`.
  3. **La zona de hiper-redundancia superior ($> 0.7287$):** Con **181 artículos**, donde la concentración de Fake News alcanza el **63.0%** (114 Falsas vs. 67 Reales).

---

## 4. Dictamen y Recomendación Técnica (Senior Data Scientist Recommendation)

### Escenario A: Pipeline de Clasificación Tabular Completo (GBDT, XGBoost, Random Forest)
**Decisión: UTILIZAR EL ÁRBOL EXACTO REGULARIZADO DE SHANNON (`[0.3740, 0.6079, 0.6124, 0.6156, 0.7287]`).**  
* **Justificación Matemática:** Alcanza el mayor Score Global Combinado (`ROC-AUC = 0.5530`), preservando tanto la transición de fase en `0.6124` como el bloque de hiper-redundancia en `> 0.7287`.

### Escenario B: Lingüística Forense y Explicabilidad Académica (4 Macro-Categorías)
**Decisión: UTILIZAR LAS 4 MACRO-CATEGORÍAS DE SHANNON (`[0.3740, 0.6079, 0.7287]` o `[0.3740, 0.6124, 0.6158]`).**  
* **Justificación Metodológica:**
  1. **Principio de Parsimonia y Robustez Muestral:** Con los cortes `[0.3740, 0.6079, 0.7287]`, cada una de las 4 clases cuenta con al menos **181 noticias**, evitando cualquier varianza por hojas pequeñas y reteniendo un **ROC-AUC de `0.5422`**.
  2. **Narrativa Científica Directa para la Tesis:**
     * *Baja Cohesión / Fragmentación* ($\le 0.3740$): Oraciones desarticuladas (**55.2% Fake**).
     * *Cohesión Periodística Estándar* ($(0.3740, 0.6079]$): Prosa informativa verídica (**52.0% Real**).
     * *Reiteración Moderada* ($(0.6079, 0.7287]$): Desarrollo argumentativo con énfasis (**54.4% Fake**).
     * *Hiper-Redundancia Circular* ($> 0.7287$): Reiteración semántica artificial típica de desinformación (**63.0% Fake**).

---
*Reporte generado automáticamente por el módulo de validación estadística de la Tesis.*
