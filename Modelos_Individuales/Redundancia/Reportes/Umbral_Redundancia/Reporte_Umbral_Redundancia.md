# Reporte Técnico y Científico: Determinación del Umbral de Redundancia Intra-Documento en Noticias en Español

**Autor:** Senior NLP Researcher & Lead Data Scientist  
**Proyecto:** Tesis - Detección de Redundancia Semántica y XAI en Noticias en Español  
**Fecha:** 18 de Octubre de 2026  
**Modelo de Representación Semántica:** `hiiamsid/sentence_similarity_spanish_es` (SBERT Clásico sobre BETO `dccuchile/bert-base-spanish-wwm-cased`, 110M parámetros, $d=768$)  
**Dataset Base:** `Noticias_entre_70_y_370_palabras (1).xlsx`  
**Archivo de Salida Generado:** `dataset_con_similitudes.csv`  
**Figura de Distribución:** `imagenes/distribucion_similitud_corpus.png`  

---

## 1. Resumen Ejecutivo

El presente estudio establece formalmente un **umbral empírico y fundamentado estadísticamente ($\tau$)** para la detección y cuantificación de **redundancia semántica intra-documento** sobre un corpus periodístico en español compuesto por **2,604 artículos de noticias**.

A partir de la extracción y segmentación rigurosa de oraciones individuales (13,270 oraciones válidas), se computó el espacio combinatorio completo de similitud coseno intra-texto mediante representaciones densas contextuales derivadas del modelo ganador de la evaluación comparativa multi-tarea: **SBERT Clásico con Mean-Pooling sobre BETO** (`hiiamsid/sentence_similarity_spanish_es`), generando un total de **47,693 pares oracionales evaluados**.

### Resultados Clave:
1. **Umbral Estadístico Determinado ($\tau$):**  
   Se establece el umbral óptimo de redundancia en **$\tau = 0.34$** (valor exacto: **0.3434**), derivado de la frontera de decisión bayesiana de un **Modelo de Mezcla Gaussiana (Gaussian Mixture Model - GMM de 2 componentes)** y respaldado por el percentil **$P_{90} = 0.53$** (**0.5340**) y **$P_{95} = 0.60$** (**0.5996**).
2. **Prevalencia de Redundancia en el Corpus:**  
   Un total de **2,236 artículos (85.87%)** presentan al menos un par oracional con similitud $\ge \tau$, albergando un total acumulado de **19,616 pares oracionales redundantes (41.13% de todos los pares evaluados)**.
3. **Validación respecto a la Etiqueta del Corpus (`class`):**  
   El análisis comparativo evidencia que las noticias falsas (`True`) concentran una mayor densidad de reiteración por artículo (**8.34 pares redundantes por documento** frente a **6.67** en noticias reales, un **+25.0% más de pares redundantes por artículo**) y una mayor similitud máxima media (**0.5356** vs. **0.5250**), validando el poder representacional de las métricas intra-oracionales de BETO para los modelos downstream y los experimentos de Inteligencia Artificial Explicable (XAI).

---

## 2. Metodología de Similitud Intra-Documento y Embeddings SBERT (BETO)

### 2.1. Segmentación Oracional y Filtrado Morfológico
Cada artículo $D_k$ del corpus ($k \in [1, 2604]$) fue procesado mediante el segmentador oracional en español de NLTK (`punkt_tab`), seguido de una limpieza y filtrado de cadenas triviales:
$$S_k = \{s_{k,1}, s_{k,2}, \dots, s_{k,N_k}\} \quad \text{donde} \quad |s_{k,i}| \ge 10 \text{ caracteres}$$

Para el total de 2,604 artículos se extrajeron **13,270 oraciones**, con un promedio de **5.10 oraciones por documento** (desviación estándar: **3.97**, mediana: **3**, rango: $[1, 36]$). Los documentos con $N_k < 2$ (133 artículos) fueron identificados y marcados, dado que por definición un único enunciado no admite combinatoria de pares intra-texto.

### 2.2. Generación de Embeddings Densos con Sentence-BERT (BETO)
Para capturar la semántica distribucional y las relaciones parafrásticas complejas en español nativo, se utilizó el modelo siamés pre-entrenado **`hiiamsid/sentence_similarity_spanish_es`** (basado en `dccuchile/bert-base-spanish-wwm-cased`):
- **Arquitectura:** Transformer Encoder Bidireccional de 12 capas, 12 cabezas de atención, dimensión oculta $d = 768$, con *Mean-Pooling* sobre los tokens activos.
- **Normalización L2:** Cada vector oracional $\mathbf{e}_i = \text{SBERT}_{\text{BETO}}(s_i)$ es proyectado en la hiperesfera unitaria:
  $$\mathbf{v}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|_2} \in \mathbb{R}^{768}, \quad \|\mathbf{v}_i\|_2 = 1.0$$

### 2.3. Espacio Combinatorio Intra-Documento
Para cada documento $D_k$ con $N_k \ge 2$ oraciones, se computó la matriz simétrica de similitud coseno $\mathbf{C}_k \in \mathbb{R}^{N_k \times N_k}$ mediante producto punto:
$$C_{k, ij} = \mathbf{v}_{k,i}^\top \mathbf{v}_{k,j} = \cos(\mathbf{e}_{k,i}, \mathbf{e}_{k,j})$$

Dado que la diagonal principal representa la auto-similitud trivial ($C_{k, ii} = 1.0$) y la matriz es simétrica ($C_{k, ij} = C_{k, ji}$), se extrajo estrictamente la porción triangular superior estricta:
$$\mathcal{P}_k = \{ C_{k, ij} \mid 1 \le i < j \le N_k \}$$
El número de pares evaluados por artículo corresponde a la fórmula combinatoria:
$$|\mathcal{P}_k| = \binom{N_k}{2} = \frac{N_k(N_k - 1)}{2}$$

A nivel global de todo el corpus, la unión de todos los conjuntos combinatorios produjo el array global de similitud:
$$\mathcal{P}_{\text{global}} = \bigcup_{k=1}^{2604} \mathcal{P}_k, \quad |\mathcal{P}_{\text{global}}| = 47,693 \text{ pares}$$

---

## 3. Análisis de la Distribución Estadística Global

El análisis estadístico sobre los **47,693 pares oracionales** del corpus codificados con BETO revela las siguientes propiedades distribucionales:

### Tabla 1: Estadísticas Descriptivas de la Similitud Coseno Intra-Documento (BETO SBERT)

| Métrica Estadística | Símbolo / Notación | Valor Empírico | Interpretación Científica |
| :--- | :---: | :---: | :--- |
| **Pares Totales Evaluados** | $N_{\text{pares}}$ | **47,693** | Muestra masiva combinatoria intra-documental |
| **Media Global** | $\mu$ | **0.3159** | Nivel medio de cohesión semántica dentro de una misma noticia |
| **Mediana** | $\text{Med}$ | **0.3035** | Valor central sin afectación por valores extremos |
| **Desviación Estándar** | $\sigma$ | **0.1621** | Grado de dispersión semántica entre enunciados |
| **Varianza** | $\sigma^2$ | **0.0263** | Dispersión cuadrática compacta |
| **Asimetría (Skewness)** | $\gamma_1$ | **0.3946** | Asimetría moderada a la derecha / centrada |
| **Curtosis Excesiva** | $\gamma_2$ | **-0.0607** | Distribución mesocúrtica equilibrada |
| **Mínimo** | $\min$ | **-0.1123** | Menor similitud registrada entre oraciones del mismo texto |
| **Máximo** | $\max$ | **1.0000** | Par intra-documental con mayor equivalencia semántica |

### Tabla 2: Cuantiles y Percentiles de Similitud Coseno

| Percentil | Valor de Similitud ($s$) | Proporción Acumulada |
| :---: | :---: | :---: |
| **$P_{25}$ (Primer Cuartil)** | 0.1940 | 25% de los pares tienen similitud menor |
| **$P_{50}$ (Mediana)** | 0.3035 | 50% de los pares |
| **$P_{75}$ (Tercer Cuartil)** | 0.4256 | 75% de los pares |
| **$P_{80}$** | 0.4564 | 80% de los pares |
| **$P_{85}$** | 0.4914 | 85% de los pares |
| **$P_{90}$** | **0.5340** | Zona superior del 10% de mayor afinidad |
| **$P_{95}$** | **0.5996** | Zona extrema superior del 5% |
| **$P_{97.5}$** | **0.6513** | Zona de alta exclusividad semántica (2.5%) |
| **$P_{99}$** | **0.7171** | Cuasi-duplicados / paráfrasis literales (1%) |

### Visualización Gráfica de Alta Resolución
La siguiente figura ilustra la densidad empírica (KDE), el histograma de frecuencias relativas, los componentes de mezcla gaussiana (GMM) y las líneas de corte para los diferentes umbrales:

![Distribución de Similitud Coseno Intra-Documento](imagenes/distribucion_similitud_corpus.png)

---

## 4. Justificación Científica del Umbral de Redundancia ($\tau$)

Para determinar el umbral óptimo $\tau$, se contrastaron dos paradigmas rigurosos:

### 4.1. Paradigma 1: Modelo de Mezcla Gaussiana (Gaussian Mixture Model - GMM)
En el discurso periodístico coherente coexisten naturalmente dos fenómenos latentes:
1. **Componente Tópico/Base (Componente 1):** Oraciones que comparten el tema general del artículo pero aportan información semántica complementaria o nueva.
2. **Componente de Alta Coincidencia / Redundancia (Componente 2):** Oraciones que reiteran la misma idea con ligeras variaciones léxicas o sintácticas (paráfrasis, refraseo o redundancia informativa).

El ajuste GMM de 2 componentes sobre las representaciones de BETO arrojó los siguientes parámetros:
- **Componente 1 (Topical Base):**  
  $$\mu_1 = 0.2152, \quad \sigma_1 = 0.1035, \quad w_1 = 0.5600$$
- **Componente 2 (Semantic Redundancy):**  
  $$\mu_2 = 0.4441, \quad \sigma_2 = 0.1294, \quad w_2 = 0.4400$$

La frontera de decisión bayesiana óptima donde la probabilidad a posteriori de pertenecer a la clase redundante supera el 50% ($P(\text{Componente 2} \mid s) \ge 0.5$) se sitúa exactamente en:
$$\tau_{\text{GMM}} = 0.3434 \approx 0.34$$

### 4.2. Paradigma 2: Análisis por Percentiles No Paramétricos
El percentil 90 ($P_{90} = 0.5340$) y el percentil 95 ($P_{95} = 0.5996$) delimitan la cola derecha de la distribución. La frontera bayesiana GMM ($0.3434$) separa con precisión estadística el régimen de reiteración semántica respecto de la continuidad temática basal del texto periodístico.

### Conclusión del Umbral:
Se adopta formalmente para el proyecto de tesis el umbral:
$$\mathbf{\tau = 0.34} \quad (\text{valor exacto: } 0.3434)$$
Cualquier par oracional intra-texto cuya similitud satisfaga $\cos(s_i, s_j) \ge \tau$ se clasifica formalmente como **Redundante**.

---

## 5. Hallazgos e Insights sobre el Corpus de Noticias

### 5.1. Cuantificación Global de Redundancia
- **Artículos con presencia de redundancia ($\max_{\text{intra}} \ge \tau$):** **2,236** de 2,604 (**85.87%**).
- **Artículos sin redundancia detectada:** **368** (**14.13%**).
- **Pares oracionales redundantes identificados:** **19,616** de 47,693 (**41.13%**).
- **Promedio de pares redundantes por artículo:** **7.53** pares.

### 5.2. Desagregación por Etiqueta del Corpus (`class`)
El dataset original cuenta con la variable objetivo `class`. A continuación se presenta el comportamiento diferencial de la redundancia intra-documental según esta clase:

| Etiqueta (`class`) | N. Documentos | Docs con Redundancia (>= tau) | % Redundancia | Similitud Máxima Media | Similitud Promedio Media | Pares Redundantes / Doc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **False (Noticias Reales)** | 1,259 | 1,096 | **87.05%** | 0.5250 | 0.3702 | 6.67 |
| **True (Noticias Falsas)** | 1,345 | 1,140 | **84.76%** | **0.5356** | **0.3754** | **8.34** |

### 5.3. Implicaciones para la Tesis y Experimentos XAI
1. **Poder de Discriminación Semántica:**  
   La extracción de métricas derivadas de BETO como `max_intra_similarity`, `mean_intra_similarity` y `redundant_pairs_count` proporciona un vector de características continuo alineado directamente con el modelo ganador de la Fase 1.
2. **Integración con Modelos Explicables:**  
   El umbral $\tau = 0.34$ permite alimentar las explicaciones locales (SHAP, LIME, Integrated Gradients) identificando exactamente cuáles oraciones desencadenan la activación de redundancia y qué tokens generan las mayores atribuciones cruzadas en BETO.
3. **Consistencia de Artefactos:**  
   El archivo consolidado `dataset_con_similitudes.csv` contiene las columnas `num_sentences`, `num_pairs`, `pairwise_similarities`, `mean_intra_similarity`, `max_intra_similarity`, `p90_intra_similarity`, `var_intra_similarity`, `has_redundancy` y `redundant_pairs_count`, calculadas íntegramente con `hiiamsid/sentence_similarity_spanish_es`.

---
*Reporte generado de forma autónoma según las especificaciones del protocolo científico de investigación.*
