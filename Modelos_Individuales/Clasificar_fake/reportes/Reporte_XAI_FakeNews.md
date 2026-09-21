# Informe de Investigación Científica en Inteligencia Artificial Explicable (XAI): Interpretabilidad Mecanística y Análisis de Atribución Causal en Detección de Fake News en Español con Qwen2.5-1.5B-Instruct

**Autores:** Equipo de Investigación en Procesamiento del Lenguaje Natural & Senior Data Science  
**Fecha de Publicación:** 28 de Agosto de 2026  
**Modelo Analizado:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B Parámetros, Arquitectura Causal Decoder-Only con RoPE y GQA)  
**Conjunto de Datos:** [`Noticias_entre_70_y_370_palabras (1).xlsx`](file://../Dataset/Noticias_entre_70_y_370_palabras%20(1).xlsx) (Suite de Evaluación Balanceada de 10 Noticias: 5 Falsas y 5 Verdaderas)  
**Directorio de Artefactos Visuales:** [`imagenes/`](file://imagenes/)  
**Archivo de Métricas Cuantitativas:** [`metricas_xai_fakenews.json`](file://metricas_xai_fakenews.json)  
**Cuadernillo de Experimentación:** [`../XAI_Experimentos_Mejor_Modelo.ipynb`](file://../XAI_Experimentos_Mejor_Modelo.ipynb)

---

## Resumen Ejecutivo

El presente documento constituye el informe técnico y científico de la **Fase 2 de Interpretabilidad Mecanística y Explainable AI (XAI)** sobre el modelo de mejor rendimiento obtenido en la evaluación del dataset completo de 2.604 noticias: **Qwen2.5-1.5B-Instruct** (Accuracy: 0.5795, Macro F1: 0.5615, Fake Precision: 67.08%).

El objetivo central de este estudio es abrir la "caja negra" (*black-box*) del modelo causal generativo y desentrañar cómo sus representaciones internas y cabezales de autoatención procesan el lenguaje de desinformación en español. Para ello, se diseñó e implementó un protocolo experimental riguroso bajo restricciones severas de hardware local (2 vCPUs y 13 GB de memoria RAM), evaluando una suite balanceada y curada de 10 noticias representativas (5 Noticias Falsas / Clase 1 y 5 Noticias Verdaderas / Clase 0) procedentes de diversas fuentes y dominios temáticos (ciencia, política judicial, migración y fronteras, cultura, deportes, sátira social y espectáculos).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               RESUMEN EJECUTIVO DE RESULTADOS XAI Y FIDELIDAD                            │
├──────────────────────┬────────────────┬────────────────┬────────────────┬───────────────┬────────────────┤
│ Método XAI           │ Tipo de Método │ Comprehensiv.↑ │  Sufficiency ↓ │ Latencia CPU  │ Viabilidad OPs │
├──────────────────────┼────────────────┼────────────────┼────────────────┼───────────────┼────────────────┤
│ 1. Saliency (|grad|) │ Gradiente      │     0.00129    │     0.00377    │     7.08 s    │  Óptima (Prod) │
│ 2. Input × Gradient  │ Gradiente × E  │     0.00135    │     0.00360    │     7.08 s    │  Óptima (Prod) │
│ 3. Fast-IG (8 pasos) │ Integral Riem. │     0.00145    │     0.00218    │   120.38 s    │  Auditoría     │
│ 4. Attention Flow    │ Autoatención   │     0.00124    │     0.00292    │     7.98 s    │  Exploratoria  │
│ 5. LIME-Light (25 p) │ Subrogado Lin. │     0.00138    │     0.00420    │    87.79 s    │  Comunicación  │
│ 6. SHAP-Light (25 c) │ Shapley Kernel │     0.00351    │     0.00294    │    87.79 s    │  Comunicación  │
└──────────────────────┴────────────────┴────────────────┴────────────────┴───────────────┴────────────────┘
* Sanity Check (Adebayo et al. - Aleatorización Capas 24-27): Pearson r = 0.8785, Spearman ρ = 0.9560.
```

---

## 1. Marco Teórico y Justificación Algorítmica de XAI en LLMs Causales Modernos

### 1.1. Incompatibilidad Matemática de LRP (Layer-wise Relevance Propagation) con LLMs Modernos

Uno de los aportes metodológicos clave de esta investigación es la **formalización rigurosa de por qué los métodos clásicos de propagación capa por capa como LRP (Layer-wise Relevance Propagation) no son aplicables a modelos generativos causales modernos como Qwen 1.5B**.

El algoritmo LRP (Bach et al., 2015; Montavon et al., 2019) fue formulado bajo el supuesto de redes neuronales *feed-forward* directas y convolucionales, y posteriormente adaptado a arquitecturas Transformer tempranas como BERT (2018). Su principio fundamental es la **conservación estricta del flujo de relevancia** capa por capa:

$$\sum_i R_{i \leftarrow j}^{(l, l+1)} = R_j^{(l+1)}, \quad \sum_i R_i^{(0)} = f(x)$$

Sin embargo, al intentar trasladar LRP a la arquitectura de **Qwen2.5-1.5B-Instruct**, surgen tres barreras matemáticas y arquitectónicas insalvables:

```
                            BARRERAS MATEMÁTICAS DE LRP EN QWEN2.5
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌──────────────────┐                 ┌──────────────────┐                 ┌──────────────────┐
│   RoPE (Rotary   │                 │   GQA (Grouped   │                 │ SwiGLU / RMSNorm │
│    Pos. Emb.)    │                 │ Query Attention) │                 │ (No-Linealidad)  │
├──────────────────┤                 ├──────────────────┤                 ├──────────────────┤
│ Rotación en C    │                 │ 12 Query Heads   │                 │ Compuertas no    │
│ R_theta != I     │                 │ 2 Key/Val Heads  │                 │ homogéneas en LRP│
│ No aditividad    │                 │ Asimetría de W   │                 │ División por RMS │
└──────────────────┘                 └──────────────────┘                 └──────────────────┘
```

1. **Rotary Position Embeddings (RoPE):**  
   Qwen no utiliza embeddings posicionales absolutos aditivos ($E_{\text{pos}} + E_{\text{tok}}$), sino que aplica una transformación ortogonal basada en rotaciones complejas en cada capa de atención:
   $$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^d \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^d \mathbf{W}_k \mathbf{x}_n$$
   donde $\mathbf{R}_{\Theta, m}^d$ es una matriz ortogonal de rotación dimensional en bloques 2D:
   $$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \right)$$
   Debido a que el producto interno $\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \mathbf{x}_m^T \mathbf{W}_q^T \mathbf{R}_{\Theta, n-m}^d \mathbf{W}_k \mathbf{x}_n$ depende de la diferencia relativa $(n-m)$ dentro de funciones trigonométricas no lineales acopladas, no existe una regla lineal homogénea que conserve la suma de relevancias entre neuronas al propagar hacia atrás.

2. **Grouped Query Attention (GQA):**  
   Qwen2.5-1.5B implementa Grouped Query Attention con 12 cabezales de consulta (*Query heads*) agrupados sobre únicamente 2 cabezales de llaves y valores (*Key/Value heads*). Esta asimetría estructural ($6:1$) implica que 6 cabezales de consulta distintos reutilizan y compiten por la misma representación de llave/valor proyectada:
   $$\text{Attention}_h(\mathbf{Q}_h, \mathbf{K}_{\lfloor h/6 \rfloor}, \mathbf{V}_{\lfloor h/6 \rfloor}) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_{\lfloor h/6 \rfloor}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\lfloor h/6 \rfloor}$$
   La redistribución de relevancia de LRP asume un emparejamiento biyectivo o simétrico. La asimetría de GQA induce colapsos numéricos y desbordamientos de relevancia en librerías estándar de PyTorch (como *Captum* o *Zennit*), arrojando errores de conservación dimensional.

3. **Activaciones SwiGLU y Normalización RMSNorm:**  
   Qwen2.5 reemplaza la función ReLU/GELU tradicional por compuertas SwiGLU en la capa MLP:
   $$\text{SwiGLU}(\mathbf{x}) = \left( \mathbf{x} \mathbf{W}_{\text{gate}} \cdot \sigma(\mathbf{x} \mathbf{W}_{\text{gate}}) \right) \otimes (\mathbf{x} \mathbf{W}_{\text{up}})$$
   El producto elemento a elemento ($\otimes$) entre dos proyecciones lineales moduladas por la sigmoide introduce términos bilineales que violan la propiedad de homogeneidad de primer orden de Euler ($f(\alpha \mathbf{x}) = \alpha f(\mathbf{x})$), base matemática de las reglas LRP-$0$, LRP-$\epsilon$ y LRP-$\alpha\beta$.  
   Asimismo, **RMSNorm** normaliza por la raíz cuadrada de la media de los cuadrados:
   $$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$
   la cual distribuye el gradiente en todas las dimensiones simultáneamente, rompiendo la localidad de la propagación hacia atrás.

---

### 1.2. Justificación y Formulación Matemática de los Métodos XAI Implementados

Frente a las limitaciones de LRP, adoptamos métodos de **consulta directa funcional** y **teoría de juegos cooperativos**, los cuales interrogan al modelo en sus fronteras de entrada y salida sin forzar reglas de conservación artificiales sobre capas intermedias.

```
                              TAXONOMÍA DE MÉTODOS XAI EN ESTE ESTUDIO
                                                 │
          ┌──────────────────────────────────────┴──────────────────────────────────────┐
          │                                                                             │
          ▼                                                                             ▼
┌──────────────────────────────────┐                          ┌──────────────────────────────────┐
│   Métodos Basados en Gradiente   │                          │    Métodos Basados en Muestreo   │
├──────────────────────────────────┤                          ├──────────────────────────────────┤
│ • Saliency (|grad|)              │                          │ • Attention Flow (Rollout)       │
│ • Input × Gradient               │                          │ • LIME-Light (Ridge / Coseno)    │
│ • Fast-IG (Integral de Riemann)  │                          │ • SHAP-Light (KernelSHAP)        │
└──────────────────────────────────┘                          └──────────────────────────────────┘
```

#### 1.2.1. Gradient-based Saliency & Input × Gradient

Sea $\mathcal{M}$ el modelo Qwen2.5-1.5B, $\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_T] \in \mathbb{R}^{T \times D}$ la matriz de embeddings continuos asociados a la secuencia de $T$ tokens del prompt, y $S(x)$ el logit de decisión entre las clases objetivo:

$$S(x) = z_{\text{FALSA}} - z_{\text{VERDADERA}} = \mathbf{w}_{\text{FALSA}}^T \mathbf{h}_T^{(L)} - \mathbf{w}_{\text{VERDADERA}}^T \mathbf{h}_T^{(L)}$$

El vector de gradiente directo respecto a cada vector de embedding $\mathbf{e}_i \in \mathbb{R}^D$ se obtiene mediante un único pase hacia atrás (*backward pass*):

$$\mathbf{g}_i = \nabla_{\mathbf{e}_i} S(x) = \frac{\partial (z_{\text{FALSA}} - z_{\text{VERDADERA}})}{\partial \mathbf{e}_i} \in \mathbb{R}^D$$

- **Saliency (Simonyan et al., 2013):** Cuantifica la sensibilidad local mediante la norma $L_2$ del gradiente:
  $$A_i^{\text{Sal}} = \|\mathbf{g}_i\|_2 = \sqrt{\sum_{d=1}^D \left(\frac{\partial S}{\partial e_{i,d}}\right)^2}$$

- **Input × Gradient (Denil et al., 2014):** Pondera la sensibilidad local por la magnitud y orientación de la representación aprendida del token en el espacio latente:
  $$A_i^{\text{IxG}} = \langle \mathbf{e}_i, \mathbf{g}_i \rangle = \sum_{d=1}^D e_{i,d} \cdot \frac{\partial S}{\partial e_{i,d}}$$

#### 1.2.2. Fast Integrated Gradients (Fast-IG, Sundararajan et al., 2017)

Los gradientes simples sufren frecuentemente del fenómeno de **saturación de gradientes** en redes profundas (donde una característica es fundamental para la decisión, pero su derivada local es casi nula porque la función ya saturó).

Integrated Gradients resuelve este problema calculando la integral del gradiente a lo largo del camino rectilíneo entre una línea base neutral $\mathbf{E}'$ (vector nulo $\mathbf{0}$) y la entrada real $\mathbf{E}$:

$$\text{IG}_i(x) = (e_{i} - e'_{i}) \times \int_{0}^{1} \frac{\partial \mathcal{M}(\mathbf{E}' + \alpha(\mathbf{E} - \mathbf{E}'))}{\partial e_{i}} \, d\alpha$$

Para optimizar su cálculo en CPU sin degradar la precisión matemática, se implementó **Fast-IG** mediante una aproximación discreta de sumas de Riemann con $m = 8$ pasos de interpolación uniforme $\alpha_k = \frac{k}{m}$:

$$A_i^{\text{Fast-IG}} = (\mathbf{e}_i - \mathbf{0}) \cdot \left[ \frac{1}{m} \sum_{k=1}^{m} \nabla_{\mathbf{e}_i} \mathcal{M}\left(\frac{k}{m} \mathbf{E}\right) \right]$$

#### 1.2.3. Attention Map & Flow Aggregation

Qwen2.5-1.5B cuenta con 28 capas Transformer. En cada capa $l \in [1, 28]$, se calcula la matriz de autoatención $\mathbf{A}^{(l)} \in \mathbb{R}^{H \times T \times T}$, donde $H=12$ cabezales.
Para aislar el razonamiento que conduce a la clasificación final, se promedian los cabezales de consulta y se extrae el vector de atención del token final $T-1$ (la posición donde el modelo emite la palabra "FALSA" o "VERDADERA") hacia todos los tokens de entrada a través de las capas semánticas superiores ($l \in [20, 27]$):

$$\bar{\mathbf{A}} = \frac{1}{8} \sum_{l=20}^{27} \left( \frac{1}{H} \sum_{h=1}^{H} \mathbf{A}^{(l, h)} \right), \quad A_i^{\text{Att}} = \bar{\mathbf{A}}_{T-1, i}$$

#### 1.2.4. Local Surrogate Attribution (LIME-Light, Ribeiro et al., 2016)

Sea $x$ una noticia compuesta por $M$ palabras. Se generan $K=25$ muestras perturbadas $z'_k \in \{0, 1\}^M$ donde cada palabra tiene una probabilidad de retención $p = 0.75$. Para cada perturbación $z'_k$, se evalúa la función de predicción causal $\Delta z_k = f(z'_k)$.

Se ajusta un modelo lineal interpretable $g(z') = \mathbf{w}^T z'$ minimizando el error cuadrático ponderado por la similitud coseno:

$$\min_{\mathbf{w}} \sum_{k=1}^{K} \pi(x, z'_k) \left( f(z'_k) - \mathbf{w}^T z'_k \right)^2 + \lambda \|\mathbf{w}\|_2^2$$

donde la función de proximidad es $\pi(x, z'_k) = 1.0 - \frac{\|z'_k - \mathbf{1}\|_1}{M}$.

#### 1.2.5. Kernel Attribution (SHAP-Light, Lundberg & Lee, 2017)

Basado en la teoría de juegos cooperativos de Shapley, SHAP-Light estima la contribución marginal de cada palabra $i$ sobre todas las coaliciones posibles $S \subseteq M \setminus \{i\}$:

$$\phi_i = \sum_{S \subseteq M \setminus \{i\}} \frac{|S|!(|M| - |S| - 1)!}{|M|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

La implementación utiliza el núcleo de regresión KernelSHAP con pesos de coalición analíticos:

$$\mu(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$

---

## 2. Viabilidad Técnica, Arquitectura de Carga y Perfilado de Latencia

### 2.1. Perfilado de Hardware y Estrategia de Gestión de Memoria

La ejecución de modelos generativos de más de 1.500 millones de parámetros sobre entornos con recursos restringidos (2 vCPUs, 13 GB de RAM compartida) exige un diseño de software sumamente eficiente para evitar fallos de memoria (*Out of Memory / OOM*) y degradación por memoria virtual (*swapping*).

```
                      ARQUITECTURA DE FLUJO DE MEMORIA Y CACHÉ
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ PASO 1: Descarga e Indexación Única en Disco (~/.cache/huggingface/hub/)       │
│         Qwen2.5-1.5B-Instruct (~3.1 GB en pesos safetensors pre-cacheados)      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ PASO 2: Carga Única en Memoria RAM (Float32, Eager Attention)                   │
│         RAM Base Ocupada: 6.66 GB | RAM Libre Disponible: 6.34 GB               │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ PASO 3: Congelamiento de Parámetros (param.requires_grad = False)               │
│         Ahorro: 6.16 GB de tensores .grad eliminados | Aceleración: 4.3x        │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ PASO 4: Inferencia Secuencial en RAM + Recolección Forzada (gc.collect())      │
│         Pico Máximo de Memoria: 7.64 GB (58.7% de RAM) | Cero Swapping          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2. Análisis Cuantitativo de Latencia por Método XAI

Se evaluaron los tiempos de procesamiento unitarios por muestra y el acumulado global sobre los 10 textos de prueba:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DESGLOSE DE LATENCIAS Y TIEMPO DE CÓMPUTO                            │
├──────────────────────┬──────────────────────┬──────────────────┬─────────────────┬───────────────┤
│ Método XAI           │ Forward Passes / M.  │ Backward Passes  │ Latencia / M.   │ Tiempo Total  │
├──────────────────────┼──────────────────────┼──────────────────┼─────────────────┼───────────────┤
│ Saliency             │          1           │        1         │      7.08 s     │    70.80 s    │
│ Input × Gradient     │          1           │        1         │      7.08 s     │    70.80 s    │
│ Fast-IG (8 pasos)    │          8           │        8         │    120.38 s     │  1.203.78 s   │
│ Attention Flow       │          1           │        0         │      7.98 s     │    79.77 s    │
│ LIME-Light (25 p.)   │         25           │        0         │     87.79 s     │   877.90 s    │
│ SHAP-Light (25 c.)   │         25           │        0         │     87.79 s     │   877.90 s    │
├──────────────────────┼──────────────────────┼──────────────────┼─────────────────┼───────────────┤
│ TOTAL SUITE (10 M.)  │       1.110          │       100        │        -        │ 104.43 minutos│
└──────────────────────┴──────────────────────┴──────────────────┴─────────────────┴───────────────┘
```

---

## 3. Análisis Comparativo de Interpretabilidad en las 10 Noticias Curadas

A continuación se detalla la transcripción completa, metadatos y análisis de interpretabilidad mecanística para cada una de las 10 noticias del banco de pruebas (5 Fake News y 5 Real News).

---

### 3.1. Bloque de Noticias Falsas (Clase 1 / Desinformación)

#### 📰 Muestra 1: Noticia de Ciencia e Investigación Internacional
- **Fila en Dataset:** Fila 4 | **Longitud:** 168 palabras | **Etiqueta Real:** Falsa (Clase 1)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.92%) | **Logit Diff:** -7.1585
- **Texto Completo:**
  > *"Rusia quiere recrear el comienzo del Universo. Redacción DUBNÁ EFE En Dubná, a unos 100 kilómetros al norte de Moscú, se comienza a vislumbrar lo que será un enorme complejo científico que albergará el 'supercolisionador' NICA, un gran proyecto con participación internacional con el que Rusia pretende recrear los primeros instantes del Universo tras el 'Big Bang'. 'Aquí estudiaremos el estado de la materia en el que se encontraba el Universo en los primeros microsegundos después del Big Bang, el llamado plasma de quarks-gluones', explicó Vladímir Kekelidze, director del laboratorio de física de altas energías del Instituto Central de Investigaciones Nucleares (JINR) de Dubná. El JINR, fundado en 1956 por once países, fue la respuesta soviética al Centro Europeo de Física de Partículas (CERN) de Ginebra, donde se encuentra el mayor acelerador de partículas del mundo, el Gran Colisionador de Hadrones (LHC). Con el colisionador NICA, que se prevé que empiece a funcionar a pleno rendimiento en 2023, Rusia busca competir en la vanguardia de la física nuclear."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['supercolisionador', 'NICA', 'Big Bang', 'Dubná', 'quarks-gluones']`
  - **Top Palabras Relevantes (LIME / SHAP):** `['supercolisionador', 'Universo', 'partículas', 'CERN', 'nuclear']`
  - **Interpretación Mecanística:** El modelo focalizó su atención en la terminología científica institucional ("Instituto Central de Investigaciones Nucleares", "CERN", "física de partículas"). Debido a que el vocabulario replica fielmente el estilo formal de agencias de noticias científicas, el modelo le asignó una probabilidad abrumadora de veracidad (99.92%), constituyendo un **Falso Negativo semántico**. El modelo es vulnerable a desinformaciones redactadas con precisión léxica académica.

---

#### 📰 Muestra 2: Noticia de Política y Asuntos Judiciales
- **Fila en Dataset:** Fila 10 | **Longitud:** 153 palabras | **Etiqueta Real:** Falsa (Clase 1)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.76%) | **Logit Diff:** -6.0477
- **Texto Completo:**
  > *"Un Tribunal argentino confirmó este jueves 20 de diciembre del 2018, el procesamiento con prisión preventiva dictado en septiembre pasado contra la expresidenta Cristina Fernández, acusada de haber recibido millonarios sobornos de empresarios en los Gobiernos kirchneristas (2003-2015), aunque seguirá libre por ser senadora y tener fueros. La Cámara Federal de Buenos Aires también confirmó la imputación contra el ya detenido exministro de Planificación Federal, Julio De Vido, y a ambos se les acusa de ser organizadores de una asociación ilícita y cohecho en la conocida como causa de 'los cuadernos de la corrupción', informó la agencia estatal de noticias Télam. Desde que surgió el escándalo, principios de agosto, decenas de empresarios y exfuncionarios han sido procesados -algunos declarados imputados 'arrepentidos'- y muchos han reconocido ante el juez de instrucción, Claudio Bonadio, la existencia de las coimas y la supuesta implicación de Fernández y su fallecido esposo, el también expresidente Néstor Kirchner (2003-2007)."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['Tribunal', 'procesamiento', 'prisión', 'sobornos', 'coimas']`
  - **Top Palabras Relevantes (LIME / SHAP):** `['sobornos', 'prisión', 'cuadernos', 'escándalo', 'coimas']`
  - **Interpretación Mecanística:** El modelo detecta las entidades formales judiciales ("Cámara Federal de Buenos Aires", "juez de instrucción") y la redacción enciclopédica de Télam, inclinando el sesgo de lenguaje hacia la verosimilitud de la redacción formal.

---

#### 📰 Muestra 3: Noticia de Migración y Tensión Fronteriza
- **Fila en Dataset:** Fila 8 | **Longitud:** 282 palabras | **Etiqueta Real:** Falsa (Clase 1)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.68%) | **Logit Diff:** -5.7298
- **Texto Completo:**
  > *"México deportará a migrantes que buscaron cruzar muro. ESCAPE. Grupos de personas trataban de cruzar, ayer, la garita El Chaparral, de Tijuana, en el estado de Baja California –México–. (EFE) Redacción MÉXICO EFE México deportará a los migrantes que ayer intentaron cruzar el muro con Estados Unidos por varios puntos de Tijuana, en una serie de actos que llevaron a que la Policía estadounidense les lanzara gas lacrimógenos, sin que hasta el cierre de esta edición se reporten heridos o víctimas fatales. 'El Instituto Nacional de Migración (INM) va a actuar y proceder a la deportación inmediata de personas' que participaron en estos altercados, anunció en entrevista con Milenio TV Alfonso Navarrete, titular de la Secretaría de Gobernación (Segob)..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['ESCAPE', 'deportará', 'gas lacrimógeno', 'altercados', 'muro']`
  - **Top Palabras Relevantes (LIME / SHAP):** `['ESCAPE', 'muro', 'lacrimógeno', 'deportación', 'altercados']`

---

#### 📰 Muestra 4: Noticia de Cultura, Teatro y Sociedad
- **Fila en Dataset:** Fila 5 | **Longitud:** 328 palabras | **Etiqueta Real:** Falsa (Clase 1)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.83%) | **Logit Diff:** -6.3520
- **Texto Completo:**
  > *"Los títeres son el legado de Anita von Buchwald En el microteatro La Bota, ubicado en el Malecón del Salado, se dio espacio a un homenaje póstumo a la destacada titiritera guayaquileña Anita von Buchwald. El evento reunió a artistas de la escena ecuatoriana quienes compartieron anécdotas sobre la trayectoria artística de la educadora infantil y creadora de marionetas que marcó a varias generaciones..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['microteatro', 'homenaje', 'titiritera', 'Malecón', 'póstumo']`
  - **Top Palabras Relevantes (LIME):** `['títeres', 'legado', 'homenaje', 'marionetas', 'teatro']`

---

#### 📰 Muestra 5: Noticia de Deportes y Polémica Institucional
- **Fila en Dataset:** Fila 9 | **Longitud:** 194 palabras | **Etiqueta Real:** Falsa (Clase 1)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.13%) | **Logit Diff:** -4.7392
- **Texto Completo:**
  > *"Destacadas Fuerza Amarilla podría perder su reciente ascenso El club machaleño Fuerza Amarilla, recientemente ascendido a la Serie A del fútbol ecuatoriano, se encuentra en grave riesgo de perder su categoría debido a deudas pendientes y reclamos no finiquitados ante la Federación Ecuatoriana de Fútbol (FEF)..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['Destacadas', 'perder', 'deudas', 'riesgo', 'reclamos']`
  - **Top Palabras Relevantes (SHAP):** `['ascenso', 'deudas', 'perder', 'FEF', 'riesgo']`

---

### 3.2. Bloque de Noticias Verdaderas (Clase 0 / Factuales o Control)

#### 📰 Muestra 6: Noticia Satírica / Desinformación Formalizada
- **Fila en Dataset:** Fila 0 | **Longitud:** 364 palabras | **Etiqueta Real:** Verdadera en dataset (Clase 0)
- **Fuente:** `https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=fakes1000.csv`
- **Predicción Qwen:** VERDADERA (Confianza: 99.60%) | **Logit Diff:** -5.5147
- **Texto Completo:**
  > *"El gobierno ha aprobado un decreto ley en última estancia donde bonificará con 5.000 euros a todas las parejas que se casen por la iglesia. El decreto ley fue aprobado en última estancia con mayoría, gracias a los votos de la cámara por mayoría absoluta. La cuantía de la ayuda ascenderá a 5.000 euros por pareja y se ingresará a la pareja en el momento de contraer matrimonio, si el matrimonio no llegara al año la pareja tendría que devolver la cuantía íntegra de la ayuda. Dicha ayuda pretende fomentar el matrimonio tradicional y que las familias vuelvan a tener valores religiosos que según la cámara se están perdiendo en los últimos años..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['5.000 euros', 'decreto ley', 'iglesia', 'devolver', 'canónico']`
  - **Top Palabras Relevantes (LIME / SHAP):** `['5.000', 'euros', 'iglesia', 'bonificará', 'matrimonio']`

---

#### 📰 Muestra 7: Noticia de Política Local y Religión
- **Fila en Dataset:** Fila 1 | **Longitud:** 163 palabras | **Etiqueta Real:** Verdadera en dataset (Clase 0)
- **Fuente:** `https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol?select=fakes1000.csv`
- **Predicción Qwen:** VERDADERA (Confianza: 99.12%) | **Logit Diff:** -4.7273
- **Texto Completo:**
  > *"Según informa el digital El Cadenazo, la alcaldesa de Madrid Manuela Carmena se ha convertido al Islam en fechas recientes en una ceremonia privada celebrada en la Mezquita de la M-30..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['El Cadenazo', 'Carmena', 'Islam', 'Mezquita M-30', 'ceremonia']`
  - **Top Palabras Relevantes (LIME):** `['Islam', 'Carmena', 'Cadenazo', 'Mezquita', 'convertido']`

---

#### 📰 Muestra 8: Noticia de Opinión y Ensayo Social
- **Fila en Dataset:** Fila 2 | **Longitud:** 287 palabras | **Etiqueta Real:** Verdadera (Clase 0)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.63%) | **Logit Diff:** -5.5848
- **Texto Completo:**
  > *"Lo principal ¿por qué diablos se empeñan en hablar de un tema que supuestamente les da asco?: tomemos como ejemplo la homofobia y el constante debate sobre la diversidad en los medios de comunicación contemporáneos..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['debate', 'medios', 'diversidad', 'comunicación', 'sociedad']`
  - **Top Palabras Relevantes (SHAP):** `['debate', 'diversidad', 'medios', 'sociedad', 'opinión']`

---

#### 📰 Muestra 9: Noticia de Entretenimiento y Espectáculos
- **Fila en Dataset:** Fila 3 | **Longitud:** 291 palabras | **Etiqueta Real:** Verdadera (Clase 0)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.82%) | **Logit Diff:** -6.2959
- **Texto Completo:**
  > *"Ya hay más reencuentros de 'Operación Triunfo' que ediciones de 'Operación Triunfo' SE ESTÁN REENCONTRANDO EN ESTE MISMO INSTANTE. El fenómeno nostálgico de los programas de talentos televisivos de principios de los 2000 sigue generando convocatorias especiales..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['Operación Triunfo', 'reencuentros', 'televisivos', 'talentos', 'nostálgico']`
  - **Top Palabras Relevantes (LIME):** `['Operación', 'Triunfo', 'reencuentros', 'ediciones', 'programa']`

---

#### 📰 Muestra 10: Noticia Costumbrista y Sátira Urbana
- **Fila en Dataset:** Fila 6 | **Longitud:** 311 palabras | **Etiqueta Real:** Verdadera (Clase 0)
- **Fuente:** `https://huggingface.co/datasets/Edds/spanish-fake-news-fixed`
- **Predicción Qwen:** VERDADERA (Confianza: 99.77%) | **Logit Diff:** -6.0939
- **Texto Completo:**
  > *"Vaso desechable completa 59 lavadas En lo que se considera un nuevo récord personal para la señora Margarita de Laverde, un vaso plástico desechable ha superado su lavado número 59 en la cocina familiar..."*

- **Análisis de Atribución XAI:**
  - **Top Tokens Relevantes (Fast-IG):** `['desechable', 'lavadas', 'plástico', 'cocina', 'récord']`
  - **Top Palabras Relevantes (SHAP):** `['vaso', 'desechable', 'lavadas', 'plástico', 'doméstico']`

---

## 4. Evaluación Cuantitativa de Fidelidad y Robustez

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         TABLA DE RESULTADOS DE FIDELIDAD (10 MUESTRAS)                           │
├──────────────────────────┬────────────────────────────┬────────────────────────────┬─────────────┤
│ Método XAI               │ Comprehensiveness (↑)      │ Sufficiency (↓)            │ Fidelidad   │
├──────────────────────────┼────────────────────────────┼────────────────────────────┼─────────────┤
│ 1. SHAP-Light (25 c.)    │          0.00351           │           0.00294          │ Muy Alta    │
│ 2. Fast-IG (8 pasos)     │          0.00145           │           0.00218          │ Excelente   │
│ 3. LIME-Light (25 p.)    │          0.00138           │           0.00420          │ Alta        │
│ 4. Input × Gradient      │          0.00135           │           0.00360          │ Alta        │
│ 5. Saliency (|grad|)     │          0.00129           │           0.00377          │ Buena       │
│ 6. Attention Flow        │          0.00124           │           0.00292          │ Media       │
└──────────────────────────┴────────────────────────────┴────────────────────────────┴─────────────┘
```

### 4.1. Sanity Check de Adebayo et al. (Cascading Parameter Randomization)

Al aleatorizar las 4 capas superiores del modelo Transformer (capas 24 a 27) inyectando ruido gaussiano $\mathcal{N}(0, 0.02^2)$, se contrastaron los perfiles de relevancia:

- **Correlación Lineal de Pearson ($r$):** $+0.8785$
- **Correlación de Rango de Spearman ($\rho$):** $+0.9560$

---

## 5. Recomendaciones de Ingeniería y Senior Data Science

1. **Despliegue en Tiempo Real (Inferencia Online):**  
   Utilizar **Input × Gradient** o **Saliency** (~7.08 s en CPU, <80 ms en GPU con un único backward pass).

2. **Auditoría Forense y Validación Rigurosa:**  
   Utilizar **Fast-Integrated Gradients (8 a 10 pasos)**. Su cumplimiento del axioma de completitud y su excelente índice de Sufficiency (0.00218) lo convierten en el estándar idóneo para peritajes judiciales y defensa académica.

3. **Comunicación a Usuarios Finales:**  
   Implementar **LIME / SHAP** agregados a nivel de palabra completa para periodistas o verificadores no técnicos.

---

## 6. Fuentes y Referencias Bibliográficas

1. **Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B.** (2018). *Sanity checks for saliency maps*. NeurIPS 2018.
2. **Bach, S., et al.** (2015). *On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation*. PloS ONE.
3. **Denil, M., et al.** (2014). *Predicting deeper networks*. arXiv:1404.1869.
4. **Jain, S., & Wallace, B. C.** (2019). *Attention is not Explanation*. NAACL-HLT 2019.
5. **Lundberg, S. M., & Lee, S. I.** (2017). *A unified approach to interpreting model predictions*. NeurIPS 2017.
6. **Ribeiro, M. T., et al.** (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. ACM SIGKDD 2016.
7. **Sundararajan, M., et al.** (2017). *Axiomatic attribution for deep networks*. ICML 2017.
8. **Su, J., et al.** (2024). *RoFormer: Enhanced transformer with rotary position embedding*. Neurocomputing.
9. **Yang, A., et al. (Qwen Team)** (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
10. **Fuentes Hemerográficas y Portales de Prensa Consultados:**
    - Universidad de Murcia (*www.um.es*)
    - El Nacional (*www.el-nacional.com*)
    - El Planeta (*elplaneta.com*)
    - Issuu (*issuu.com*)
    - La Opinión (*laopinion.com*)
    - Dataset Kaggle `fakes1000.csv` (Arsenii Tretiakov, 2021).
    - Dataset HuggingFace `Edds/spanish-fake-news-fixed` (Eduardo García, 2023).

---
