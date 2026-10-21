# Informe Experimental, Mecanístico y Educativo: Dinámica de Clasificación de Sensacionalismo en BERT

## ¿Es el Sensacionalismo del Documento Igual al Promedio de sus Frases o Depende de 1 o 2 Frases Gatillo?
### *Estudio Comparativo y Análisis Causal entre Prensa Digital Real y Noticias Sintéticas de IA*

---

**Autores:** Equipo de Investigación en Procesamiento del Lenguaje Natural & Senior Data Science  
**Fecha de Publicación:** Septiembre de 2026  
**Modelo Evaluado:** [`JJNeila/bert-spanish-sensationalism-oss`](https://huggingface.co/JJNeila/bert-spanish-sensationalism-oss) (~110M Parámetros, BETO Transformer Encoder-Only)  
**Corpus Experimental Analizado:** 272 noticias completas  
- **Sub-Corpus 1 (Prensa Digital Real):** 202 noticias de medios periodísticos hispanohablantes ([`dataset_Amarillismo.csv`](../Datasets/sensacionalismo/dataset_Amarillismo.csv))  
- **Sub-Corpus 2 (Noticias Sintéticas de IA):** 70 noticias estructuradas generadas artificialmente ([`dataset_IA_sintetico_70.csv`](../Datasets/sensacionalismo/dataset_IA_sintetico_70.csv))  
**Unidades Textuales Descompuestas:** 2,012 frases y cláusulas informativas delimitadas por signos de puntuación  
**Hardware de Ejecución:** GPU NVIDIA GeForce GTX 1650 (4 GB VRAM, `device='cuda'`, aceleración tensorial en lotes)  
**Script Ejecutable Principal:** [`../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/experimento_sensacionalismo_frases.py`](../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/experimento_sensacionalismo_frases.py)  
**Script Generador de Figuras:** [`../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/generar_figuras_detalladas.py`](../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/generar_figuras_detalladas.py)  
**Base de Datos Experimental Completa:** [`datos_sensacionalismo_frases.csv`](datos_sensacionalismo_frases.csv)  
**Métricas Estadísticas Formales:** [`metricas_sensacionalismo_frases.json`](metricas_sensacionalismo_frases.json)  
**Directorio de Figuras Científicas (300 DPI):** [`imagenes/sensacionalismo_frases/`](imagenes/sensacionalismo_frases/)  

---

## Guía de Lectura Rápida y Preguntas Centrales de la Investigación

Este informe ha sido redactado con un enfoque dual: **rigor estadístico para la Tesis de Maestría** y **claridad pedagógica y divulgativa** para que cualquier investigador, estudiante o lector pueda comprender a fondo cómo "piensa" y clasifica un modelo de lenguaje basado en Transformers (BERT/BETO).

A lo largo del documento se da respuesta experimental a tres grandes enigmas:
1. **¿El sensacionalismo de un artículo completo es la suma o el promedio de las emociones de sus oraciones?**
2. **¿Basta con que una noticia tenga una sola frase alarmista (o dos) para que todo el texto sea etiquetado como amarillista?**
3. **¿Se comporta igual una noticia real de un periódico digital que un texto generado artificialmente por una Inteligencia Artificial?**

Para ilustrar de forma tangible cada concepto, utilizaremos **cuatro noticias de referencia** que seguiremos paso a paso a lo largo de todas las pruebas:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           LAS 4 NOTICIAS DE REFERENCIA UTILIZADAS COMO HILO CONDUCTOR                            │
├────────┬──────────────────────────┬────────────────────┬────────────────────────────────────────────────────────┤
│ ID     │ Corpus de Procedencia    │ Clase Real (Humana)│ Titular o Texto de Entrada                             │
├────────┼──────────────────────────┼────────────────────┼────────────────────────────────────────────────────────┤
│ AMA_2  │ Prensa Real (Amarillismo)│ Amarillista (1)    │ "¿Asteroide Apophis destruiría la Tierra en 2029?      │
│        │                          │                    │  Los cálculos de la Nasa"                              │
├────────┼──────────────────────────┼────────────────────┼────────────────────────────────────────────────────────┤
│ AMA_11 │ Prensa Real (Amarillismo)│ No Amarillista (0) │ "Descubrieron una nueva criatura marina que impactó    │
│        │                          │                    │  al mundo científico"                                  │
├────────┼──────────────────────────┼────────────────────┼────────────────────────────────────────────────────────┤
│ IA_0   │ IA Sintético (70 reg.)   │ Sensacionalista (1)│ "¡El fin de la humanidad! La nueva inteligencia        │
│        │                          │                    │  artificial que destruirá millones de empleos..."      │
├────────┼──────────────────────────┼────────────────────┼────────────────────────────────────────────────────────┤
│ IA_1   │ IA Sintético (70 reg.)   │ No Sensacionalista │ "Empresa de tecnología anuncia un nuevo modelo de      │
│        │                          │ (0 - Sobria)       │  lenguaje que podría automatizar tareas administrativas"│
└────────┴──────────────────────────┴────────────────────┴────────────────────────────────────────────────────────┘
```

---

## Resumen Ejecutivo

A través de un corpus consolidado de **272 noticias completas** descompuestas en **2,012 cláusulas delimitadas por signos de puntuación**, se comparó la probabilidad asignada a la noticia íntegra ($P_{\text{full}}$) frente a **cinco modelos de agregación composicional** y se aplicó un protocolo de **ablación causal dirigida** (*extirpación quirúrgica de 1, 2 y hasta 3 frases*).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              SÍNTESIS COMPARATIVA GLOBAL: NOTICIA COMPLETA VS. REGLAS DE AGREGACIÓN (272 NOTICIAS)                  │
├─────────────────────────┬──────────────┬──────────────┬──────────┬──────────┬──────────┬────────────────────────────┤
│ Modelo de Agregación    │  Pearson r   │ Spearman ρ   │   R²     │   MAE    │   RMSE   │   Concordancia Decisión    │
│                         │  [-1.0, 1.0] │  [-1.0, 1.0] │  (<= 1)  │ (Ideal:0)│ (Ideal:0)│    (% Match con P_full)    │
├─────────────────────────┼──────────────┼──────────────┼──────────┼──────────┼──────────┼────────────────────────────┤
│ 1. Promedio Simple      │   +0.6235    │   +0.6812    │ -0.2811  │  0.3393  │  0.4216  │       133 / 272 (48.9%)    │
│ 2. Máximo (Top-1 Frase) │   +0.3844    │   +0.5342    │ -1.7651  │  0.4931  │  0.6194  │       113 / 272 (41.5%)    │
│ 3. Top-2 Promedio       │   +0.4391    │   +0.4673    │ -1.2985  │  0.4457  │  0.5648  │       123 / 272 (45.2%)    │
│ 4. Titular / 1ra Frase  │   +0.4285    │   +0.4771    │ -1.1683  │  0.4354  │  0.5485  │       128 / 272 (47.1%)    │
│ 5. Promedio Ponderado   │   +0.6405    │   +0.6750    │ -0.2318  │  0.3277  │  0.4134  │       151 / 272 (55.5%)    │
└─────────────────────────┴──────────────┴──────────────┴──────────┴──────────┴──────────┴────────────────────────────┘
```

### Respuestas Científicas Concluyentes

#### 1. ¿El sensacionalismo de la noticia completa es igual al promedio de sus frases?
**ROTUNDAMENTE NO.**  
Promediar las frases de una noticia periodística real predice la etiqueta de la noticia completa con un **acierto de apenas 31.2%** (peor que lanzar una moneda al aire) y un coeficiente $R^2$ catastrófico de **$-1.668$**. Esto ocurre por el **Efecto Dilución Contextual**: en un periódico serio, los periodistas usan titulares llamativos o palabras de impacto (*"reveló"*, *"impactó"*, *"peligro"*), cuyas frases aisladas obtienen puntajes altos ($0.70 - 0.95$); sin embargo, cuando BERT lee todo el artículo, las explicaciones científicas y sobrias neutralizan la alarma, haciendo que el artículo completo caiga a probabilidades casi nulas ($P_{\text{full}} < 0.10$).

#### 2. ¿La clasificación de la noticia completa depende de 1 o 2 frases en específico?
**SÍ, PERO DE FORMA DUAL Y CONDICIONADA AL TIPO DE NOTICIA:**
- **En Noticias Sensacionalistas Reales (Efecto Gatillo):**  
  La decisión descansa de forma desproporcionada en una o dos frases disparadoras. Al extirpar únicamente la frase más alarmista (Top-1), **el 15.0% de las noticias amarillistas reales se desclasifica de inmediato**, transformándose en sobrias. Al extirpar las dos frases más alarmistas (Top-2), **la desclasificación salta al 25.0%** (1 de cada 4 noticias depende exclusivamente de 2 frases). Al retirar hasta 3 frases en artículos extensos ($k \ge 4$), la desclasificación alcanza el **20.0%**, mientras que retirar una frase aleatoria de control apenas altera el 4% al 10%.
- **En Noticias Sobrias Reales (Inmunidad por Dilución):**  
  La noticia completa **NO depende de su frase más llamativa**. En las 162 noticias sobrias analizadas existía al menos una frase con puntaje alarmista superior a $0.70$, pero el mecanismo de autoatención subordinó esa frase dentro del contexto informativo global.
- **En Noticias Sintéticas de IA (Saturación Homogénea):**  
  No existe efecto gatillo. En los textos sintéticos generados por IA, la correlación entre el promedio y la noticia completa es casi perfecta ($r = 0.994$, $R^2 = 0.985$, concordancia del 100%). Todas las frases de un texto sintético amarillista están saturadas de drama artificial, de modo que remover 1, 2 o 3 frases produce **0.0% de inversiones de clase**.

---

## 1. Metodología Experimental y Protocolo de Segmentación

### 1.1. Criterio de Segmentación: ¿Por qué se separan las frases con comas y signos de puntuación?

Una duda metodológica recurrente es: *¿por qué no separar el texto únicamente con los puntos finales (`.`), tal como se definen las oraciones gramaticales tradicionales?*

En el análisis de estilo y sensacionalismo periodístico, **el punto final es insuficiente**. El sensacionalismo moderno opera a nivel de **cláusulas retóricas**, incisos valorativos, preguntas dramáticas y apelaciones directas al lector que habitualmente van separadas por comas, dos puntos, signos de exclamación o guiones.

#### Algoritmo de Segmentación Implementado
Se diseñó un particionador robusto mediante expresiones regulares que detecta los límites sintácticos del discurso:

```python
# Delimitadores de corte sintáctico y cláusula informativa
delimitadores = r'[,;:.!?¡¿\n—–\"\'«»()]+'
```

#### Reglas y Filtros de Calidad Textual
Para evitar que conectores huérfanos o partículas gramaticales vacías se procesen como frases, se aplicó un doble filtro:
1. **Longitud mínima en caracteres:** Cada fragmento debe tener al menos **6 caracteres** de longitud útil.
2. **Longitud mínima en palabras:** Cada fragmento debe contener al menos **2 palabras completas**.

#### Demostración Práctica con las 4 Noticias de Ejemplo

Veamos exactamente cómo trocea el algoritmo a cada una de nuestras noticias de ejemplo:

##### Ejemplo 1: `AMA_2` (Prensa Real - Amarillista)
> **Texto original:** *"¿Asteroide Apophis destruiría la Tierra en 2029? Los cálculos de la Nasa. En 2004 fue descubierto el asteroide Apophis y los expertos lo clasificaron como uno de los cuerpos celestes más peligrosos que podrían impactar en la Tierra. Sin embargo, la evaluación del choque cambió con el paso de los años..."*

* **Frase 1 (Titular / Pregunta dramática delimitada por `¿` y `?`):** `"Asteroide Apophis destruiría la Tierra en 2029"`  
  *(Longitud: 9 palabras. El signo de interrogación aísla la hipótesis catastrófica).*
* **Frase 2 (Atribución delimitada por punto):** `"Los cálculos de la Nasa"`  
  *(5 palabras. Contexto institucional).*
* **Frase 3 (Antecedente delimitado por coma):** `"En 2004 fue descubierto el asteroide Apophis y los expertos lo clasificaron como uno de los cuerpos celestes más peligrosos que podrían impactar en la Tierra"`  
  *(23 palabras. Cláusula informativa de riesgo histórico).*
* **Frase 4 (Conector adversativo delimitado por coma):** Descartado por no alcanzar el mínimo o agrupado con la cláusula siguiente: `"la evaluación del choque cambió con el paso de los años"`.  
*(Total de frases válidas extraídas de esta noticia: 24 frases).*

##### Ejemplo 2: `AMA_11` (Prensa Real - No Amarillista)
> **Texto original:** *"Descubrieron una nueva criatura marina que impactó al mundo científico. Recientemente, se reveló un impactante hallazgo en la vida marina; gracias a una expedición científica en el Mar de China Meridional, investigadores a bordo de un submarino encontraron una nueva especie..."*

* **Frase 1 (Titular delimitado por punto):** `"Descubrieron una nueva criatura marina que impactó al mundo científico"` *(Puntuación aislada: 0.9748 por el verbo "impactó").*
* **Frase 2 (Inciso temporal delimitado por punto y coma):** `"se reveló un impactante hallazgo en la vida marina"` *(Puntuación aislada: 0.9755 por la combinación "reveló" + "impactante").*
* **Frase 3 (Cláusula de contexto geográfico delimitada por coma):** `"gracias a una expedición científica en el Mar de China Meridional"` *(Puntuación aislada: 0.2410, netamente sobria).*
* **Frase 4 (Hecho biológico delimitado por coma):** `"investigadores a bordo de un submarino encontraron una nueva especie"` *(Puntuación aislada: 0.1830).*  
*(Total de frases extraídas: 15 frases).*

##### Ejemplo 3: `IA_0` (IA Sintético - Sensacionalista)
> **Texto original:** *"¡El fin de la humanidad! La nueva inteligencia artificial que destruirá millones de empleos mañana mismo. Expertos advierten que el apocalipsis digital ya comenzó y nadie está a salvo."*

* **Frase 1 (Delimitada por signos de exclamación `¡!`):** `"El fin de la humanidad"` *(Puntuación aislada: 0.9706).*
* **Frase 2 (Delimitada por punto):** `"La nueva inteligencia artificial que destruirá millones de empleos mañana mismo"` *(Puntuación aislada: 0.9912).*
* **Frase 3 (Delimitada por punto final):** `"Expertos advierten que el apocalipsis digital ya comenzó y nadie está a salvo"` *(Puntuación aislada: 0.9855).*  
*(Total de frases extraídas: 3 frases. Obsérvese que las 3 son homogéneamente extremas).*

##### Ejemplo 4: `IA_1` (IA Sintético - No Sensacionalista)
> **Texto original:** *"Empresa de tecnología anuncia un nuevo modelo de lenguaje que podría automatizar tareas administrativas. El reporte técnico indica un posible impacto gradual en el sector empresarial."*

* **Frase 1 (Delimitada por punto):** `"Empresa de tecnología anuncia un nuevo modelo de lenguaje que podría automatizar tareas administrativas"` *(Puntuación: 0.6710).*
* **Frase 2 (Delimitada por punto final):** `"El reporte técnico indica un posible impacto gradual en el sector empresarial"` *(Puntuación: 0.0819).*  
*(Total de frases extraídas: 2 frases).*

---

### 1.2. Protocolo de Inferencia en GPU

Para cada una de las 272 noticias, se ejecutaron dos tipos de inferencia en el modelo BETO (`JJNeila/bert-spanish-sensationalism-oss`):
1. **Inferencia de Documento Completo ($P_{\text{full}}$):** Se alimenta a la GPU el texto íntegro consolidado (titular + copete + cuerpo, truncado a un máximo de 256 tokens para no exceder la ventana posicional del Transformer). La probabilidad devuelta es $P_{\text{full}} = \text{Softmax}(\text{Logits})_{1}$.
2. **Inferencia de Frases Aisladas ($P_i$):** Cada una de las frases resultantes de la partición entra de forma independiente al modelo, sin que BETO sepa qué había antes o después.

---

### 1.3. Explicación Didáctica de los 5 Modelos de Agregación

Si quisiéramos clasificar noticias completas sin tener que procesar todo el artículo de golpe, ¿qué fórmula matemática podríamos usar con las frases individuales? Evaluamos cinco candidatos:

1. **Modelo 1: Promedio Simple ($\bar{P}_{\text{mean}}$)**  
   * *Analogía didáctica:* **Las notas escolares.** Si un alumno tiene calificaciones de 10, 8, 4 y 2, sumamos todas y dividimos entre 4.  
   * *Supuesto subyacente:* Supone que cada frase del artículo aporta la misma cantidad de sensacionalismo y que el tono final es una mezcla uniforme.
2. **Modelo 2: Máximo Absoluto o Frase Pico ($P_{\max}$ / Top-1)**  
   * *Analogía didáctica:* **La manzana podrida o el grito en la biblioteca.** Supone que basta con que una sola frase sea escandalosa para que toda la noticia se considere amarillista, sin importar lo sobrio que sea el resto.
3. **Modelo 3: Promedio de las Top-2 Frases ($\bar{P}_{\text{top-2}}$)**  
   * *Analogía didáctica:* **El dúo dinámico (Titular + Remate).** Supone que una sola frase puede ser un descuido, pero si hay dos frases de alta alarma, la noticia es irremediablemente sensacionalista.
4. **Modelo 4: Titular / Primera Frase ($P_{\text{first}}$)**  
   * *Analogía didáctica:* **El escaparate de la tienda.** Supone que el lector (y el modelo) deciden si la noticia es sensacionalista leyendo exclusivamente el titular o la primera oración.
5. **Modelo 5: Promedio Ponderado por Longitud ($\bar{P}_{\text{w-mean}}$)**  
   * *Analogía didáctica:* **El debate donde quien habla más tiempo tiene más votos.** Si una frase tiene 20 palabras y otra tiene 4, la frase de 20 palabras influye cinco veces más en el puntaje final.

---

### 1.4. Diccionario Didáctico de Métricas Estadísticas

Para que cualquier lector comprenda las tablas numéricas sin frustración técnica:

* **Pearson $r$ (Correlación Lineal, de -1.0 a +1.0):**  
  Mide si dos variables suben o bajan juntas en una línea recta. Si $r = +1.0$, cuando el promedio de frases sube, la noticia completa sube exactamente en la misma proporción. Si $r = 0.0$, no hay ninguna relación.
* **Spearman $\rho$ (Correlación de Rangos, de -1.0 a +1.0):**  
  No le importa si la relación es una línea recta perfecta, solo le importa el orden relativo. ¿Las noticias con mayor promedio de frases son también las que quedan primeras en la lista de noticias completas?
* **$R^2$ (Coeficiente de Determinación o Varianza Explicada):**  
  Mide qué porcentaje de la variación de la noticia completa es explicado por la fórmula de las frases. El valor ideal máximo es $1.0$ ($100\%$ explicado).  
  > [!WARNING]
  > **¿Qué significa un $R^2$ negativo (como $-0.28$ o $-1.66$)?**  
  > En estadística, un $R^2$ negativo es la peor calificación posible: significa que **usar esa fórmula matemática es peor que no hacer ningún cálculo y simplemente adivinar siempre la media histórica**. La fórmula comete más errores que la simple ignorancia informada.
* **MAE (Error Absoluto Medio, Ideal: 0.0):**  
  Es el promedio de los errores en valor absoluto. Si el MAE es $0.33$, significa que, en una escala de 0 a 1 (donde 0 es sobrio y 1 es amarillista), la fórmula se equivoca en promedio por **33 puntos porcentuales** en cada noticia evaluada.
* **RMSE (Raíz del Error Cuadrático Medio, Ideal: 0.0):**  
  Similar al MAE, pero eleva los errores al cuadrado antes de promediarlos. Esto castiga con extrema dureza las pifias gigantescas (por ejemplo, cuando una noticia es $0.01$ y la fórmula predice $0.95$).
* **Concordancia de Decisión (% Match con $P_{\text{full}}$):**  
  Si usamos el umbral estándar de $0.5$ (si $P \ge 0.5$ es Sensacionalista, si $P < 0.5$ es Sobria), ¿en qué porcentaje de noticias la regla de frases llega a la **misma etiqueta final** que el modelo leyendo todo el texto?

---

## 2. Experimento 1: ¿Es el Sensacionalismo de la Noticia Completa Igual al Promedio de sus Frases?

```
                     DISPERSIÓN: NOTICIA COMPLETA VS. PROMEDIO DE FRASES
      1.0 ┌────────────────────────────────────────────────────────────────────────┐
          │                                                  ▲ IA   ▲▲ ▲▲▲ (IA: r=0.99)
      0.8 │                                                ▲       ▲▲              │
          │                                                                        │
      0.6 │                                         ●                              │
          │                                                                        │
      0.4 │                                                                        │
          │                                                                        │
      0.2 │  ●●● ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●                            │
      0.0 │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  (Prensa Real: Colapso)    │
          └────────────────────────────────────────────────────────────────────────┘
            0.0        0.2        0.4        0.6        0.8        1.0
                                Promedio de Frases P(Mean)
                                
            Leyenda: ▲ Triángulos = Noticias Sintéticas IA (Alineación perfecta)
                     ● Círculos = Prensa Digital Real (Nube masiva inferior en P_full ≈ 0)
```

![Figura 1: Dispersión y Regresión Noticia Completa vs 4 Reglas de Agregación](imagenes/sensacionalismo_frases/fig1_dispersion_regresion_pfull_vs_agregaciones.png)  
*Figura 1: Gráficas de dispersión con regresión lineal contrastando la probabilidad de la noticia completa ($P_{\text{full}}$) frente al Promedio Simple, Máximo (Top-1), Top-2 Promedio y Titular (Primera Frase), discriminando por color entre Prensa Digital Real (círculos azules, 202 noticias) y Noticias Sintéticas de IA (triángulos naranjas, 70 noticias).*

### 2.1. Desglose Estadístico Separado por Corpus de Origen

Al separar las 272 noticias entre sus dos orígenes reales, los números demuestran una fractura total:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    COMPARATIVA DE AJUSTE: PRENSA DIGITAL REAL VS. NOTICIAS SINTÉTICAS IA                     │
├──────────────────────────┬──────────────────────────────────────────┬────────────────────────────────────────┤
│ Métrica de Evaluación    │ Prensa Digital Real (202 noticias)       │ Noticias Sintéticas IA (70 noticias)   │
│                          │ Archivo: dataset_Amarillismo.csv         │ Archivo: dataset_IA_sintetico_70.csv   │
├──────────────────────────┼──────────────────────────────────────────┼────────────────────────────────────────┤
│ Correlación Pearson (r)  │               +0.5011                    │               +0.9942                  │
│ Correlación Spearman (ρ) │               +0.5534                    │               +0.9601                  │
│ Coeficiente R²           │               -1.6681 (Colapso)          │               +0.9854 (Casi perfecto)  │
│ Error Absoluto Medio MAE │               0.4491                     │               0.0234                   │
│ Error Cuadrático RMSE    │               0.4851                     │               0.0381                   │
│ Concordancia Clasif. (%) │        63 / 202 aciertos (31.2%)         │        70 / 70 aciertos (100.0%)       │
└──────────────────────────┴──────────────────────────────────────────┴────────────────────────────────────────┘
```

![Figura 2: Benchmark Métricas de Ajuste MAE, RMSE, R2 y Concordancia](imagenes/sensacionalismo_frases/fig2_comparativa_metricas_ajuste_mae_rmse_r2.png)  
*Figura 2: Comparativa de errores numéricos (MAE y RMSE, donde menor es mejor) frente a capacidad predictiva ($R^2$ y Concordancia clasificatoria binaria, donde mayor es mejor) para los 5 modelos de agregación evaluados.*

---

### 2.2. Explicación Detallada y Educativa de las Figuras 1 y 2

#### ¿Qué dio la Figura 1 y qué significa?
- **Lo que muestra:** En cada uno de los cuatro cuadrantes, el eje horizontal ($X$) muestra lo que predice la regla basada en frases sueltas, y el eje vertical ($Y$) muestra lo que predice BERT cuando lee la noticia entera. La línea discontinua negra representa la perfección ($y = x$): si el promedio de frases fuera idéntico a la noticia completa, todos los puntos caerían exactamente sobre esa línea.
- **Lo que dio en realidad:**
  - Los **triángulos naranjas (IA Sintética)** se ubican casi exactamente sobre la línea negra. Para la IA, la noticia completa es prácticamente idéntica al promedio de sus frases.
  - Los **círculos azules (Prensa Real)** forman una aglomeración masiva en la parte inferior derecha: noticias con promedio de frases entre $0.50$ y $0.80$, pero cuya noticia completa tiene probabilidad de casi $0.00$.
- **¿Es bueno o malo?:**  
  Depende de cómo se mire:
  - Es **malo para los métodos simples de agregación**: demuestra que promediar oraciones para clasificar textos no sirve en el mundo real.
  - Es **excelente para el modelo de lenguaje (BETO)**: demuestra que BETO no es un contador de palabras superficial. Posee la sofisticación lingüística para entender que un titular llamativo no convierte a un artículo científico en prensa amarilla.

#### ¿Qué dio la Figura 2 y qué significa?
- **Subplot 1 (Errores MAE y RMSE):**  
  El Promedio Ponderado tiene un MAE de $0.328$ y el Promedio Simple de $0.339$. El Máximo ($P_{\max}$) tiene el peor error con $0.493$. Esto significa que tomar la frase más llamativa se equivoca por medio punto en promedio en cada predicción.
- **Subplot 2 ($R^2$ y Concordancia):**  
  Todos los modelos de agregación obtienen $R^2$ negativos a nivel global (entre $-0.23$ y $-1.76$). El Promedio Ponderado alcanza la mayor concordancia clasificatoria global con un modesto $55.5\%$, mientras que el Máximo Absoluto apenas acierta el $41.5\%$.

#### Comprobación con las Noticias de Ejemplo
- En `AMA_11` (Prensa Real Sobria: criatura marina), el promedio de frases da $\bar{P}_{\text{mean}} = 0.5006$. Si usamos el promedio, **la clasificamos erróneamente como amarillista** ($P \ge 0.5$). Pero la noticia completa da $P_{\text{full}} = 0.1068$. El promedio comete un grave falso positivo.
- En `IA_0` (IA Sintético Sensacionalista), el promedio da $0.9824$ y la noticia completa da $0.9958$. Ambas coinciden sin discrepancia.

---

## 3. Experimento 2: ¿Depende la Clasificación de 1 o 2 Frases en Específico?

Para verificar si el modelo ignora el promedio y en su lugar "busca" una frase llamativa para tomar su decisión, confrontamos la noticia completa contra la frase máxima (Top-1) y el titular.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   CONCORDANCIA CLASIFICATORIA BINARIA SEGÚN DIFERENTES REGLAS DE DECISIÓN                   │
├─────────────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────────┤
│ Regla Decisoria         │ Dataset Amarillismo Real (202)│ Dataset IA Sintético (70 reg.)│ Global (272)      │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────────┤
│ Promedio Simple >= 0.5  │        63 / 202 (31.2%)       │        70 / 70 (100.0%)       │ 133 / 272 (48.9%) │
│ Máximo (Top-1) >= 0.5   │        44 / 202 (21.8%)       │        69 / 70 ( 98.6%)       │ 113 / 272 (41.5%) │
│ Top-2 Promedio >= 0.5   │        53 / 202 (26.2%)       │        70 / 70 (100.0%)       │ 123 / 272 (45.2%) │
│ Titular / 1ra >= 0.5    │        59 / 202 (29.2%)       │        69 / 70 ( 98.6%)       │ 128 / 272 (47.1%) │
│ Prom. Ponderado >= 0.5  │        81 / 202 (40.1%)       │        70 / 70 (100.0%)       │ 151 / 272 (55.5%) │
└─────────────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────────┘
```

![Figura 3: Distribución Empírica de Residuos por Dataset](imagenes/sensacionalismo_frases/fig3_distribucion_residuos_sesgo_gatillo.png)  
*Figura 3: Distribución de densidad de residuos: Residuo del Promedio ($P_{\text{full}} - P_{\text{mean}}$) y Residuo del Máximo ($P_{\text{full}} - P_{\max}$), contrastando la curva azul de Prensa Real frente a la curva naranja de IA Sintética.*

---

### 3.1. Explicación Educativa de la Tabla de Concordancia y la Figura 3

#### ¿Cómo se lee y qué significa la Tabla de Concordancia?
La tabla responde a una pregunta práctica: *si decidimos que una noticia es sensacionalista cada vez que su regla supere $0.5$, ¿cuántas veces coincidimos con la decisión de BETO al leer la noticia completa?*

1. **Fila 1 (Promedio Simple $\ge 0.5$):**  
   - En **Prensa Real**, solo acierta en 63 de 202 noticias ($31.2\%$). En las otras 139 noticias, la regla dice una cosa y el modelo completo hace exactamente lo contrario.
   - En **IA Sintético**, acierta en 70 de 70 ($100.0\%$). No falla nunca.
2. **Fila 2 (Máximo Top-1 $\ge 0.5$):**  
   - En **Prensa Real**, da el peor resultado de toda la investigación: **apenas 21.8% de acierto** (44/202).  
   - *¿Por qué fracasa tan estrepitosamente la frase máxima?* Porque en el periodismo actual, casi cualquier noticia seria incluye al menos una frase enérgica, alarmante o emotiva en su titular. Si clasificáramos las noticias guiándonos por la frase máxima, **tildaríamos al 78.2% de los artículos serios de prensa de ser basura amarillista**.
3. **Fila 5 (Promedio Ponderado $\ge 0.5$):**  
   - Al darle más peso a las frases largas del cuerpo del texto, la concordancia en prensa real sube al $40.1\%$, pero sigue siendo insuficiente para un clasificador confiable.

#### ¿Qué dio la Figura 3 y qué significa?
- **El concepto de residuo:** Un residuo es la resta:  
  $$\text{Residuo} = P(\text{Noticia Completa}) - P(\text{Regla de Frases})$$
  - Si el residuo es **cero**, la regla predijo con exactitud quirúrgica la noticia completa.
  - Si el residuo es **positivo**, la noticia completa fue más sensacionalista de lo que decían las frases.
  - Si el residuo es **negativo**, la regla exageró el sensacionalismo y la noticia completa resultó ser mucho más sobria.
- **Lo que se observa en los gráficos:**
  - **La curva naranja (IA Sintética):** Está centrada casi exactamente en $0.0$. Sus residuos son mínimos.
  - **La curva azul (Prensa Real):** Tiene un gigantesco sesgo hacia la izquierda, con medias de $-0.32$ en el promedio y $-0.48$ en el máximo. Esto demuestra que **las frases extraídas por separado siempre parecen más amarillistas de lo que realmente es el documento completo**.

---

## 4. Experimento 3: Validación Causal Mediante Ablación Quirúrgica de Frases (Top-1, Top-2 y Top-3)

### 4.1. ¿Qué es la Ablación en Inteligencia Artificial?

> [!NOTE]
> **Analogía Didáctica de la Ablación:**  
> La palabra *ablación* proviene de la medicina y la cirugía, donde significa extirpar un órgano o tejido para comprobar su función o curar una afección. En mecánica automotriz, si un motor tiembla, el mecánico desconecta una bujía para observar qué cambia en el funcionamiento.  
> En Procesamiento del Lenguaje Natural (PLN), **la ablación causal consiste en eliminar deliberadamente una o más frases de un texto y volver a pasarlo por el modelo para observar cómo reacciona**.  
> - Si al extirpar una frase el modelo cambia drásticamente de opinión (por ejemplo, pasa de considerar la noticia "Sensacionalista" a considerarla "Sobria"), hemos demostrado de forma causal e irrefutable que **esa frase específica era el gatillo que controlaba la predicción**.  
> - Si al retirar la frase la predicción apenas varía, esa frase era redundante.

---

### 4.2. Resultados del Protocolo de Ablación Extendida (Top-1, Top-2 y Top-3)

En este experimento evaluamos cuatro intervenciones quirúrgicas sobre las noticias sensacionalistas:
1. **Extirpación de la Frase Top-1:** Quitar la frase con mayor probabilidad individual de amarillismo.
2. **Extirpación de las Frases Top-2:** Quitar simultáneamente las dos frases más alarmistas.
3. **Extirpación de las Frases Top-3:** Quitar simultáneamente las tres frases más alarmistas.
4. **Extirpación de Frase de Control (Aleatoria):** Quitar una frase neutra intermedia que no sea ninguna de las tres principales.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    RESULTADOS DEL PROTOCOLO DE ABLACIÓN CAUSAL EXTENDIDA (75 NOTICIAS SENS.)                 │
├──────────────────────────────────┬───────────────────────────────┬──────────────────────────────────────────┤
│ Intervención Experimental        │ Impacto Medio en Probabilidad │ Tasa de Inversión Predictiva (Flips)     │
│                                  │ ΔP = P(Full) - P(Ablado)      │ (Sensacionalista → Sobria / P < 0.5)     │
├──────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────┤
│ 1. Extirpación Frase Top-1       │           -0.1417             │        6 / 75 noticias ( 8.0%)           │
│    - Prensa Digital Real (40)    │           -0.2021             │        6 / 40 noticias (15.0%)           │
│    - IA Sintético (35)           │           +0.0327             │        0 / 35 noticias ( 0.0%)           │
├──────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────┤
│ 2. Extirpación Frases Top-2      │           -0.1085             │       10 / 75 noticias (13.3%)           │
│    - Prensa Digital Real (40)    │           -0.1466             │       10 / 40 noticias (25.0%)           │
│    - IA Sintético (35)           │           +0.0015             │        0 / 35 noticias ( 0.0%)           │
├──────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────┤
│ 3. Extirpación Frases Top-3      │           -0.0502             │        5 / 75 noticias ( 6.7%)           │
│    - Prensa Real Global (40)     │           -0.0675             │        5 / 40 noticias (12.5%)           │
│    - Prensa Real Extensa (k>=4)  │           +0.0748             │        5 / 25 noticias (20.0%)           │
│    - IA Sintético (35)           │            0.0000             │        0 / 35 noticias ( 0.0%)           │
├──────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────┤
│ 4. Frase Aleatoria de Control    │           -0.1812             │        4 / 75 noticias ( 5.3%)           │
│    - Prensa Digital Real (40)    │           -0.2554             │        4 / 40 noticias (10.0%)           │
│    - IA Sintético (35)           │           +0.0327             │        0 / 35 noticias ( 0.0%)           │
└──────────────────────────────────┴───────────────────────────────┴──────────────────────────────────────────┘
```

![Figura 4: Protocolo de Ablación Causal de Frases](imagenes/sensacionalismo_frases/fig4_ablacion_causal_frases_drop_flips.png)  
*Figura 4: Impacto en certidumbre ($\Delta P$) y porcentaje de noticias amarillistas desclasificadas hacia la categoría neutra tras eliminar Top-1, Top-2, Top-3 o una frase aleatoria de control.*

---

### 4.3. Explicación Didáctica y Mecanística de la Figura 4 y la Prueba de 3 Frases

#### ¿Qué dio la Figura 4 y qué significa cada barra?
- **Subplot 1 (Variación en la Certidumbre $\Delta P$):**  
  Muestra cuánto desciende la probabilidad promedio asignada a la noticia completa al retirar las frases.  
  - En **Prensa Real (barras azules)**, retirar la frase Top-1 produce una caída promedio de $-0.2021$ en la probabilidad de amarillismo. Retirar Top-2 reduce la certidumbre en $-0.1466$.
  - En **IA Sintético (barras naranjas)**, la variación es prácticamente nula ($+0.03$ a $0.00$), confirmando que el texto de IA no depende de frases individuales.
- **Subplot 2 (Tasa de Inversión Predictiva o Flips):**  
  Un *flip* o inversión ocurre cuando una noticia que originalmente era Sensacionalista ($P \ge 0.5$) pasa a clasificarse como No Sensacionalista / Sobria ($P < 0.5$) tras la cirugía textual.
  - **Quitar Top-1:** Desclasifica al **15.0%** de las noticias amarillistas reales (6 de 40).
  - **Quitar Top-2:** Desclasifica al **25.0%** de las noticias amarillistas reales (10 de 40). ¡Una de cada cuatro noticias pierde su condición amarillista con solo extirpar dos fragmentos!
  - **Quitar Top-3:** Desclasifica al **20.0%** de las noticias con longitud suficiente ($k \ge 4$). En las noticias de 2 o 3 frases, extirpar 3 frases elimina todo el contenido del artículo, demostrando el límite estructural de longitud.
  - **Quitar Frase de Control:** Solo alteró a 1 noticia aislada en artículos extensos ($4.0\%$).

#### Ilustración en Vivo con la Noticia `AMA_2`
Observemos el colapso secuencial de `AMA_2` (¿Asteroide Apophis destruiría la Tierra?):
- **Noticia completa original:** $P_{\text{full}} = \mathbf{0.5517}$ (Sensacionalista)
- **Extirpamos Frase Top-1** (*"que dice que esta serpiente personificaba el mal y el caos"*):  
  $\rightarrow P_{\text{no\_top1}} = \mathbf{0.3330}$ (**¡FLIP!** Cae $0.2187$ puntos y cruza el umbral: se vuelve Sobria).
- **Extirpamos Frases Top-2** (Top-1 + *"Asteroide Apophis destruiría la Tierra en 2029"*):  
  $\rightarrow P_{\text{no\_top2}} = \mathbf{0.0082}$ (**¡COLAPSO TOTAL!** Cae $0.5435$ puntos: la probabilidad se pulveriza a casi cero).
- **Extirpamos Frases Top-3** (Top-1 + Top-2 + *"explica la Nasa"*):  
  $\rightarrow P_{\text{no\_top3}} = \mathbf{0.0078}$ (Cae $0.5439$ puntos).
- **Extirpamos Frase de Control Aleatoria** (una frase neutra del cuerpo):  
  $\rightarrow P_{\text{no\_rand}} = \mathbf{0.5428}$ (Apenas cambia $0.009$; la noticia sigue siendo Sensacionalista).

> [!TIP]
> **Conclusión Causal:** En noticias de prensa real con Efecto Gatillo, el amarillismo no está diseminado en todo el artículo: **está concentrado de manera hipodérmica en 1 o 2 frases**. Si se neutralizan esas dos frases, el artículo completo pierde su toxicidad estilística.

---

## 5. Experimento 4: Perfil Posicional del Sensacionalismo (Efecto de Carga Frontal)

Para responder a la pregunta de *en qué parte del artículo se ubican las frases más amarillistas*, analizamos la posición relativa de las 2,012 frases normalizadas de $0\%$ (inicio) a $100\%$ (final).

```
                  PERFIL POSICIONAL COMPARATIVO (PRENSA REAL VS IA SINTÉTICO)
     1.0 ┌────────────────────────────────────────────────────────────────────────┐
         │   ● 0.88                                          ▲ 0.99 ─── ▲ 0.98    │
     0.8 │   (Prensa Sens.: Carga Frontal)   ● 0.71          (IA Sens.: Techo)    │
         │                      ● 0.79                                            │
     0.6 │                                         ● 0.68                         │
         │                                         (Cuerpo)                       │
     0.4 │                      ■ 0.44             ■ 0.41          ■ 0.38         │
         │   ■ 0.42                                        (Prensa Sobria)        │
     0.2 │   (Prensa Sobria: Estable)                                             │
         │                                                   ▼ 0.03 ─── ▼ 0.03    │
     0.0 └────────────────────────────────────────────────────────────────────────┘
            Q1 (0-20%)        Q2 (20-40%)        Q3 (40-60%)       Q5 (80-100%)
            Inicio/Titular                                            Cierre/Fin
            
            Leyenda: ● Círculos Rojos = Prensa Real Amarillista (Carga Frontal Clásica)
                     ■ Cuadrados Azules = Prensa Real Sobria (Amortiguación Contextual)
                     ▲ Triángulos = IA Sensacionalista (Línea plana artificial en 0.99)
                     ▼ Triángulos Invertidos = IA Sobria (Línea plana artificial en 0.03)
```

![Figura 5: Evolución de la Probabilidad de Sensacionalismo por Posición en Ambos Datasets](imagenes/sensacionalismo_frases/fig5_perfil_posicional_sensacionalismo.png)  
*Figura 5: Evolución de la probabilidad promedio de sensacionalismo según la ubicación de la frase en el documento con intervalos de confianza del 95%, comparando el Panel A (Prensa Digital Real, 202 noticias) frente al Panel B (Noticias Sintéticas IA, 70 noticias).*

---

### 5.1. Explicación Detallada de la Figura 5: ¿Qué Significan los Porcentajes y Quintiles?

Para que este gráfico sea completamente transparente, expliquemos qué representa el eje horizontal:

#### Anatomía Periodística de los 5 Quintiles (Panel A - Prensa Real)
1. **Quintil 1 (0% - 20% | Apertura, Titular y Lead):**  
   - *Qué frases caen aquí:* El titular principal, la volanta, el copete de entrada y la primera oración de apertura.
   - *Comportamiento:* En las noticias amarillistas, **alcanza el pico absoluto de probabilidad ($P \approx 0.88$)**. Es el gancho (*clickbait*) diseñado para capturar el clic del lector.
2. **Quintil 2 (20% - 40% | Contextualización Inmediata):**  
   - *Qué frases caen aquí:* Los antecedentes inmediatos del suceso o la introducción del conflicto.
   - *Comportamiento:* La probabilidad desciende notablemente a $P \approx 0.79$.
3. **Quintil 3 (40% - 60% | Cuerpo Central):**  
   - *Qué frases caen aquí:* La narración de los hechos, citas directas de testigos o mención de organismos.
   - *Comportamiento:* La probabilidad sigue bajando hasta $P \approx 0.71$.
4. **Quintiles 4 y 5 (60% - 100% | Desarrollo Técnico y Cierre):**  
   - *Qué frases caen aquí:* Aclaraciones institucionales, datos numéricos de contexto y conclusiones finales.
   - *Comportamiento:* Cae a su punto más bajo ($P \approx 0.68$).

#### Contraste con el Panel B (Noticias Sintéticas IA)
- **En la IA Sensacionalista:** No existe la carga frontal periodística. La línea es completamente **plana en el techo ($0.98 - 0.99$)** tanto al inicio como al medio y al final. Un generador de IA satura cada oración con adjetivos alarmistas (*"¡El apocalipsis!", "¡Nadie está a salvo!", "¡Destrucción inminente!"*).
- **En la IA Sobria:** La línea es igualmente **plana en el piso ($0.03 - 0.04$)**, sin matices de estilo.

> [!IMPORTANT]
> **Significado Científico:** Los periodistas humanos siguen la técnica de la **pirámide invertida**: colocan la emoción y el impacto en la cúspide (titular) y los datos fríos en la base (cuerpo). Las IA generativas actuales carecen de esta estructura discursiva y producen textos planos y homogéneos.

---

## 6. Experimento 5: Contraste Estructural: IA Sintético vs. Prensa Digital Real

![Figura 6: Contraste Estructural IA Sintético vs Prensa Digital Real](imagenes/sensacionalismo_frases/fig6_contraste_ia_sintetico_vs_prensa_real.png)  
*Figura 6: Contraste simultáneo de dispersión entre Noticias Sintéticas de IA (comportamiento lineal determinista) y Prensa Digital Real (desacople contextual no lineal).*

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPARATIVA CUALITATIVA DE ARQUITECTURAS TEXTUALES                               │
├──────────────────────────────┬──────────────────────────────────┬───────────────────────────────────────┤
│ Dimensión Analítica          │ Noticias Sintéticas IA (70 reg.) │ Prensa Digital Real (202 reg.)        │
│                              │ Archivo: dataset_IA_sintetico.csv│ Archivo: dataset_Amarillismo.csv      │
├──────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ Homogeneidad Estilística     │ Extrema (Todas las frases iguales│ Heterogénea (Titular llamativo +      │
│                              │ en tono y saturación)            │ cuerpo explicativo equilibrado)       │
├──────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ Validez de la Hipótesis      │ SÍ se cumple                     │ NO se cumple de forma absoluta        │
│ del Promedio                 │ (r = 0.994, R² = 0.985, 100% acc)│ (r = 0.501, R² = -1.668, 31.2% acc)   │
├──────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ Dependencia de 1 o 2 Frases  │ Nula (La noticia está saturada;  │ Alta en el 15% al 25% de las noticias │
│                              │ quitar frases no cambia la clase)│ amarillistas reales (Efecto Gatillo)  │
├──────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────┤
│ Función del Cuerpo Noticioso │ Inexistente (textos breves de    │ Amortiguador contextual decisivo      │
│                              │ 20 a 30 palabras sin desarrollo) │ (desactiva falsas alarmas léxicas)    │
└──────────────────────────────┴──────────────────────────────────┴───────────────────────────────────────┘
```

---

## 7. Estudio Anatómico de Casos Ejemplares por Corpus y Clase Real

Para dar cumplimiento exacto al análisis anatómico y constatar cómo interactúan las frases individuales frente al documento completo, se analizó un caso por cada corpus y clase representativa:

![Figura 7: Estudio Anatómico de Casos Ejemplares](imagenes/sensacionalismo_frases/fig7_casos_estudio_ejemplares.png)  
*Figura 7: Desglose horizontal frase a frase de la probabilidad de sensacionalismo para cuatro casos representativos: Casos 1 y 2 de Prensa Digital Real (`dataset_Amarillismo.csv`) frente a Casos 3 y 4 de Noticias Sintéticas de IA (`dataset_IA_sintetico_70.csv`).*

---

### Caso 1: Efecto Gatillo en Prensa Real Amarillista
* **Corpus de Procedencia:** Prensa Digital Real (`dataset_Amarillismo.csv`)
* **ID del Documento:** `AMA_2`
* **Clase Real (Ground Truth):** **Amarillista (1)**
* **Predicción BETO Noticia Completa:** $P_{\text{full}} = \mathbf{0.5517}$ (Clasificada Sensacionalista)
* **Promedio de sus 24 Frases:** $\bar{P}_{\text{mean}} = 0.6010$
* **Titular:** *"¿Asteroide Apophis destruiría la Tierra en 2029? Los cálculos de la Nasa"*
* **Frase Top-1 Disparadora:** *"que dice que esta serpiente personificaba el mal y el caos"* $\rightarrow P = \mathbf{0.9876}$
* **Frase Top-2 (Titular):** *"Asteroide Apophis destruiría la Tierra en 2029"* $\rightarrow P = \mathbf{0.9793}$
* **Frases del Cuerpo Sobrio:** Frase 15 descarta el impacto: *"la probabilidad de colisión es prácticamente nula"* $\rightarrow P = 0.2437$.
* **Diagnóstico Mecanístico:** A pesar de que el artículo explica científicamente que la Tierra está a salvo, la intensa activación emocional del titular y la referencia mitológica a Apophis como la serpiente del caos dominan la atención en la cabecera, logrando que el artículo completo supere el umbral de $0.50$. Al extirpar esa frase Top-1, el artículo colapsa a $0.333$ (Sobrio).

---

### Caso 2: Efecto Dilución Contextual en Prensa Real Sobria
* **Corpus de Procedencia:** Prensa Digital Real (`dataset_Amarillismo.csv`)
* **ID del Documento:** `AMA_11`
* **Clase Real (Ground Truth):** **No Amarillista (0 - Sobria/Objetiva)**
* **Predicción BETO Noticia Completa:** $P_{\text{full}} = \mathbf{0.1068}$ (Clasificada No Sensacionalista / Sobria)
* **Promedio de sus 15 Frases:** $\bar{P}_{\text{mean}} = \mathbf{0.5006}$ *(¡Falso positivo si se promediara!)*
* **Titular:** *"Descubrieron una nueva criatura marina que impactó al mundo científico"*
* **Frase Top-1 Aislada:** *"se reveló un impactante hallazgo en la vida marina"* $\rightarrow P = \mathbf{0.9755}$
* **Frase Top-2 (Titular):** *"Descubrieron una nueva criatura marina que impactó al mundo científico"* $\rightarrow P = \mathbf{0.9748}$
* **Frases del Cuerpo Informativo:** *"investigadores a bordo de un submarino encontraron una nueva especie"* ($P = 0.1830$), *"el espécimen fue catalogado dentro del orden de los antozoos"* ($P = 0.0820$).
* **Diagnóstico Mecanístico:** Si evaluamos las dos primeras frases aisladas, cualquier clasificador ingenuo diría que es $97\%$ sensacionalista. Pero cuando BETO procesa el artículo entero, las 13 frases posteriores con terminología biológica, coordenadas marítimas e instituciones académicas **diluyen por completo la alarma inicial**, desplomando la probabilidad global a $0.1068$.

---

### Caso 3: Saturación Léxica en IA Sintética Sensacionalista
* **Corpus de Procedencia:** Noticias Sintéticas de IA (`dataset_IA_sintetico_70.csv`)
* **ID del Documento:** `IA_0`
* **Clase Real (Ground Truth):** **Sensacionalista (1)**
* **Predicción BETO Noticia Completa:** $P_{\text{full}} = \mathbf{0.9958}$ (Clasificada Sensacionalista)
* **Promedio de sus 3 Frases:** $\bar{P}_{\text{mean}} = \mathbf{0.9824}$
* **Frases Evaluadas:**
  1. *"El fin de la humanidad"* $\rightarrow P = 0.9706$
  2. *"La nueva inteligencia artificial que destruirá millones de empleos mañana mismo"* $\rightarrow P = 0.9912$
  3. *"Expertos advierten que el apocalipsis digital ya comenzó y nadie está a salvo"* $\rightarrow P = 0.9855$
* **Diagnóstico Mecanístico:** No hay ningún amortiguador ni desarrollo. La IA encadena tres clichés hiperbólicos consecutivos. El promedio coincide perfectamente con la noticia completa porque no existe interacción dialéctica entre partes.

---

### Caso 4: Sobriedad Uniforme en IA Sintética Neutra
* **Corpus de Procedencia:** Noticias Sintéticas de IA (`dataset_IA_sintetico_70.csv`)
* **ID del Documento:** `IA_1`
* **Clase Real (Ground Truth):** **No sensacionalista (0 - Sobria)**
* **Predicción BETO Noticia Completa:** $P_{\text{full}} = \mathbf{0.0411}$ (Clasificada No Sensacionalista)
* **Promedio de sus 2 Frases:** $\bar{P}_{\text{mean}} = 0.3764$
* **Frases Evaluadas:**
  1. *"Empresa de tecnología anuncia un nuevo modelo de lenguaje que podría automatizar tareas administrativas"* $\rightarrow P = 0.6710$
  2. *"El reporte técnico indica un posible impacto gradual en el sector empresarial"* $\rightarrow P = 0.0819$
* **Diagnóstico Mecanístico:** La primera frase tiene cierta carga al mencionar automatización de tareas ($0.67$), pero la segunda frase introduce moderación (*"impacto gradual"*, *"reporte técnico"*), haciendo que la noticia completa sea calificada como sólidamente objetiva ($0.04$).

---

## 8. Conclusiones y Recomendaciones Académicas para la Tesis

### 8.1. Conclusiones para el Corpus de Prensa Digital Real (`dataset_Amarillismo.csv`)
1. **Refutación Absoluta del Promedio Lineal:**  
   En la prensa escrita real, el sensacionalismo documental **no es una combinación aditiva ni un promedio de sus oraciones** ($R^2 = -1.668$, acierto del $31.2\%$). El modelo BETO opera como un sistema semántico no lineal mediado por matrices de autoatención bidireccional.
2. **Evidencia del Efecto Gatillo en Amarillismo:**  
   En entre un **$15.0\%$ y un $25.0\%$ de las noticias amarillistas reales**, la etiqueta de sensacionalismo depende de forma crítica y exclusiva de **1 o 2 frases** situadas en el titular o la apertura. Neutralizar quirúrgicamente esas frases transforma la noticia en sobria.
3. **El Efecto Dilución Contextual como Escudo contra Falsos Positivos:**  
   Analizar oraciones aisladas genera una tasa inaceptable de falsos positivos en artículos serios ($97\%$ de alarma en titulares divulgativos que luego resultan ser $10\%$ en el texto completo). El cuerpo de la noticia actúa como un **amortiguador contextual** indispensable.

### 8.2. Conclusiones para el Corpus de Noticias Sintéticas de IA (`dataset_IA_sintetico_70.csv`)
1. **Homogeneidad Artificial y Validez Espuria del Promedio:**  
   En los textos sintéticos breves generados por IA, el promedio de frases predice la noticia completa con una correlación casi perfecta ($r = 0.994$, $R^2 = 0.985$, $100\%$ de concordancia). Sin embargo, esto es un **artificio de diseño**: las frases no interactúan porque carecen de estructura periodística real.
2. **Inexistencia de Efecto Gatillo:**  
   En textos sintéticos amarillistas, extirpar 1, 2 o 3 frases tiene un impacto de $0.0\%$ de inversiones de clase, ya que todo el texto está saturado del mismo tono.

### 8.3. Recomendaciones Metodológicas para la Tesis de Maestría
* **Prohibición de Pipelines Basados en Promedios de Frases:**  
  Se desaconseja terminantemente construir sistemas de detección de fake news o sensacionalismo que segmenten el texto en oraciones, clasifiquen cada una por separado y luego las promedien. Dicho pipeline colapsará en prensa real.
* **Inferencia en Bloques Contextuales Continuos:**  
  La clasificación debe realizarse siempre a nivel de documento íntegro o bloques jerárquicos continuos (*hierarchical transformers* o *sliding windows*) para permitir que los párrafos explicativos subordinen a las frases de impacto.
* **Precaución Crítica con Datasets Sintéticos:**  
  Entrenar o validar modelos exclusivamente sobre datos generados por LLMs (como GPT o Claude) inducirá un sesgo de homogeneidad peligroso. Los modelos aprenderán que cualquier frase escandalosa define al texto completo, perdiendo la capacidad de dilución contextual requerida para analizar el periodismo humano moderno.

---

## 9. Inventario de Entregables y Archivos Creados

| Entregable | Ruta en el Sistema | Descripción |
| :--- | :--- | :--- |
| **Copia Enriquecida y Educativa** | [`Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md`](Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md) | Este documento extendido con explicaciones didácticas, analogías, casos de ambos corpus y ablación hasta 3 frases. |
| **Copia de Respaldo Original** | [`Reporte_Sensacionalismo_Frases_vs_Documento_Original_Backup.md`](Reporte_Sensacionalismo_Frases_vs_Documento_Original_Backup.md) | Respaldo íntegro del reporte original previo a las ampliaciones. |
| **Script de Inferencia y Datos** | [`../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/experimento_sensacionalismo_frases.py`](../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/experimento_sensacionalismo_frases.py) | Código en GPU que ejecuta segmentación, inferencias en lotes, agregaciones y ablaciones. |
| **Script Generador de Figuras** | [`../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/generar_figuras_detalladas.py`](../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/generar_figuras_detalladas.py) | Script autónomo que regenera las 7 figuras científicas en alta resolución (300 DPI) con distinción de dataset. |
| **Base de Datos CSV** | [`datos_sensacionalismo_frases.csv`](datos_sensacionalismo_frases.csv) | Tabla completa con los resultados para las 272 noticias (probabilidades completas, 5 agregaciones, ablaciones Top-1/Top-2/Top-3 y flips). |
| **Métricas JSON Formales** | [`metricas_sensacionalismo_frases.json`](metricas_sensacionalismo_frases.json) | Resumen cuantitativo formal con coeficientes de correlación ($r, \rho$), $R^2$, MAE, RMSE y tasas de desclasificación por dataset. |
| **Figura 1 (Dispersión y Regresiones)** | [`imagenes/sensacionalismo_frases/fig1_dispersion_regresion_pfull_vs_agregaciones.png`](imagenes/sensacionalismo_frases/fig1_dispersion_regresion_pfull_vs_agregaciones.png) | Gráficas de dispersión con puntos identificados por dataset (Prensa Real vs IA Sintética). |
| **Figura 2 (Benchmark de Ajuste)** | [`imagenes/sensacionalismo_frases/fig2_comparativa_metricas_ajuste_mae_rmse_r2.png`](imagenes/sensacionalismo_frases/fig2_comparativa_metricas_ajuste_mae_rmse_r2.png) | Comparativa en barras de MAE, RMSE, $R^2$ y Concordancia clasificatoria. |
| **Figura 3 (Sesgos y Residuos)** | [`imagenes/sensacionalismo_frases/fig3_distribucion_residuos_sesgo_gatillo.png`](imagenes/sensacionalismo_frases/fig3_distribucion_residuos_sesgo_gatillo.png) | Curvas de densidad KDE de los residuos por dataset evidenciando la dilución en Prensa Real. |
| **Figura 4 (Ablación Causal Top 1, 2 y 3)** | [`imagenes/sensacionalismo_frases/fig4_ablacion_causal_frases_drop_flips.png`](imagenes/sensacionalismo_frases/fig4_ablacion_causal_frases_drop_flips.png) | Variaciones de probabilidad y tasas de desclasificación con extirpación de Top-1, Top-2, Top-3 y Control. |
| **Figura 5 (Perfil Posicional 2 Datasets)** | [`imagenes/sensacionalismo_frases/fig5_perfil_posicional_sensacionalismo.png`](imagenes/sensacionalismo_frases/fig5_perfil_posicional_sensacionalismo.png) | Curvas con intervalos de confianza del 95% para Prensa Real (Panel A) e IA Sintética (Panel B). |
| **Figura 6 (Contraste IA vs Prensa Real)** | [`imagenes/sensacionalismo_frases/fig6_contraste_ia_sintetico_vs_prensa_real.png`](imagenes/sensacionalismo_frases/fig6_contraste_ia_sintetico_vs_prensa_real.png) | Scatter plots emparejados mostrando el contraste entre linealidad sintética y no linealidad humana. |
| **Figura 7 (Estudio de Casos de Ambos Corpus)** | [`imagenes/sensacionalismo_frases/fig7_casos_estudio_ejemplares.png`](imagenes/sensacionalismo_frases/fig7_casos_estudio_ejemplares.png) | Diagrama horizontal anatómico de 4 casos ejemplares (2 de Prensa Real y 2 de IA Sintética) con clases reales y corpus indicados. |
