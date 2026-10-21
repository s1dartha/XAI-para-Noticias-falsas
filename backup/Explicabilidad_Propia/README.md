# Módulo de Explicabilidad Forense y Extracción de Características Textuales

**Ubicación:** `/home/ubuntu/Documentos/Tesis/Explicabilidad_Propia`  
**Tesis:** Detección de Desinformación y Análisis Forense del Discurso en Español  
**Enfoque Metodológico:** Diagnóstico Estilométrico Multi-Nivel sin Cajas Negras ni Veredictos Binarios

---

## 1. Propósito y Replanteamiento Epistemológico

La mayoría de los clasificadores de desinformación actuales pretenden resolver el problema mediante una caja negra que emite una probabilidad binaria:
$$\text{Texto} \to [0: \text{Verdadero}, \ 1: \text{Falso}]$$

Como se fundamenta en la investigación de esta tesis, esta formulación es insostenible:
1. **La Verdad Factual es Extrínseca:** Que un evento sea real o falso depende de su correspondencia con la realidad empírica externa, no de la disposición léxica de las oraciones.
2. **Las Dos Paradojas del Periodismo:**
   - *Paradoja 1 (Propaganda Sobria / Astroturfing):* Bulos redactados deliberadamente en un tono frío, académico o institucional, sin signos de admiración ni adjetivos llamativos.
   - *Paradoja 2 (Periodismo de Choque Legítimo):* Noticias 100% verídicas sobre desastres, descubrimientos científicos o denuncias judiciales que emplean titulares dramáticos o enfáticos.

### Objetivo de este Módulo
Proporcionar un **Método Explicable de Auditoría Forense** que, ante cualquier noticia entrante en español, extraiga y reporte:
- **Las 5 Dimensiones del Índice de Manipulación Lingüística (IML)** normalizadas en $[0, 100]$.
- **El Modelo de Sensacionalismo Documental y Oracional (BETO)**, con cálculo de oraciones gatillo y ablación causal contrafáctica.
- **El Modelo de Redundancia Semántica Intra-Documental (SBERT)**, con matriz de similitud coseno, identificación de bucles y discretización entrópica de Shannon.
- **Las Variables Enriquecidas de las 5 Dimensiones del Ensamble Avanzado** (flujo secuencial, lingüística forense/epistémica, riqueza léxica/legibilidad, morfosintaxis spaCy y anclajes factuales).
- **Desglose Local Frase a Frase**, identificando oraciones gatillo, banderas forenses, citas, fuentes y saltos temáticos.
- **Sin clasificación binaria de caja negra:** Ofrece una ficha de auditoría forense para asistir el juicio crítico de periodistas y verificadores.

---

## 2. Arquitectura de Archivos en `Tesis/Explicabilidad_Propia`

```
Tesis/Explicabilidad_Propia/
├── metodo_explicabilidad.py           # Clase central AuditorExplicabilidadNoticia con todo el pipeline forense
├── ejecutar_auditoria_ejemplos.py     # Script que ejecuta la auditoría sobre las 4 noticias de control y compila el informe maestro
├── auditar_texto_personalizado.py     # Interfaz CLI / script para auditar cualquier texto o archivo externo
├── generar_visualizaciones.py         # Generador de gráficos y radiografías estilométricas (300 DPI)
├── README.md                          # Guía técnica y metodológica completa
├── codigos/                           # Scripts especializados del perfilador forense
│   ├── perfilador_estilo_manipulacion.py         # Perfilador de los 5 arquetipos y las 5 dimensiones IML
│   └── perfiles_estilos_manipulacion_2604.csv    # Base de datos con los perfiles calculados para 2.604 noticias
├── graficos/                          # Figuras científicas en alta resolución
│   ├── fig_radar_iml_4_noticias.png           # Radar chart de las 5 dimensiones del IML
│   ├── fig_trayectoria_sensacionalismo.png    # Dinámica oracional y detección de gatillos
│   ├── fig_mapas_calor_redundancia.png        # Matrices de calor de similitud coseno SBERT
│   ├── fig_comparativa_variables_clave.png    # Comparativa morfosintáctica y de anclajes
│   ├── fig4_radar_estilos_manipulacion.png    # Radar de los 5 arquetipos discursivos teóricos
│   └── fig5_tasa_fakenews_por_arquetipo.png   # Tasa de fake news e IML por arquetipo en 2.604 noticias
└── reportes/                          # Informes y datos serializados
    ├── Reporte_Explicabilidad_Forense_4_Noticias.md  # Gran informe maestro y exhaustivo en Markdown
    ├── Reporte_Estilos_Manipulacion_Textual.md       # Informe de investigación sobre los 5 arquetipos discursivos
    ├── auditoria_completa_4_noticias.json            # JSON consolidado con todas las métricas
    ├── metricas_estilos_manipulacion.json            # Resumen cuantitativo por arquetipo
    ├── noticia_1_falsa_salud_mayusculas.json         # JSON detallado Noticia 1
    ├── noticia_2_falsa_conspiracion_coronacirco.json # JSON detallado Noticia 2
    ├── noticia_3_verdadera_puebla_salud.json         # JSON detallado Noticia 3
    └── noticia_4_verdadera_ciencia_efe.json          # JSON detallado Noticia 4
```

---

## 3. Las 5 Dimensiones del Índice de Manipulación Lingüística (IML)

Formalizadas en una escala de $[0.0, 100.0]$:

1. **Dimensión 1: Carga Emocional / Sensacionalismo ($D_1 \in [0, 100]$):**  
   $$D_1 = \text{clip}(P_{\text{full}} \times 100, 0, 100)$$  
   Mide la densidad de superlativos, adjetivos de grado y dramatismo léxico asignado por el modelo BETO entrenado en amarillismo.

2. **Dimensión 2: Volatilidad de Gatillo / Desbalance Cabecera-Cuerpo ($D_2 \in [0, 100]$):**  
   $$D_2 = \min(100.0, 1.5 \times \max(0, P_{\max} - \bar{P}_{\text{mean}}) \times 100)$$  
   Cuantifica qué tan concentrado está el dramatismo en una o dos frases aisladas frente a la sobriedad media del artículo.

3. **Dimensión 3: Amortiguación Contextual / Resiliencia ($D_3 \in [0, 100]$):**  
   $$D_3 = \begin{cases} \max(0, 1.0 - DR) \times 100 & \text{si } P_{\max} \ge 0.50 \\ 100.0 & \text{en otro caso} \end{cases}$$  
   Donde $DR = \frac{P_{\text{full}}}{P_{\max} + 10^{-6}}$ es el *Dilution Ratio*. Mide la capacidad del cuerpo noticioso para neutralizar el impacto del titular mediante datos técnicos y contexto.

4. **Dimensión 4: Reiteración y Bucle Argumental / Redundancia ($D_4 \in [0, 100]$):**  
   $$D_4 = \text{clip}\left(\frac{S_{\max} - 0.34}{0.90 - 0.34} \times 100, 0, 100\right)$$  
   A partir de la similitud coseno máxima intra-documental de SBERT ($S_{\max}$). Detecta el bucle retórico calculado para inducir el efecto psicológico de verdad ilusoria (*Illusory Truth Effect*).

5. **Dimensión 5: Cohesión y Fluidez Discursiva ($D_5 \in [0, 100]$):**  
   $$D_5 = \begin{cases} 
   \frac{S_{\max}}{0.5924} \times 60.0 & \text{si } S_{\max} \le 0.5924 \text{ (Desarticulado)} \\ 
   90.0 + \left(1.0 - \frac{|S_{\max} - 0.625|}{0.18}\right) \times 10.0 & \text{si } 0.5924 < S_{\max} \le 0.8077 \text{ (Banda Óptima Shannon)} \\ 
   \max\left(30.0, 100.0 - \frac{S_{\max} - 0.8077}{0.19} \times 70.0\right) & \text{si } S_{\max} > 0.8077 \text{ (Bucle circular)} 
   \end{cases}$$

### Score Global IML
$$\text{IML} = 0.35 \cdot D_1 + 0.20 \cdot D_2 + 0.15 \cdot (100 - D_3) + 0.20 \cdot D_4 + 0.10 \cdot (100 - D_5)$$
- **$\text{IML} \le 35.0$:** Bajo Riesgo de Manipulación Estilométrica (Texto Neutro / Sobrio).
- **$35.0 < \text{IML} \le 60.0$:** Riesgo Moderado de Manipulación (Retórica Comercial o Polarizada).
- **$\text{IML} > 60.0$:** Alto Riesgo de Manipulación Estilométrica (Anomalía Forense Severa).

### Los 5 Arquetipos Discursivos
- **Estilo I:** Desinformación Estridente / Cliché Hiperbólico ($D_1 \ge 65 \land D_4 \ge 65$)
- **Estilo II:** Cebo Comercial / Clickbait de Cabecera ($D_2 \ge 50 \land D_3 \ge 50$)
- **Estilo III:** Propaganda Institucional / Astroturfing ($D_1 < 45 \land D_4 \ge 65$)
- **Estilo IV:** Incoherencia Estructural / Generación Rota ($D_5 \le 55 \land D_4 < 40$)
- **Estilo V:** Periodismo Profesional Balanceado (Equilibrado sin anomalías severas)

---

## 4. Instrucciones de Uso

### A. Ejecución de las 4 Noticias de Control
Para regenerar toda la auditoría y figuras sobre las 4 noticias de ejemplo:
```bash
/home/ubuntu/Documentos/Tesis/venv_pipeline/bin/python /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/ejecutar_auditoria_ejemplos.py
```

### B. Auditar una Noticia Personalizada desde la Terminal
```bash
# Mediante argumento de texto directo:
/home/ubuntu/Documentos/Tesis/venv_pipeline/bin/python /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/auditar_texto_personalizado.py \
    --texto "Texto completo de la noticia aquí..." \
    --titulo "Mi Noticia a Auditar" \
    --salida_json mi_auditoria.json \
    --salida_md mi_informe.md

# Mediante archivo .txt:
/home/ubuntu/Documentos/Tesis/venv_pipeline/bin/python /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/auditar_texto_personalizado.py \
    --archivo /ruta/a/mi_noticia.txt \
    --salida_json mi_auditoria.json
```

### C. Uso como Módulo Python en Otros Scripts
```python
import sys
sys.path.append("/home/ubuntu/Documentos/Tesis/Explicabilidad_Propia")

from metodo_explicabilidad import AuditorExplicabilidadNoticia

# 1. Instanciar auditor (carga BETO, SBERT y spaCy en GPU/CPU)
auditor = AuditorExplicabilidadNoticia()

# 2. Auditar noticia (sin clasificar, devuelve diccionario completo)
resultado = auditor.auditar_noticia(
    texto="Texto de la noticia...",
    titulo="Titular opcional"
)

# 3. Acceder a las dimensiones IML
print("Score IML:", resultado["las_5_dimensiones_iml"]["score_global_iml"])
print("Arquetipo:", resultado["las_5_dimensiones_iml"]["arquetipo_discursivo"]["nombre"])

# 4. Acceder al sensacionalismo y gatillos
print("Sensacionalismo Global:", resultado["sensacionalismo_beto"]["P_full"])
print("Gatillo Principal:", resultado["sensacionalismo_beto"]["oracion_gatillo_principal"])

# 5. Acceder a la redundancia SBERT
print("Similitud Máxima:", resultado["redundancia_sbert"]["max_intra_similarity"])
print("Par Redundante:", resultado["redundancia_sbert"]["par_maxima_redundancia"])

# 6. Generar reporte Markdown formateado
informe_md = auditor.formatear_informe_markdown(resultado)
print(informe_md)
```

---

## 5. Mapa de Dependencias: ¿Qué Otros Archivos se Necesitan para Entender Este?

1. **Marco Teórico y Replanteamiento Epistemológico:**  
   [`Plan_tesis.md`](../Plan_tesis.md) — Fundamentación de por qué la verdad es extrínseca al texto y refutación de la clasificación binaria de caja negra.
2. **Formulación Matemática del IML y Arquetipos Discursivos:**  
   [`Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Estilos_Manipulacion_Textual.md`](../Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Estilos_Manipulacion_Textual.md) — Deducción teórica de las 5 dimensiones IML, calibración empírica en 2.604 noticias y Figuras 4 y 5.
3. **Ensamble Adaptativo y Ranking de Importancia SHAP:**  
   [`Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Ensamble_Adaptativo_FakeNews.md`](../Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/Reporte_Ensamble_Adaptativo_FakeNews.md) — Calibración del umbral bayesiano ($\theta = 0.42$) y prueba de que las variables morfosintácticas superan a los embeddings puros.
4. **Catálogo Ontológico de las 36 Variables:**  
   [`dashboard/diccionario_variables_5d.json`](../dashboard/diccionario_variables_5d.json) — Definiciones, fórmulas, hipótesis periodísticas y rankings de importancia de cada variable.
5. **Códigos Fuente de Extracción en GPU:**  
   [`Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/extraer_features_avanzadas_5d.py`](../Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/extraer_features_avanzadas_5d.py) y [`extraer_features_ensamble_ampliado.py`](../Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/extraer_features_ensamble_ampliado.py) — Extracción en masa sobre 4.418 noticias con PyTorch, BETO, SBERT y spaCy.
6. **Calibración GMM de Redundancia:**  
   [`Modelos_Individuales/Redundancia/Reportes/`](../Modelos_Individuales/Redundancia/Reportes/) — Ajuste del umbral $\tau = 0.34$ mediante Mezclas Gaussianas y partición entrópica de Shannon.
7. **Estabilidad y Ablaciones de Sensacionalismo:**  
   [`Modelos_Individuales/Sensacionalismo/Pruebas_Frases/`](../Modelos_Individuales/Sensacionalismo/Pruebas_Frases/) — Validación de BETO ante perturbaciones y ablaciones de gatillos oracionales.

