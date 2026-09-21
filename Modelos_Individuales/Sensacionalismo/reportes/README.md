# 📊 Estructura y Mapa de Trazabilidad de Reportes e Imágenes: Sensacionalismo

Este directorio consolida todos los reportes formales, métricas numéricas, conjuntos de datos experimentales y artefactos visuales correspondientes al módulo de **Sensacionalismo** dentro del proyecto de Tesis.

---

## 🗂️ 1. Mapa General de Trazabilidad (¿De dónde viene cada archivo?)

| Archivo / Carpeta | Tipo | Script / Cuadernillo Generador | Datasets Utilizados | Reporte Académico Asociado | Descripción |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`Reporte_Sensacionalismo.md`** | Reporte `.md` | [`3sensacionalismo.ipynb`](../3sensacionalismo.ipynb) | • `dataset_Salud.csv`<br>• `dataset_Amarillismo.csv`<br>• `dataset_IA_sintetico_70.csv` | Este mismo documento | Evaluación comparativa de los 3 modelos base (BETO supervisado, mDeBERTa Zero-Shot, RoBERTuito emociones). |
| **`Reporte_Sensacionalismo_Frases_vs_Documento.md`** | Reporte `.md` | [`Pruebas_Frases/experimento_sensacionalismo_frases.py`](../Pruebas_Frases/experimento_sensacionalismo_frases.py) | • `dataset_Amarillismo.csv`<br>• `dataset_IA_sintetico_70.csv` (272 noticias) | Este mismo documento | Estudio experimental formal sobre la disociación entre noticias completas y análisis a nivel de oración. |
| **`Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md`** | Reporte `.md` | [`Pruebas_Frases/generar_figuras_detalladas.py`](../Pruebas_Frases/generar_figuras_detalladas.py) | • Mismo corpus (272 noticias) | Este mismo documento | Versión extendida y pedagógica con desgloses anatómicos y ablación multi-frase (Top-1 a Top-3). |
| **`Reporte_Sensacionalismo_Frases_vs_Documento_Original_Backup.md`** | Reporte `.md` | Versión previa de respaldo | • Mismo corpus | Referencia histórica | Copia de seguridad del informe previo a las ampliaciones explicativas. |
| **`Reporte_XAI_Sensacionalismo.md`** | Reporte `.md` | [`Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py`](../Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py) | • `muestra_evaluacion_xai_40.csv` (20 IA, 20 Prensa) | Este mismo documento | Protocolo riguroso de Interpretabilidad (Captum IG, LIME, SHAP, MoRF/LoRF, Adebayo Sanity Check). |
| **`datos_sensacionalismo_frases.csv`** | Datos `.csv` | [`Pruebas_Frases/experimento_sensacionalismo_frases.py`](../Pruebas_Frases/experimento_sensacionalismo_frases.py) | 272 noticias procesadas | Frases vs Documento | Tabla completa con probabilidades completas ($P_{\text{full}}$), reglas de agregación, ablaciones y caídas. |
| **`metricas_sensacionalismo_frases.json`** | Métricas `.json` | [`Pruebas_Frases/experimento_sensacionalismo_frases.py`](../Pruebas_Frases/experimento_sensacionalismo_frases.py) | 272 noticias procesadas | Frases vs Documento | Resumen cuantitativo formal con $R^2$, MAE, RMSE, correlaciones ($r, \rho$) y tasas de desclasificación. |
| **`metricas_xai_sensacionalismo.json`** | Métricas `.json` | [`Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py`](../Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py) | Muestra balanceada de 40 noticias | XAI Sensacionalismo | Fidelidad cuantitativa (Comprehensiveness, Sufficiency), correlación inter-métodos y tiempos de latencia. |
| **`imagenes/`** | Directorio | Organizado en 3 subcarpetas temáticas | Diversos | Mapeado por subcarpeta | Archivos gráficos organizados sin elementos sueltos. |

---

## 🖼️ 2. Organización Interna de la Carpeta `imagenes/`

Dentro de [`imagenes/`](imagenes/) los gráficos se dividen en 3 subdirectorios claramente delimitados:

### A. `imagenes/comparativa_3_modelos/` (11 Imágenes)
* **Origen:** [`3sensacionalismo.ipynb`](../3sensacionalismo.ipynb)
* **Citadas en:** [`Reporte_Sensacionalismo.md`](Reporte_Sensacionalismo.md)
* **Contenido:**
  1. `comparativa_metricas_sensacionalismo.png`: Gráfico de barras comparativo de Exactitud, Precisión, Recall y F1-Score.
  2. `comparativa_tiempos_sensacionalismo.png`: Comparativa de tiempos totales y latencia media de inferencia por muestra.
  3. `cm_m1_salud.png`: Matriz de confusión de M1 (BETO) en el dataset de Salud.
  4. `cm_m1_amarillismo.png`: Matriz de confusión de M1 (BETO) en el dataset de Amarillismo.
  5. `cm_m1_ia.png`: Matriz de confusión de M1 (BETO) en el dataset de IA Sintético.
  6. `cm_m2_salud.png`: Matriz de confusión de M2 (mDeBERTa) en Salud.
  7. `cm_m2_amarillismo.png`: Matriz de confusión de M2 (mDeBERTa) en Amarillismo.
  8. `cm_m2_ia.png`: Matriz de confusión de M2 (mDeBERTa) en IA Sintético.
  9. `cm_m3_salud.png`: Matriz de confusión de M3 (RoBERTuito) en Salud.
  10. `cm_m3_amarillismo.png`: Matriz de confusión de M3 (RoBERTuito) en Amarillismo.
  11. `cm_m3_ia.png`: Matriz de confusión de M3 (RoBERTuito) en IA Sintético.

### B. `imagenes/sensacionalismo_frases/` (7 Figuras Científicas)
* **Origen:** [`Pruebas_Frases/generar_figuras_detalladas.py`](../Pruebas_Frases/generar_figuras_detalladas.py) y [`experimento_sensacionalismo_frases.py`](../Pruebas_Frases/experimento_sensacionalismo_frases.py)
* **Citadas en:** [`Reporte_Sensacionalismo_Frases_vs_Documento.md`](Reporte_Sensacionalismo_Frases_vs_Documento.md) y [`Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md`](Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md)
* **Contenido:**
  1. `fig1_dispersion_regresion_pfull_vs_agregaciones.png`: Dispersión y regresión entre noticia completa ($P_{\text{full}}$) y las 4 reglas de agregación (Media, Máximo, Mediana, P90).
  2. `fig2_comparativa_metricas_ajuste_mae_rmse_r2.png`: Benchmark de ajuste cuantitativo (MAE, RMSE, $R^2$) y concordancia de clasificación.
  3. `fig3_distribucion_residuos_sesgo_gatillo.png`: Distribución empírica de residuos ($P_{\text{full}} - P_{\text{mean}}$) contrastando Prensa Real vs IA Sintético.
  4. `fig4_ablacion_causal_frases_drop_flips.png`: Protocolo de ablación causal al retirar las oraciones con mayor carga sensacionalista (Top-1, Top-2 y Top-3).
  5. `fig5_perfil_posicional_sensacionalismo.png`: Curva de probabilidad posicional del sensacionalismo (inicio, desarrollo y desenlace de la noticia).
  6. `fig6_contraste_ia_sintetico_vs_prensa_real.png`: Dispersión comparativa directa de la homogeneidad sintética frente a la disociación real.
  7. `fig7_casos_estudio_ejemplares.png`: Estudio anatómico frase a frase de los cuatro casos de estudio paradigmáticos.

### C. `imagenes/XAI_pruebas_sensacionalismo/` (11 Gráficos XAI)
* **Origen:** [`Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py`](../Pruebas_XAI/ejecutar_pruebas_xai_sensacionalismo.py)
* **Citadas en:** [`Reporte_XAI_Sensacionalismo.md`](Reporte_XAI_Sensacionalismo.md)
* **Contenido:**
  1. `xai_faithfulness_comprehensiveness_sufficiency.png`: Métricas cuantitativas de fidelidad por método explicativo.
  2. `xai_curvas_morf_lorf_comparativa.png`: Curvas de degradación Most Relevant First (MoRF) y Least Relevant First (LoRF).
  3. `xai_parameter_randomization_adebayo.png`: Sanity Check de Adebayo (aleatorización de pesos por capas para descartar falsas explicaciones).
  4. `xai_latencia_por_metodo.png`: Benchmark de latencia de cómputo por método explicativo (Integrated Gradients, InputXGrad, LIME, SHAP).
  5. `xai_cambio_clase_perturbaciones_ia.png`: Sensibilidad a perturbaciones en noticias generadas por IA.
  6. `xai_cambio_clase_perturbaciones_amarillismo.png`: Sensibilidad a perturbaciones en prensa digital real.
  7. `xai_cambio_clase_perturbaciones_comparativa_global.png`: Comparativa global de estabilidad y robustez inter-corpus.
  8. `xai_heatmap_ia_sensacionalista.png`: Mapa de calor de atribución para noticia IA sensacionalista.
  9. `xai_heatmap_ia_no_sensacionalista.png`: Mapa de calor de atribución para noticia IA neutral.
  10. `xai_heatmap_amarillismo_real_sensacionalista.png`: Mapa de calor de atribución para noticia real amarillista.
  11. `xai_heatmap_amarillismo_real_no_sensacionalista.png`: Mapa de calor de atribución para noticia real sobria.
