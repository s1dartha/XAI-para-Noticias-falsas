# XAI para Detección y Análisis de Noticias Falsas

Proyecto de investigación y desarrollo para la clasificación, análisis semántico e **Inteligencia Artificial Explicable (XAI)** en la detección de noticias falsas (*Fake News*) y sensacionalismo en idioma español. Integra modelos Transformer (BETO, SaBERT, mDeBERTa, RoBERTuito), técnicas XAI (Captum Integrated Gradients, SHAP, LIME, MoRF/LoRF, Sanity Check de Adebayo) y una aplicación web interactiva (MexGen).

---

## 📁 Estructura del Repositorio

```text
XAI-para-Noticias-falsas/
├── Modelos_Individuales/
│   ├── Clasificar_fake/
│   │   ├── Dataset/                                   # Datasets de noticias falsas (.xlsx, .csv)
│   │   ├── reportes/                                  # Informes, métricas JSON y gráficos
│   │   │   ├── Reporte_FakeNews.md
│   │   │   ├── Reporte_4_Modelos_FakeNews.md
│   │   │   ├── Reporte_XAI_FakeNews.md
│   │   │   ├── Reporte_XAI_FakeNews_2.md
│   │   │   ├── metricas_*.json
│   │   │   ├── clasificar_model_4_imagenes/           # Curvas ROC, PR y matrices
│   │   │   └── imagenes/                              # Matrices de confusión y heatmaps XAI
│   │   ├── run_comparativo_fakenews.py                # Pipeline comparativo 3 modelos
│   │   └── run_4_modelos_fakenews.py                  # Pipeline comparativo 4 modelos
│   │
│   ├── Sensacionalismo/
│   │   ├── Dataset/                                   # Datasets (Salud, Amarillismo, IA Sintético)
│   │   ├── reportes/                                  # Reportes, métricas y catálogo de imágenes
│   │   │   ├── Reporte_Sensacionalismo.md
│   │   │   ├── Reporte_Sensacionalismo_Frases_vs_Documento.md
│   │   │   ├── Reporte_Sensacionalismo_Frases_vs_Documento_Detallado.md
│   │   │   ├── Reporte_XAI_Sensacionalismo.md
│   │   │   ├── metricas_sensacionalismo_frases.json
│   │   │   ├── metricas_xai_sensacionalismo.json
│   │   │   ├── datos_sensacionalismo_frases.csv
│   │   │   ├── README.md                              # Mapa de trazabilidad de reportes
│   │   │   └── imagenes/                              # 3 subcarpetas temáticas organizadas
│   │   │       ├── comparativa_3_modelos/
│   │   │       ├── sensacionalismo_frases/
│   │   │       └── XAI_pruebas_sensacionalismo/
│   │   ├── 3sensacionalismo.ipynb                     # Cuadernillo comparativo de modelos
│   │   ├── Pruebas_Frases/                            # Experimento a nivel de oraciones
│   │   │   ├── experimento_sensacionalismo_frases.py
│   │   │   └── generar_figuras_detalladas.py
│   │   └── Pruebas_XAI/                               # Pipeline de explicabilidad
│   │       ├── Pruebas_XAI_Sensacionalismo.ipynb
│   │       └── ejecutar_pruebas_xai_sensacionalismo.py
│   │
│   ├── Redundancia/                                   # Módulo de similitud y redundancia semántica
│   └── Tareas/                                        # Enunciados y especificaciones técnicas
│
├── Resultados/                                        # Aplicación Web Interactiva (MexGen)
│   ├── index.html                                     # Interfaz web de usuario
│   ├── style.css                                      # Estilos visuales
│   ├── app.js                                         # Lógica frontend interactiva
│   ├── server.py                                      # Servidor backend de inferencia (Flask)
│   └── pipeline_tesis.ipynb                           # Cuadernillo de integración
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Instalación y Requisitos

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/s1dartha/XAI-para-Noticias-falsas.git
   cd XAI-para-Noticias-falsas
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   python3 -m spacy download es_core_news_sm
   ```

---

## 🌐 Aplicación Web MexGen

Para iniciar la aplicación web de inferencia en tiempo real:
```bash
cd Resultados
python3 server.py
```
Abre en tu navegador `http://127.0.0.1:5000` para analizar noticias interactivamente (detección de Fake News con SaBERT, Sensacionalismo a nivel de párrafos y análisis de redundancia semántica con SBERT).

---

## 👥 Colaboradores

- **Autor:** Natali Alba ([@s1dartha](https://github.com/s1dartha))
- **Colaborador:** [@danisanty06](https://github.com/danisanty06)
