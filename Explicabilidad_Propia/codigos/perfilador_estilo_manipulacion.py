"""
=============================================================================
PERFILADOR DE ESTILOS DE MANIPULACIÓN LINGÜÍSTICA (IML) Y ARQUETIPOS DISCURSIVOS
=============================================================================
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/codigos/perfilador_estilo_manipulacion.py
Fase 1 de la Tesis: Explicabilidad Forense Estilométrica Multi-Nivel.

Sintetiza las 5 Dimensiones del IML (escala 0 - 100) y clasifica cada noticia en
uno de los 5 Arquetipos Discursivos cualitativos sobre el corpus de 2.604 noticias.
Establece la base empírica forense antes de plantear la pregunta de clasificación.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

def run_manipulation_profiler():
    features_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_2604.csv"
    if not os.path.exists(features_csv):
        print(f"❌ No se encontró el archivo de features: {features_csv}")
        return

    print(f"⏳ Cargando datos para perfilado de manipulación de estilo: {features_csv}")
    df = pd.read_csv(features_csv)

    # 1. Función para calcular las 5 Dimensiones del Índice de Manipulación Lingüística (IML)
    def compute_manipulation_dimensions(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        dr = row['dilution_ratio']
        delta_p = row['delta_p_gatillo']
        max_sim = row['max_intra_similarity_clean']
        red_density = row['redundancy_density']
        shannon_bin = row['shannon_bin_7']

        # Dimensión 1: Carga Emocional / Alarmismo (0 - 100)
        d1_sensacionalismo = float(np.clip(p_full * 100.0, 0.0, 100.0))

        # Dimensión 2: Volatilidad de Gatillo / Desbalance Cabecera-Cuerpo (0 - 100)
        volatilidad = max(0.0, p_max - p_mean)
        d2_volatilidad_gatillo = float(np.clip(volatilidad * 100.0 * 1.5, 0.0, 100.0))

        # Dimensión 3: Amortiguación Contextual / Dilución (0 - 100)
        # Mide qué tan sobrio es el cuerpo para mitigar el titular
        if p_max >= 0.50:
            amortiguacion = max(0.0, 1.0 - dr)
        else:
            amortiguacion = 1.0 # Si no hay alarma, el texto es completamente estable
        d3_amortiguacion = float(np.clip(amortiguacion * 100.0, 0.0, 100.0))

        # Dimensión 4: Reiteración / Hiper-Redundancia Semántica (0 - 100)
        # Basado en la escala continua y penalización por densidad de bucle
        d4_redundancia = float(np.clip((max_sim - 0.34) / (0.90 - 0.34) * 100.0, 0.0, 100.0))

        # Dimensión 5: Cohesión y Fluidez Discursiva (0 - 100)
        # Máxima en la banda óptima de Shannon (0.618 - 0.808) y decae en extremos
        if max_sim <= 0.5924:
            cohesion = (max_sim / 0.5924) * 60.0 # Desarticulado
        elif max_sim <= 0.8077:
            cohesion = 90.0 + (1.0 - abs(max_sim - 0.625) / 0.18) * 10.0 # Óptimo profesional
        else:
            cohesion = max(30.0, 100.0 - (max_sim - 0.8077) / 0.19 * 70.0) # Bucle
        d5_cohesion = float(np.clip(cohesion, 0.0, 100.0))

        # 2. Clasificación en 5 Arquetipos de Estilo Discursivo
        if d1_sensacionalismo >= 65.0 and d4_redundancia >= 65.0:
            estilo = "Estilo I: Desinformación Estridente / Cliché Hiperbólico"
            arquetipo_id = "DESINFO_ESTRIDENTE"
            desc = "Texto saturado homogéneamente de emociones extremas con reiteración circular de premisas."
        elif d2_volatilidad_gatillo >= 50.0 and d3_amortiguacion >= 50.0:
            estilo = "Estilo II: Cebo Comercial / Clickbait de Cabecera"
            arquetipo_id = "CLICKBAIT_CABECERA"
            desc = "Titular o apertura alarmista diseñado para capturar clics, pero con cuerpo informativo formal que amortigua la alarma."
        elif d1_sensacionalismo < 45.0 and d4_redundancia >= 65.0:
            estilo = "Estilo III: Propaganda Institucional / Astroturfing Encubierto"
            arquetipo_id = "PROPAGANDA_SOBRIA"
            desc = "Tono aparentemente sobrio y académico, pero con bucles de repetición semántica calculados para inducir verdad ilusoria."
        elif d5_cohesion <= 55.0 and d4_redundancia < 40.0:
            estilo = "Estilo IV: Incoherencia Estructural / Generación Rota"
            arquetipo_id = "DESARTICULADO"
            desc = "Texto con baja conectividad proposicional, fragmentación sintáctica o párrafos desvinculados."
        else:
            estilo = "Estilo V: Periodismo Profesional Balanceado"
            arquetipo_id = "PERIODISMO_EQUILIBRADO"
            desc = "Redacción equilibrada con variedad léxica, desarrollo temático progresivo y ausencia de manipulaciones estilísticas."

        # Índice Global de Manipulación Estilométrica (IML de 0 a 100)
        # Pondera la severidad de las anomalías
        iml_score = (d1_sensacionalismo * 0.35 + 
                     d2_volatilidad_gatillo * 0.20 + 
                     (100.0 - d3_amortiguacion) * 0.15 + 
                     d4_redundancia * 0.20 + 
                     (100.0 - d5_cohesion) * 0.10)
        iml_score = float(np.clip(iml_score, 0.0, 100.0))

        return (round(d1_sensacionalismo, 1), 
                round(d2_volatilidad_gatillo, 1), 
                round(d3_amortiguacion, 1), 
                round(d4_redundancia, 1), 
                round(d5_cohesion, 1), 
                round(iml_score, 1), 
                estilo, 
                arquetipo_id, 
                desc)

    print("⏳ Computando perfil estilométrico y dimensiones IML para las 2,604 noticias...")
    perfiles = [compute_manipulation_dimensions(row) for _, row in df.iterrows()]

    df['dim_sensacionalismo'] = [p[0] for p in perfiles]
    df['dim_volatilidad_gatillo'] = [p[1] for p in perfiles]
    df['dim_amortiguacion'] = [p[2] for p in perfiles]
    df['dim_redundancia'] = [p[3] for p in perfiles]
    df['dim_cohesion'] = [p[4] for p in perfiles]
    df['indice_manipulacion_iml'] = [p[5] for p in perfiles]
    df['estilo_manipulacion'] = [p[6] for p in perfiles]
    df['arquetipo_id'] = [p[7] for p in perfiles]
    df['descripcion_estilo'] = [p[8] for p in perfiles]

    # Análisis de distribución por clase real (Ground Truth) vs Arquetipo
    print("\n" + "="*80)
    print(" 🎨 DISTRIBUCIÓN DE ARQUETIPOS DE ESTILO EN EL CORPUS DE 2,604 NOTICIAS")
    print("="*80)
    
    cross_tab = pd.crosstab(df['estilo_manipulacion'], df['class'], margins=True)
    print(cross_tab)

    # Análisis de tasas empíricas de fake news por estilo de manipulación
    style_summary = []
    for estilo, group in df.groupby('estilo_manipulacion'):
        n_total = len(group)
        n_fake = (group['label_num'] == 1).sum()
        pct_fake = n_fake / n_total * 100.0
        mean_iml = group['indice_manipulacion_iml'].mean()
        style_summary.append({
            "estilo": estilo,
            "total_noticias": int(n_total),
            "noticias_falsas": int(n_fake),
            "noticias_reales": int(n_total - n_fake),
            "tasa_fakenews_pct": round(float(pct_fake), 2),
            "iml_promedio": round(float(mean_iml), 2)
        })

    print("\n" + "="*80)
    print(" 📈 TASA DE NOTICIAS FALSAS Y SCORE IML POR ARQUETIPO DE MANIPULACIÓN")
    print("="*80)
    for s in style_summary:
        print(f"• {s['estilo']}")
        print(f"  Total: {s['total_noticias']:>4} | Falsas: {s['noticias_falsas']:>4} ({s['tasa_fakenews_pct']:>5.1f}%) | IML Promedio: {s['iml_promedio']:>5.1f}/100")

    # Guardar CSV consolidado con perfilado de estilos (en Explicabilidad_Propia y en Modelo_Ensamble)
    out_perfiles_propia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perfiles_estilos_manipulacion_2604.csv")
    out_perfiles_legacy = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/perfiles_estilos_manipulacion_2604.csv"
    
    cols_export = ['class', 'label_num', 'Text', 'Fuente', 'indice_manipulacion_iml', 'estilo_manipulacion', 'arquetipo_id',
                   'dim_sensacionalismo', 'dim_volatilidad_gatillo', 'dim_amortiguacion', 'dim_redundancia', 'dim_cohesion']
    df[cols_export].to_csv(out_perfiles_propia, index=False)
    df[cols_export].to_csv(out_perfiles_legacy, index=False)
    print(f"\n💾 Base de datos de perfilado guardada en:\n  • {out_perfiles_propia}\n  • {out_perfiles_legacy}")

    # Guardar JSON con resumen de estilos (en Explicabilidad_Propia/reportes y en Modelo_Ensamble/Reportes)
    out_json_propia = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reportes", "metricas_estilos_manipulacion.json")
    out_json_legacy = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_estilos_manipulacion.json"
    with open(out_json_propia, "w", encoding="utf-8") as f:
        json.dump(style_summary, f, indent=4, ensure_ascii=False)
    with open(out_json_legacy, "w", encoding="utf-8") as f:
        json.dump(style_summary, f, indent=4, ensure_ascii=False)
    print(f"💾 Resumen de arquetipos guardado en:\n  • {out_json_propia}\n  • {out_json_legacy}")

if __name__ == "__main__":
    run_manipulation_profiler()
