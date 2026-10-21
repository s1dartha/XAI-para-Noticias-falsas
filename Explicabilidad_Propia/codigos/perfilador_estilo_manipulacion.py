"""
=============================================================================
PERFILADOR DE ESTILOS DE MANIPULACIÓN LINGÜÍSTICA (IML) Y ARQUETIPOS DISCURSIVOS
=============================================================================
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/codigos/perfilador_estilo_manipulacion.py
Fase 1 de la Tesis: Explicabilidad Forense Estilométrica Multi-Nivel.

Sintetiza las 5 Dimensiones del IML (escala 0 - 100) y clasifica cada noticia en
uno de los 5 Arquetipos Discursivos cualitativos sobre el corpus de 2.604 noticias,
utilizando de forma unificada el modelo ganador BETO SBERT
(hiiamsid/sentence_similarity_spanish_es, d=768), la jerarquía de umbrales
(0.3740, 0.5340, 0.6079, 0.7287) y el segundo estadístico de orden S_(2) (MexGen).
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from arquetipos_flexibles import clasificar_arquetipo

def run_manipulation_profiler():
    features_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_2604.csv"
    if not os.path.exists(features_csv):
        print(f"❌ No se encontró el archivo de features: {features_csv}")
        return

    print(f"⏳ Cargando datos BETO SBERT para perfilado de manipulación de estilo: {features_csv}")
    df = pd.read_csv(features_csv)

    # 1. Función para calcular las 5 Dimensiones del Índice de Manipulación Lingüística (IML)
    def compute_manipulation_dimensions(row):
        p_full = row['P_full']
        p_mean = row['P_mean']
        p_max = row['P_max']
        dr = row['dilution_ratio']
        delta_p = row['delta_p_gatillo']
        max_sim = row['max_intra_similarity_clean']
        s2_sim = row.get('second_max_intra_similarity_clean', max_sim * 0.85)
        n_sents = row.get('num_sentences', 3)

        # Dimensión 1: Carga Emocional / Alarmismo (0 - 100)
        d1_sensacionalismo = float(np.clip(p_full * 100.0, 0.0, 100.0))

        # Dimensión 2: Volatilidad de Gatillo / Desbalance Cabecera-Cuerpo (0 - 100)
        volatilidad = max(0.0, p_max - p_mean)
        d2_volatilidad_gatillo = float(np.clip(volatilidad * 100.0 * 1.5, 0.0, 100.0))

        # Dimensión 3: Amortiguación Contextual / Dilución (0 - 100)
        if p_max >= 0.50:
            amortiguacion = max(0.0, 1.0 - dr)
        else:
            amortiguacion = 1.0
        d3_amortiguacion = float(np.clip(amortiguacion * 100.0, 0.0, 100.0))

        # Dimensión 4: Reiteración / Redundancia Semántica Estricta y Multi-Bucle MexGen (0 - 100)
        # Calibrado con los umbrales de BETO SBERT (0.3434 GMM, 0.5340 PAWS-X, 0.5996/0.6079 Estricto, 0.7287 Hiper)
        base_d4 = (max_sim - 0.25) / (0.78 - 0.25) * 100.0
        multi_loop_bonus = max(0.0, (s2_sim - 0.45) * 35.0) if n_sents >= 3 else 0.0
        d4_redundancia = float(np.clip(base_d4 + multi_loop_bonus, 0.0, 100.0))

        # Dimensión 5: Cohesión y Fluidez Discursiva (0 - 100)
        # Máxima en la banda óptima In-Fold de Shannon en BETO SBERT (0.3740 - 0.6500) y decae en extremos
        if max_sim <= 0.3740:
            cohesion = (max(0.0, max_sim) / 0.3740) * 52.0  # Fragmentado / Desarticulado (Bin 1)
        elif max_sim <= 0.6500:
            cohesion = 85.0 + (1.0 - abs(max_sim - 0.5200) / 0.1500) * 15.0  # Óptimo profesional (Bin 2)
        else:
            cohesion = max(35.0, 95.0 - (max_sim - 0.6500) / 0.2500 * 60.0)  # Bucle circular (Bin 4)
        d5_cohesion = float(np.clip(cohesion, 0.0, 100.0))

        # 2. Clasificación Flexible en los 5 Arquetipos Estilométricos por Afinidad
        arq = clasificar_arquetipo(
            d1_sensacionalismo,
            d2_volatilidad_gatillo,
            d3_amortiguacion,
            d4_redundancia,
            d5_cohesion
        )
        estilo = arq["nombre"]
        id_map = {
            "ESTILO_I_DESINFO_ESTRIDENTE": "DESINFO_ESTRIDENTE",
            "ESTILO_II_CLICKBAIT_CABECERA": "CLICKBAIT_CABECERA",
            "ESTILO_III_PROPAGANDA_SOBRIA": "PROPAGANDA_SOBRIA",
            "ESTILO_IV_DESARTICULADO": "DESARTICULADO",
            "ESTILO_V_PERIODISMO_EQUILIBRADO": "PERIODISMO_EQUILIBRADO"
        }
        arquetipo_id = id_map.get(arq["id"], arq["id"])
        desc = arq["descripcion"]
        afinidad = arq["afinidad_pct"]
        arquetipo_sec = arq["arquetipo_secundario"]
        afinidad_sec = arq["afinidad_secundaria_pct"]

        # Índice Global de Manipulación Estilométrica (IML de 0 a 100)
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
                desc,
                afinidad,
                arquetipo_sec,
                afinidad_sec)

    print("⏳ Computando perfil estilométrico y dimensiones IML (BETO SBERT) para las 2,604 noticias...")
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
    df['afinidad_pct'] = [p[9] for p in perfiles]
    df['arquetipo_secundario'] = [p[10] for p in perfiles]
    df['afinidad_secundaria_pct'] = [p[11] for p in perfiles]
    df['descripcion_estilo'] = [p[8] for p in perfiles]

    print("\n" + "="*80)
    print(" 🎨 DISTRIBUCIÓN DE ARQUETIPOS DE ESTILO EN EL CORPUS DE 2,604 NOTICIAS (BETO SBERT)")
    print("="*80)

    cross_tab = pd.crosstab(df['estilo_manipulacion'], df['class'], margins=True)
    print(cross_tab)

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

    out_perfiles_propia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perfiles_estilos_manipulacion_2604.csv")
    out_perfiles_legacy = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/perfiles_estilos_manipulacion_2604.csv"

    cols_export = ['class', 'label_num', 'Text', 'Fuente', 'indice_manipulacion_iml', 'estilo_manipulacion', 'arquetipo_id',
                   'dim_sensacionalismo', 'dim_volatilidad_gatillo', 'dim_amortiguacion', 'dim_redundancia', 'dim_cohesion',
                   'afinidad_pct', 'arquetipo_secundario', 'afinidad_secundaria_pct']
    df[cols_export].to_csv(out_perfiles_propia, index=False)
    df[cols_export].to_csv(out_perfiles_legacy, index=False)
    print(f"\n💾 Base de datos de perfilado guardada en:\n  • {out_perfiles_propia}\n  • {out_perfiles_legacy}")

    out_json_propia = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reportes", "metricas_estilos_manipulacion.json")
    out_json_legacy = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Reportes/metricas_estilos_manipulacion.json"
    with open(out_json_propia, "w", encoding="utf-8") as f:
        json.dump(style_summary, f, indent=4, ensure_ascii=False)
    with open(out_json_legacy, "w", encoding="utf-8") as f:
        json.dump(style_summary, f, indent=4, ensure_ascii=False)
    print(f"💾 Resumen de arquetipos guardado en:\n  • {out_json_propia}\n  • {out_json_legacy}")

if __name__ == "__main__":
    run_manipulation_profiler()
