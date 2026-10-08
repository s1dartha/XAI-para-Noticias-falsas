"""
=============================================================================
GENERADOR DE VISUALIZACIONES FORENSES Y RADIOGRAFÍAS ESTILOMÉTRICAS
=============================================================================
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/generar_visualizaciones.py

Genera figuras científicas en alta resolución (PNG, 300 DPI):
1. fig_radar_iml_4_noticias.png: Diagrama de radar con las 5 Dimensiones del IML.
2. fig_trayectoria_sensacionalismo.png: Dinámica oracional del sensacionalismo (BETO).
3. fig_mapas_calor_redundancia.png: Matrices de similitud coseno intra-documental (SBERT).
4. fig_comparativa_variables_clave.png: Perfil morfosintáctico y forense comparado.
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8

def generar_graficos_auditoria(auditorias: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"📊 Generando figuras científicas en: {out_dir}")

    colores = {
        0: "#d9534f",  # Rojo oscuro (Falsa 1)
        1: "#e67e22",  # Naranja intenso (Falsa 2)
        2: "#2ecc71",  # Verde esmeralda (Verdadera 1)
        3: "#2980b9"   # Azul profesional (Verdadera 2)
    }

    # -------------------------------------------------------------------------
    # 1. RADAR CHART DE LAS 5 DIMENSIONES DEL IML
    # -------------------------------------------------------------------------
    labels_dims = [
        'D1: Carga Emocional\n(Sensacionalismo)',
        'D2: Volatilidad\nde Gatillo',
        'D3: Amortiguación\nContextual',
        'D4: Reiteración\n(Redundancia)',
        'D5: Cohesión y\nFluidez'
    ]
    num_vars = len(labels_dims)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels_dims, size=11, weight='bold', color='#2c3e50')
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#7f8c8d", size=9)
    plt.ylim(0, 105)

    for idx, aud in enumerate(auditorias):
        dims = aud["las_5_dimensiones_iml"]
        values = [
            dims["D1_Carga_Emocional"]["valor"],
            dims["D2_Volatilidad_Gatillo"]["valor"],
            dims["D3_Amortiguacion_Contextual"]["valor"],
            dims["D4_Reiteracion_Redundancia"]["valor"],
            dims["D5_Cohesion_Fluidez"]["valor"]
        ]
        values += values[:1]

        lbl = f"N{idx+1}: {aud['metadatos']['titulo'][:30]}... (IML: {dims['score_global_iml']})"
        color = colores[idx]
        linestyle = '--' if idx in [0, 1] else '-'
        ax.plot(angles, values, linewidth=2.2, linestyle=linestyle, label=lbl, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)

    plt.title("Huella Estilométrica en las 5 Dimensiones del IML\n(Índice de Manipulación Lingüística)", size=14, weight='bold', pad=25, color='#2c3e50')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9.5, frameon=True, facecolor='#ffffff')
    plt.tight_layout()
    radar_path = os.path.join(out_dir, "fig_radar_iml_4_noticias.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ [1/4] Radar IML guardado: {radar_path}")

    # -------------------------------------------------------------------------
    # 2. DINÁMICA ORACIONAL DE SENSACIONALISMO (BETO)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.patch.set_facecolor('#ffffff')

    for idx, aud in enumerate(auditorias):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor('#fdfdfd')
        oraciones = aud["desglose_oraciones"]
        x_vals = [s["idx"] for s in oraciones]
        y_vals = [s["sensacionalismo_pct"] for s in oraciones]
        color = colores[idx]

        # Línea de trayectoria
        ax.plot(x_vals, y_vals, marker='o', markersize=7, linewidth=2.5, color=color, label="Sensacionalismo (%)")
        ax.axhline(50.0, color='#95a5a6', linestyle=':', linewidth=1.2, label='Umbral Alarma (50%)')

        # Resaltar gatillos
        for s in oraciones:
            if s["es_oracion_gatillo"]:
                ax.scatter(s["idx"], s["sensacionalismo_pct"], color='#c0392b', s=160, zorder=5, edgecolor='black', linewidth=1.5)
                ax.annotate(f"Gatillo #{s['idx']}\n({s['sensacionalismo_pct']}%)",
                            (s["idx"], s["sensacionalismo_pct"]),
                            textcoords="offset points", xytext=(0, 10), ha='center',
                            fontsize=8.5, weight='bold', color='#c0392b')

        ax.set_title(f"N{idx+1}: {aud['metadatos']['titulo'][:42]}...\n$P_{{full}}={aud['sensacionalismo_beto']['P_full']}$ | $\\Delta P_{{gatillo}}={aud['sensacionalismo_beto']['delta_p_gatillo_causal']}$",
                     fontsize=11, weight='bold', color='#2c3e50')
        ax.set_xlabel("Índice de Oración en el Texto", fontsize=10)
        ax.set_ylabel("Sensacionalismo BETO (%)", fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8.5)

    plt.suptitle("Trayectoria Oracional de Carga Sensacionalista y Detección de Gatillos", fontsize=15, weight='bold', y=0.99, color='#2c3e50')
    plt.tight_layout()
    tray_path = os.path.join(out_dir, "fig_trayectoria_sensacionalismo.png")
    plt.savefig(tray_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ [2/4] Trayectoria de sensacionalismo guardada: {tray_path}")

    # -------------------------------------------------------------------------
    # 3. MATRICES DE SIMILITUD COSENO DE REDUNDANCIA (SBERT)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.patch.set_facecolor('#ffffff')

    for idx, aud in enumerate(auditorias):
        ax = axes[idx // 2, idx % 2]
        k = aud["metadatos"]["num_oraciones"]
        # Extraer matriz de similitud guardada
        if "matriz_similitud" in aud["redundancia_sbert"]:
            mat_sim = np.array(aud["redundancia_sbert"]["matriz_similitud"])
        else:
            mat_sim = np.eye(k)

        sns.heatmap(mat_sim, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=1.0, annot=k<=9, fmt=".2f",
                    cbar_kws={'label': 'Similitud Coseno SBERT'}, annot_kws={"size": 8.5})
        ax.set_title(f"N{idx+1}: Redundancia Intra-Doc ($S_{{max}}={aud['redundancia_sbert']['max_intra_similarity']}$)\n{aud['redundancia_sbert']['diagnostico']}",
                     fontsize=10.5, weight='bold', color='#2c3e50')
        ax.set_xlabel("Índice de Oración", fontsize=9.5)
        ax.set_ylabel("Índice de Oración", fontsize=9.5)

    plt.suptitle("Matrices de Similitud Coseno Intra-Documental (Sentence-BERT)\nIdentificación de Bucles y Pares Redundantes ($\\tau \\geq 0.34$)",
                 fontsize=14, weight='bold', y=0.99, color='#2c3e50')
    plt.tight_layout()
    red_path = os.path.join(out_dir, "fig_mapas_calor_redundancia.png")
    plt.savefig(red_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ [3/4] Mapas de redundancia guardados: {red_path}")

    # -------------------------------------------------------------------------
    # 4. COMPARATIVA DE VARIABLES CLAVE ENRIQUECIDAS (D1-D5)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#ffffff')

    noticia_labels = [f"N{i+1}: {'Falsa' if i in [0, 1] else 'Real'}" for i in range(len(auditorias))]
    palette = [colores[i] for i in range(len(auditorias))]

    # A. Mayúsculas Sostenidas (%)
    ax = axes[0, 0]
    vals = [aud["variables_ensamble_avanzado_5d"]["D5_Anclajes_Factuales_Puntuacion"]["all_caps_ratio_pct"] for aud in auditorias]
    bars = ax.bar(noticia_labels, vals, color=palette, edgecolor='black', linewidth=1)
    ax.set_title("Mayúsculas Sostenidas en Palabras (%) [D5]\nTop Predictor Invariante de Urgencia Viral", fontsize=11, weight='bold', color='#2c3e50')
    ax.set_ylabel("% Palabras en MAYÚSCULAS")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9.5, weight='bold')

    # B. Densidad de Adverbios (%)
    ax = axes[0, 1]
    vals = [aud["variables_ensamble_avanzado_5d"]["D4_Perfilado_Morfosintactico_POS"]["adv_density"] for aud in auditorias]
    bars = ax.bar(noticia_labels, vals, color=palette, edgecolor='black', linewidth=1)
    ax.set_title("Densidad de Adverbios (%) [D4]\nTop 1 Predictor de Subjetividad y Carga Valorativa", fontsize=11, weight='bold', color='#2c3e50')
    ax.set_ylabel("% Adverbios sobre total palabras")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9.5, weight='bold')

    # C. Densidad de Verbos Dicendi (%)
    ax = axes[1, 0]
    vals = [aud["variables_ensamble_avanzado_5d"]["D2_Linguistica_Forense_Epistemica"]["dicendi_density"] for aud in auditorias]
    bars = ax.bar(noticia_labels, vals, color=palette, edgecolor='black', linewidth=1)
    ax.set_title("Densidad de Verbos Dicendi (%) [D2]\nAtribución Deontológica a Fuentes Oficiales", fontsize=11, weight='bold', color='#2c3e50')
    ax.set_ylabel("% Verbos de Atribución")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9.5, weight='bold')

    # D. Anclajes Numéricos y Cuantitativos (%)
    ax = axes[1, 1]
    vals = [aud["variables_ensamble_avanzado_5d"]["D5_Anclajes_Factuales_Puntuacion"]["numbers_density"] for aud in auditorias]
    bars = ax.bar(noticia_labels, vals, color=palette, edgecolor='black', linewidth=1)
    ax.set_title("Densidad de Anclajes Numéricos (%) [D5]\nRigor Cuantitativo Verificable y Estadístico", fontsize=11, weight='bold', color='#2c3e50')
    ax.set_ylabel("% Cifras sobre total palabras")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9.5, weight='bold')

    plt.suptitle("Radiografía Estilométrica Comparada en Características Forenses Clave", fontsize=15, weight='bold', y=0.99, color='#2c3e50')
    plt.tight_layout()
    vars_path = os.path.join(out_dir, "fig_comparativa_variables_clave.png")
    plt.savefig(vars_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ [4/4] Comparativa de variables guardada: {vars_path}")

    return {
        "radar": radar_path,
        "trayectoria": tray_path,
        "redundancia": red_path,
        "variables": vars_path
    }
