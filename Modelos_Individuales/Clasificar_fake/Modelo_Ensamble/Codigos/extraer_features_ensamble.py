import os
import sys
import ast
import time
import json
import torch
import numpy as np
import pandas as pd

# Jerarquía de Umbrales Semánticos Calibrados con BETO SBERT (hiiamsid/sentence_similarity_spanish_es)
TAU_CONTINUIDAD_GMM = 0.3434      # Continuidad temática basal (GMM)
TAU_PARAFRASIS_PAWSX = 0.5340     # Paráfrasis moderada (P90 / PAWS-X = 0.533)
TAU_REDUNDANCIA_ESTRICTA = 0.5996 # Redundancia semántica estricta (P95 / Shannon = 0.6079)
TAU_HIPER_REDUNDANCIA = 0.7287    # Hiper-redundancia circular (P99)

def main():
    sim_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Redundancia/Dataset/dataset_con_similitudes.csv"
    mex_path = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Redundancia/Reportes/Umbral_Redundancia/validacion_explicabilidad_mexgen.csv"
    out_csv = "/home/ubuntu/Documentos/Tesis/Modelos_Individuales/Clasificar_fake/Modelo_Ensamble/Codigos/features_ensamble_2604.csv"

    print(f"⏳ Leyendo dataset de similitudes BETO SBERT: {sim_path}")
    df_sim = pd.read_csv(sim_path)
    print(f"✅ Cargado: {len(df_sim)} registros desde dataset_con_similitudes.csv (hiiamsid/sentence_similarity_spanish_es)")

    # Conservar columnas de Sensacionalismo BETO y metadatos si ya existen en features_ensamble_2604.csv
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        sens_cols = ['Fuente', 'conteo_palabras_text', 'P_full', 'P_mean', 'P_max', 'P_top2',
                     'sigma_sens', 'dilution_ratio', 'delta_p_gatillo', 'categoria']
        for c in sens_cols:
            if c in df_old.columns and len(df_old) == len(df_sim):
                df_sim[c] = df_old[c].values

    df = df_sim.copy()
    # Etiqueta canónica en Clasificar_fake: class=True -> 0 (Real, N=1345), class=False -> 1 (Fake, N=1259)
    df['label_num'] = df['class'].map({True: 0, False: 1, 'True': 0, 'False': 1, 'REAL': 0, 'FAKE': 1, 0: 1, 1: 0}).fillna(0).astype(int)
    df['label_name'] = df['label_num'].map({0: 'Real', 1: 'Fake'})
    if 'conteo_palabras_text' not in df.columns:
        df['conteo_palabras_text'] = df['Text'].astype(str).apply(lambda t: len(t.split()))

    # Calcular hiper-redundancia (>= 0.7287) desde pairwise_similarities
    hyper_counts = []
    for s_str in df['pairwise_similarities']:
        try:
            sims = ast.literal_eval(str(s_str)) if pd.notna(s_str) else []
        except Exception:
            sims = []
        hyper_counts.append(int(sum(1 for x in sims if x >= TAU_HIPER_REDUNDANCIA)))
    df['hyper_redundant_pairs_count'] = hyper_counts

    # Integrar métricas MexGen por doc_id si existe validacion_explicabilidad_mexgen.csv
    if os.path.exists(mex_path):
        df_mex = pd.read_csv(mex_path)
        mex_map_post = dict(zip(df_mex['doc_id'], df_mex['max_after_star']))
        mex_map_gap = dict(zip(df_mex['doc_id'], df_mex['delta_max_star']))
        mex_map_morf = dict(zip(df_mex['doc_id'], df_mex['delta_mean_hub']))
        df['max_post_ablation_mexgen'] = [mex_map_post.get(i, np.nan) for i in range(len(df))]
        df['mexgen_persistence_gap'] = [mex_map_gap.get(i, 0.0) for i in range(len(df))]
        df['delta_mean_sim_morf'] = [mex_map_morf.get(i, 0.0) for i in range(len(df))]
    else:
        df['max_post_ablation_mexgen'] = df['second_max_intra_similarity']
        df['mexgen_persistence_gap'] = (df['max_intra_similarity'] - df['second_max_intra_similarity']).fillna(0.0)
        df['delta_mean_sim_morf'] = 0.0

    # Imputar nulos de manera neutra (133 docs con 1 sola oración)
    for col in ['max_intra_similarity', 'mean_intra_similarity', 'second_max_intra_similarity',
                'top3_mean_intra_similarity', 'p90_intra_similarity', 'var_intra_similarity', 'max_post_ablation_mexgen']:
        med_val = df[col].dropna().median()
        df[f'{col}_clean'] = df[col].fillna(med_val)

    # Densidades por nivel de la jerarquía de umbrales
    n_pairs_safe = df['num_pairs'].replace(0, 1)
    df['topic_continuity_density'] = df['redundant_pairs_count'] / n_pairs_safe          # >= 0.3434 (GMM)
    df['paraphrase_density_p90'] = df['paraphrase_pairs_count_p90'] / n_pairs_safe       # >= 0.5340 (PAWS-X)
    df['strict_redundancy_density_p95'] = df['strict_redundant_pairs_count_p95'] / n_pairs_safe # >= 0.5996 (P95)
    df['hyper_redundancy_density'] = df['hyper_redundant_pairs_count'] / n_pairs_safe    # >= 0.7287 (Hiper)

    # Redundancy density principal alineada con Paráfrasis PAWS-X / Redundancia Estricta
    df['redundancy_density'] = df['strict_redundancy_density_p95']

    # Discretizaciones canónicas sobre BETO SBERT (para análisis descriptivo; en CV se calculan In-Fold)
    cuts_4 = [-np.inf, 0.3740, 0.6079, 0.7287, np.inf]
    df['shannon_bin_4'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_4, labels=[1, 2, 3, 4]).astype(int)

    cuts_7 = [-np.inf, 0.3740, 0.4500, 0.5340, 0.6079, 0.6650, 0.7287, np.inf]
    df['shannon_bin_7'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)

    df.to_csv(out_csv, index=False)
    print(f"🎉 ¡Características BETO SBERT consolidadas y guardadas en: {out_csv} ({len(df)} filas, {len(df.columns)} cols)!")

if __name__ == "__main__":
    main()
