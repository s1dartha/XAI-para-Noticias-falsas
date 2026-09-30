#!/usr/bin/env python3
"""
Pipeline Maestro: Ensamble Morfológico Avanzado 5D para Clasificación de Fake News
Ejecuta la extracción de las 5 dimensiones textuales y el entrenamiento con 5-Fold CV.
"""
import os
import sys
import subprocess

def run_step(step_name, script_path):
    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO: {step_name}")
    print(f"📄 Script: {script_path}")
    print(f"{'='*80}\n")
    
    python_bin = sys.executable
    cmd = [python_bin, script_path]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"❌ Error en {step_name} (código de salida: {ret.returncode})")
        sys.exit(ret.returncode)
    print(f"\n✅ {step_name} completado con éxito.\n")

def main():
    codigos_dir = os.path.dirname(os.path.abspath(__file__))
    
    step1_script = os.path.join(codigos_dir, "extraer_features_avanzadas_5d.py")
    step2_script = os.path.join(codigos_dir, "entrenar_evaluar_ensamble_avanzado_5d.py")
    
    print("="*80)
    print("PIPELINE MAESTRO: ENSAMBLE AVANZADO 5D (N = 4.418 NOTICIAS)")
    print("="*80)
    
    # 1. Extracción de Features (si no existen o si se fuerza)
    csv_adv = os.path.join(codigos_dir, "features_ensamble_ampliado_avanzado_4418.csv")
    if not os.path.exists(csv_adv):
        run_step("Paso 1: Extracción de Características de las 5 Dimensiones Textuales", step1_script)
    else:
        print(f"ℹ️ Archivo de features avanzadas ya existente: {csv_adv}")
        print("   (Para recalcular desde cero, elimine dicho archivo y vuelva a ejecutar).")

    # 2. Entrenamiento, Validación y Generación de Gráficos
    run_step("Paso 2: Entrenamiento 5-Fold CV, Calibración de Umbral y Generación de Figuras", step2_script)

    print("="*80)
    print("🎉 ¡PIPELINE COMPLETO CULMINADO CON ÉXITO!")
    print("="*80)

if __name__ == '__main__':
    main()
