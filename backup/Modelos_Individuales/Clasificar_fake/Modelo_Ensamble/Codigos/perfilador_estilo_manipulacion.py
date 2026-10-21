"""
=============================================================================
PERFILADOR DE ESTILOS DE MANIPULACIÓN LINGÜÍSTICA (IML) [REDIRECCIÓN]
=============================================================================
NOTA: Como parte de la reorganización modular de la tesis, este script ha sido
trasladado al módulo oficial de explicabilidad:
👉 /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/codigos/perfilador_estilo_manipulacion.py

Este archivo actúa como proxy de compatibilidad para asegurar que cualquier
llamada legacy continúe funcionando transparentemente.
=============================================================================
"""

import os
import sys

# Ruta oficial en Explicabilidad_Propia
ruta_oficial_dir = "/home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/codigos"
if ruta_oficial_dir not in sys.path:
    sys.path.insert(0, ruta_oficial_dir)

from perfilador_estilo_manipulacion import run_manipulation_profiler

if __name__ == "__main__":
    print("ℹ️ [Aviso Metodológico] Ejecutando Perfilador IML desde su ubicación oficial en Explicabilidad_Propia/codigos/...")
    run_manipulation_profiler()
