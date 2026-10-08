"""
=============================================================================
AUDITOR FORENSE DE NOTICIAS PERSONALIZADAS (INTERFAZ CLI / SCRIPT)
=============================================================================
Ubicación: /home/ubuntu/Documentos/Tesis/Explicabilidad_Propia/auditar_texto_personalizado.py

Permite ingresar cualquier texto de noticia y obtener:
- Las 5 Dimensiones del Índice de Manipulación Lingüística (IML)
- Sensacionalismo Global y Frase a Frase (BETO)
- Redundancia Global y Frase a Frase (SBERT)
- Variables Morfosintácticas y Forenses del Ensamble 5D
- Desglose Oracional con Gatillos y Advertencias
(Sin clasificación binaria de caja negra)

Ejemplos de Uso:
  python auditar_texto_personalizado.py --texto "Texto de la noticia aquí..."
  python auditar_texto_personalizado.py --archivo noticia.txt --salida resultado.json
=============================================================================
"""

import os
import sys
import json
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from metodo_explicabilidad import AuditorExplicabilidadNoticia

def main():
    parser = argparse.ArgumentParser(description="Auditoría Forense Explicable de Noticia")
    parser.add_argument("--texto", type=str, help="Texto plano de la noticia a auditar")
    parser.add_argument("--archivo", type=str, help="Ruta al archivo .txt con el cuerpo de la noticia")
    parser.add_argument("--titulo", type=str, default="Noticia Personalizada", help="Título o encabezado de la noticia")
    parser.add_argument("--salida_json", type=str, default=None, help="Ruta para guardar el resultado en formato JSON")
    parser.add_argument("--salida_md", type=str, default=None, help="Ruta para guardar el informe en formato Markdown")
    args = parser.parse_args()

    texto = ""
    if args.archivo:
        if not os.path.exists(args.archivo):
            print(f"❌ Error: El archivo especificado no existe: {args.archivo}")
            sys.exit(1)
        with open(args.archivo, "r", encoding="utf-8") as f:
            texto = f.read().strip()
    elif args.texto:
        texto = args.texto.strip()
    else:
        print("✍️ Ingrese o pegue el texto de la noticia a auditar (finalice con CTRL+D o línea vacía doble):")
        lineas = []
        try:
            while True:
                linea = input()
                if not linea and lineas and not lineas[-1]:
                    break
                lineas.append(linea)
        except EOFError:
            pass
        texto = "\n".join(lineas).strip()

    if not texto:
        print("❌ Error: No se proporcionó ningún texto para auditar.")
        sys.exit(1)

    print(f"\n⏳ Cargando modelos e iniciando auditoría forense para '{args.titulo}' ({len(texto.split())} palabras)...")
    auditor = AuditorExplicabilidadNoticia()
    resultado = auditor.auditar_noticia(texto=texto, titulo=args.titulo)

    # Imprimir resumen en consola
    md_report = auditor.formatear_informe_markdown(resultado)
    print("\n" + "=" * 80)
    print(" 📋 INFORME FORENSE GENERADO")
    print("=" * 80)
    print(md_report)

    if args.salida_json:
        with open(args.salida_json, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Archivo JSON guardado con éxito en: {args.salida_json}")

    if args.salida_md:
        with open(args.salida_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"💾 Archivo Markdown guardado con éxito en: {args.salida_md}")

if __name__ == "__main__":
    main()
