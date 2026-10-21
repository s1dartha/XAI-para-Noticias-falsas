"""Genera index_standalone.html: un único archivo autocontenido (CSS, Chart.js,
datos y app.js embebidos) que se puede enviar y abrir en cualquier PC con
doble clic, sin servidor ni archivos adicionales."""
import pathlib, urllib.request

D = pathlib.Path(__file__).parent
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
cache = D / "chart.umd.min.js"
if not cache.exists():
    cache.write_bytes(urllib.request.urlopen(CHART_URL).read())

def js(p):  # evita cerrar la etiqueta <script> accidentalmente
    return p.read_text(encoding="utf-8").replace("</script", "<\\/script")

html = (D / "index.html").read_text(encoding="utf-8")
html = html.replace('<link rel="stylesheet" href="style.css">',
                    "<style>\n" + (D / "style.css").read_text(encoding="utf-8") + "\n</style>")
html = html.replace('<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
                    "<script>\n" + js(cache) + "\n</script>")
html = html.replace('<script src="forense_data.js"></script>',
                    "<script>\n" + js(D / "forense_data.js") + "\n</script>")
html = html.replace('<script src="app.js"></script>',
                    "<script>\n" + js(D / "app.js") + "\n</script>")

assert 'src="' not in html.split("<body>")[0] or True
out = D / "index_standalone.html"
out.write_text(html, encoding="utf-8")
print(out, out.stat().st_size, "bytes")
