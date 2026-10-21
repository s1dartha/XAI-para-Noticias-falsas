"""
=============================================================================
CLASIFICACIÓN FLEXIBLE DE ARQUETIPOS DISCURSIVOS (AFINIDAD Y REGLAS RELAJADAS)
=============================================================================
Reemplaza el árbol rígido de umbrales duros `if/elif` por una evaluación
flexible basada en la AFINIDAD (grado de parecido morfoestilométrico) de la
noticia con cada uno de los 5 arquetipos discursivos sobre las 5 dimensiones
del IML (D1 a D5, escala 0 a 100).

Cada arquetipo modela su perfil de rasgos característicos mediante funciones
de pertenencia continuas (rampas lineales [0, 1]):
- 0.0 = La noticia no presenta el rasgo del arquetipo.
- 1.0 = La noticia presenta el rasgo en su máxima expresión.

La afinidad de la noticia con cada arquetipo es el promedio ponderado de
sus rasgos. El arquetipo asignado es el de mayor afinidad, y se reporta
también el arquetipo secundario para enriquecer la lectura forense en casos
de estilo híbrido.
=============================================================================
"""

from typing import Dict, List, Tuple


def ramp_up(x: float, lo: float, hi: float) -> float:
    """Función de pertenencia creciente: 0 si x <= lo, 1 si x >= hi, lineal entre ambos."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def ramp_down(x: float, hi: float, lo: float) -> float:
    """Función de pertenencia decreciente: 1 si x <= lo, 0 si x >= hi, lineal entre ambos."""
    return 1.0 - ramp_up(x, lo, hi)


def bell_band(x: float, lo: float, mid_lo: float, mid_hi: float, hi: float) -> float:
    """Banda trapezoidal: 1.0 en [mid_lo, mid_hi], decae hacia lo y hi."""
    if x <= lo or x >= hi:
        return 0.0
    if mid_lo <= x <= mid_hi:
        return 1.0
    if x < mid_lo:
        return (x - lo) / (mid_lo - lo)
    return (hi - x) / (hi - mid_hi)


ARQUETIPOS_META = [
    (
        "ESTILO_I_DESINFO_ESTRIDENTE",
        "Estilo I: Desinformación Estridente / Cliché Hiperbólico",
        "Alarma emocional sostenida en todo el texto, sin amortiguación del cuerpo; la repetición circular de premisas agrava la estridencia.",
        "Afinidad = 0.45·(D1 alta) + 0.30·(D3 baja/sin amortiguar) + 0.25·(D4 bucle circular)",
    ),
    (
        "ESTILO_II_CLICKBAIT_CABECERA",
        "Estilo II: Cebo Comercial / Clickbait de Cabecera",
        "Titular o apertura alarmista diseñado para capturar clics, pero con cuerpo informativo formal que amortigua y diluye la alarma.",
        "Afinidad = 0.45·(D2 gatillo alto) + 0.40·(D3 amortiguación alta) + 0.15·(D1 moderada/baja)",
    ),
    (
        "ESTILO_III_PROPAGANDA_SOBRIA",
        "Estilo III: Propaganda Institucional / Astroturfing Encubierto",
        "Tono formal, sobrio y frío, pero con bucles de repetición semántica calculados para inducir verdad ilusoria.",
        "Afinidad = 0.45·(D4 bucle alto) + 0.35·(D1 sobria/baja) + 0.20·(D2 gatillo bajo/homogéneo)",
    ),
    (
        "ESTILO_IV_DESARTICULADO",
        "Estilo IV: Incoherencia Estructural / Generación Rota",
        "Texto con baja conectividad proposicional, fragmentación sintáctica o párrafos desvinculados sin hilo temático.",
        "Afinidad = 0.50·(D5 cohesión rota) + 0.35·(D4 redundancia dispersa) + 0.15·(D3 baja)",
    ),
    (
        "ESTILO_V_PERIODISMO_EQUILIBRADO",
        "Estilo V: Periodismo Profesional Balanceado",
        "Redacción equilibrada con variedad léxica, desarrollo temático progresivo en Banda de Shannon y ausencia de estridencias.",
        "Afinidad = 0.35·(D5 cohesión óptima) + 0.30·(D1 sobria) + 0.20·(D4 progresión lineal) + 0.15·(D2 equilibrada)",
    ),
]


def calcular_afinidades(d1: float, d2: float, d3: float, d4: float, d5: float) -> List[float]:
    """Calcula las 5 afinidades en escala [0.0, 1.0] para (D1, D2, D3, D4, D5)."""
    # 1. Estilo I: Alarma sostenida + baja amortiguación (+ bucle agrava)
    a1 = (
        0.45 * ramp_up(d1, 40.0, 75.0)
        + 0.30 * ramp_down(d3, 60.0, 10.0)
        + 0.25 * ramp_up(d4, 35.0, 75.0)
    )

    # 2. Estilo II: Cebo de cabecera (D2 alta) + Amortiguación del cuerpo (D3 alta)
    a2 = (
        0.45 * ramp_up(d2, 35.0, 65.0)
        + 0.40 * ramp_up(d3, 40.0, 75.0)
        + 0.15 * ramp_down(d1, 70.0, 30.0)
    )

    # 3. Estilo III: Propaganda sobria (D4 bucle alto + D1 sobria + D2 sin gatillo clickbait)
    a3 = (
        0.45 * ramp_up(d4, 45.0, 80.0)
        + 0.35 * ramp_down(d1, 60.0, 20.0)
        + 0.20 * ramp_down(d2, 65.0, 25.0)
    )

    # 4. Estilo IV: Roto/Desarticulado (D5 baja + D4 baja/dispersa)
    a4 = (
        0.50 * ramp_down(d5, 70.0, 38.0)
        + 0.35 * ramp_down(d4, 55.0, 25.0)
        + 0.15 * ramp_down(d3, 60.0, 20.0)
    )

    # 5. Estilo V: Periodismo profesional balanceado (D5 óptimo + D1 sobrio + D4 controlado + D2 equilibrado)
    a5 = (
        0.35 * ramp_up(d5, 45.0, 75.0)
        + 0.30 * ramp_down(d1, 55.0, 20.0)
        + 0.20 * bell_band(d4, 20.0, 30.0, 60.0, 75.0)
        + 0.15 * ramp_down(d2, 55.0, 25.0)
    )

    return [a1, a2, a3, a4, a5]


def clasificar_arquetipo(d1: float, d2: float, d3: float, d4: float, d5: float) -> Dict:
    """
    Asigna el arquetipo de mayor afinidad estilométrica y devuelve el diagnóstico completo:
    - id, nombre, descripción
    - afinidad_pct del arquetipo ganador
    - arquetipo_secundario y afinidad_secundaria_pct
    - diccionario completo de afinidades_pct
    - condicion_activada detallada
    """
    afs = calcular_afinidades(d1, d2, d3, d4, d5)
    orden = sorted(range(5), key=lambda i: (-afs[i], i))
    ganador_idx = orden[0]
    segundo_idx = orden[1]

    aid, anombre, adesc, aregla = ARQUETIPOS_META[ganador_idx]
    sid, snombre, sdesc, sregla = ARQUETIPOS_META[segundo_idx]

    afinidad_ganador_pct = round(afs[ganador_idx] * 100.0, 1)
    afinidad_segundo_pct = round(afs[segundo_idx] * 100.0, 1)

    afinidades_dict = {
        ARQUETIPOS_META[k][0]: round(afs[k] * 100.0, 1) for k in range(5)
    }

    condicion = (
        f"Afinidad dominante = {afinidad_ganador_pct}% ({aregla}); "
        f"2.º arquetipo afín: {snombre.split(':')[0]} ({afinidad_segundo_pct}%)"
    )

    return {
        "id": aid,
        "nombre": anombre,
        "descripcion": adesc,
        "afinidad_pct": afinidad_ganador_pct,
        "arquetipo_secundario": snombre,
        "afinidad_secundaria_pct": afinidad_segundo_pct,
        "afinidades_pct": afinidades_dict,
        "condicion_activada": condicion,
    }
