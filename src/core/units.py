# Sistema di unità di misura per il simulatore ESWT
"""
Sistema di unità di misura basato su Pint.

Questo modulo fornisce un registry di unità di misura configurato per
le grandezze tipiche della fisica ESWT (pressioni in MPa, tempi in μs, etc.).

Uso tipico:
    from src.core.units import ureg, Q_

    # Creare grandezze con unità
    pressione = Q_(50, "MPa")
    tempo = Q_(10, "us")  # microsecondi
    energia = Q_(100, "J")

    # Conversioni automatiche
    pressione_bar = pressione.to("bar")

    # Verifiche dimensionali
    velocita = distanza / tempo  # Pint verifica la consistenza

Riferimenti:
    - Pint documentation: https://pint.readthedocs.io/
"""

import pint

# Creare il registry delle unità
ureg = pint.UnitRegistry()

# Alias per comodità - Quantity constructor
Q_ = ureg.Quantity

# Definire unità custom per ESWT se necessario
# (per ora usiamo le unità standard di Pint)

# Unità comunemente usate in ESWT:
# - Pressione: Pa, kPa, MPa, bar
# - Tempo: s, ms, us (microsecondi), ns (nanosecondi)
# - Energia: J, mJ, kJ
# - Potenza: W, kW, MW
# - Tensione: V, kV
# - Corrente: A, kA
# - Capacità: F, uF, nF
# - Induttanza: H, uH, mH
# - Resistenza: ohm, mohm
# - Lunghezza: m, cm, mm, um
# - Massa: kg, g
# - Densità: kg/m^3, g/cm^3
# - Velocità: m/s
# - Energy flux density: mJ/mm^2

# Configurazione per output più leggibile
ureg.formatter.default_format = "~P"  # Formato compatto con simboli


def verifica_dimensioni(grandezza: pint.Quantity, dimensione_attesa: str) -> bool:
    """
    Verifica che una grandezza abbia le dimensioni attese.

    Parametri:
        grandezza: Grandezza fisica con unità
        dimensione_attesa: Stringa con l'unità attesa (es. "MPa", "J", "m/s")

    Ritorna:
        True se le dimensioni sono compatibili, False altrimenti

    Esempio:
        >>> pressione = Q_(50, "MPa")
        >>> verifica_dimensioni(pressione, "Pa")  # True, stessa dimensione
        >>> verifica_dimensioni(pressione, "J")   # False, dimensione diversa
    """
    try:
        grandezza.to(dimensione_attesa)
        return True
    except pint.DimensionalityError:
        return False


def formatta_grandezza(grandezza: pint.Quantity, unita_output: str = None) -> str:
    """
    Formatta una grandezza fisica per output leggibile.

    Parametri:
        grandezza: Grandezza fisica con unità
        unita_output: Unità desiderata per l'output (opzionale)

    Ritorna:
        Stringa formattata della grandezza

    Esempio:
        >>> pressione = Q_(50000000, "Pa")
        >>> formatta_grandezza(pressione, "MPa")
        '50.0 MPa'
    """
    if unita_output:
        grandezza = grandezza.to(unita_output)
    return f"{grandezza:~.3fP}"


# Unità tipiche ESWT per riferimento rapido
UNITA_PRESSIONE = ureg.MPa
UNITA_TEMPO = ureg.microsecond
UNITA_ENERGIA = ureg.joule
UNITA_TENSIONE = ureg.kilovolt
UNITA_CORRENTE = ureg.kiloampere
UNITA_CAPACITA = ureg.microfarad
UNITA_INDUTTANZA = ureg.microhenry
