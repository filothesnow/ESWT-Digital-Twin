# Calcolo energia per simulazioni ESWT
"""
Modulo per il calcolo dell'energia nelle scariche ESWT.

Implementa le equazioni di Chen et al. (2010) per:
    - Energia iniettata E(t)
    - Pressione picco empirica P_0
    - Bilancio energetico (riscaldamento vs shockwave)

Equazioni principali:
    E(t) = 0.5 * C * (U_M² - U(t)²)  # Energia iniettata al tempo t
    P_0 = k * E_B^α                   # Pressione picco empirica
    ΔE = m * c * ΔT                   # Energia termica

Riferimenti:
    - Chen et al. (2010) Eq. 1-3
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import ChenModelParameters, PhysicalConstants


@dataclass
class EnergyBreakdown:
    """
    Ripartizione dell'energia nella scarica ESWT.

    Secondo Chen et al. (2010), circa il 10% dell'energia serve
    per riscaldare l'acqua, il restante 90% genera l'onda d'urto.

    Attributi:
        energia_totale: Energia totale iniettata (J)
        energia_riscaldamento: Energia per riscaldamento acqua (~10%)
        energia_shockwave: Energia per generazione onda d'urto (~90%)
        efficienza: Rapporto energia_shockwave / energia_totale
    """

    energia_totale: "Q_"
    energia_riscaldamento: "Q_"
    energia_shockwave: "Q_"
    efficienza: float


def calcola_energia_condensatore(capacita: "Q_", tensione: "Q_") -> "Q_":
    """
    Calcola l'energia immagazzinata in un condensatore.

    E = 0.5 * C * V²

    Parametri:
        capacita: Capacità del condensatore (F)
        tensione: Tensione ai capi (V)

    Ritorna:
        Energia in Joule
    """
    c = capacita.to("F").magnitude
    v = tensione.to("V").magnitude
    return Q_(0.5 * c * v**2, "J")


def calcola_energia_rilasciata(
    capacita: "Q_",
    tensione_iniziale: "Q_",
    tensione_finale: "Q_",
) -> "Q_":
    """
    Calcola l'energia rilasciata durante una scarica parziale.

    ΔE = 0.5 * C * (V_i² - V_f²)

    Da Chen et al. (2010), Eq. 1.

    Parametri:
        capacita: Capacità del condensatore (F)
        tensione_iniziale: Tensione iniziale (V)
        tensione_finale: Tensione finale (V)

    Ritorna:
        Energia rilasciata in Joule
    """
    c = capacita.to("F").magnitude
    v_i = tensione_iniziale.to("V").magnitude
    v_f = tensione_finale.to("V").magnitude
    return Q_(0.5 * c * (v_i**2 - v_f**2), "J")


def calcola_pressione_picco_chen(
    energia_breakdown: "Q_",
    distanza: "Q_",
    k: float = None,
    alpha: float = None,
) -> "Q_":
    """
    Calcola la pressione di picco usando la formula empirica di Chen.

    P_0 = k * E_B^α

    Da Chen et al. (2010), Eq. 3.
    I coefficienti k e α dipendono dalla geometria e dalla distanza.

    Parametri:
        energia_breakdown: Energia al momento del breakdown (J)
        distanza: Distanza dal punto di misura (cm)
        k: Coefficiente empirico (default: stimato dalla distanza)
        alpha: Esponente empirico (default: ~0.5)

    Ritorna:
        Pressione di picco in MPa

    Note:
        I valori di default sono stime basate sui dati sperimentali
        di Chen et al. Valori più precisi richiedono calibrazione.
    """
    e = energia_breakdown.to("J").magnitude
    d = distanza.to("cm").magnitude

    # Stima coefficienti se non forniti
    # Basato su fitting dati Chen Table 1
    # Formula: P_0 = k * E^α / d
    # Fitting empirico dai 4 test di Chen (errore ~10%):
    # E=3300J, d=17.5cm → 8 MPa; E=600J, d=17.5cm → 4.4 MPa
    # E=31J, d=9cm → 2.8 MPa; E=20J, d=9cm → 2.0 MPa
    if k is None:
        k = 5.2 / d  # MPa·J^-α·cm (calibrato su dati Chen)

    if alpha is None:
        alpha = 0.4  # Tipico per PAED

    p0 = k * (e**alpha)
    return Q_(p0, "MPa")


def calcola_ripartizione_energia(energia_totale: "Q_") -> EnergyBreakdown:
    """
    Calcola la ripartizione dell'energia secondo il modello Chen.

    Circa il 10% dell'energia serve per riscaldare l'acqua fino
    alla vaporizzazione, il restante 90% genera l'onda d'urto.

    Parametri:
        energia_totale: Energia totale iniettata (J)

    Ritorna:
        EnergyBreakdown con la ripartizione
    """
    e_tot = energia_totale.to("J").magnitude

    fraz_risc = ChenModelParameters.FRAZIONE_ENERGIA_RISCALDAMENTO
    fraz_shock = ChenModelParameters.FRAZIONE_ENERGIA_SHOCKWAVE

    e_risc = e_tot * fraz_risc
    e_shock = e_tot * fraz_shock

    return EnergyBreakdown(
        energia_totale=energia_totale,
        energia_riscaldamento=Q_(e_risc, "J"),
        energia_shockwave=Q_(e_shock, "J"),
        efficienza=fraz_shock,
    )


def calcola_energia_riscaldamento_acqua(
    massa: "Q_",
    temperatura_iniziale: "Q_",
    temperatura_finale: "Q_",
) -> "Q_":
    """
    Calcola l'energia necessaria per riscaldare una massa d'acqua.

    ΔE = m * c * ΔT

    Da Chen et al. (2010), Eq. 2.

    Parametri:
        massa: Massa d'acqua (kg o g)
        temperatura_iniziale: Temperatura iniziale (°C o K)
        temperatura_finale: Temperatura finale (°C o K)

    Ritorna:
        Energia in Joule
    """
    m = massa.to("kg").magnitude
    c = PhysicalConstants.CALORE_SPECIFICO_ACQUA.magnitude

    # Converti temperature a Kelvin per calcolo ΔT
    t_i = temperatura_iniziale.to("K").magnitude
    t_f = temperatura_finale.to("K").magnitude

    delta_t = t_f - t_i
    return Q_(m * c * delta_t, "J")


def calcola_massa_acqua_riscaldata(
    energia: "Q_",
    temperatura_iniziale: "Q_" = None,
    temperatura_finale: "Q_" = None,
) -> "Q_":
    """
    Calcola la massa d'acqua che può essere riscaldata con una certa energia.

    m = ΔE / (c * ΔT)

    Parametri:
        energia: Energia disponibile (J)
        temperatura_iniziale: Temperatura iniziale (default: 25°C)
        temperatura_finale: Temperatura finale (default: 95°C)

    Ritorna:
        Massa d'acqua in kg
    """
    e = energia.to("J").magnitude
    c = PhysicalConstants.CALORE_SPECIFICO_ACQUA.magnitude

    t_i = (temperatura_iniziale or Q_(25, "degC")).to("K").magnitude
    t_f = (
        temperatura_finale or ChenModelParameters.TEMPERATURA_VAPORIZZAZIONE_TARGET
    ).to("K").magnitude

    delta_t = t_f - t_i
    m = e / (c * delta_t)
    return Q_(m, "kg")


def calcola_volume_plasma_chen(energia: "Q_", gap: "Q_") -> "Q_":
    """
    Stima il volume della zona di plasma secondo Chen et al.

    Nel modello Chen, l'energia viene iniettata in una sfera di
    diametro pari alla distanza inter-elettrodo.

    Parametri:
        energia: Energia iniettata (J)
        gap: Distanza inter-elettrodo (mm)

    Ritorna:
        Volume della sfera di plasma (mm³)
    """
    d = gap.to("mm").magnitude
    # Volume sfera = (4/3) * π * r³ dove r = d/2
    r = d / 2
    volume = (4 / 3) * np.pi * r**3
    return Q_(volume, "mm^3")


def calcola_pulse_integral_intensity(
    pressione_array: np.ndarray,
    tempo_array: np.ndarray,
    densita: "Q_" = None,
    velocita_suono: "Q_" = None,
) -> "Q_":
    """
    Calcola il Pulse Integral Intensity (PII) da dati pressione-tempo.

    PII = (1/ρc) * ∫ p²(t) dt

    Questo è l'energy flux density spesso usato per caratterizzare
    i dispositivi ESWT.

    Parametri:
        pressione_array: Array delle pressioni (Pa)
        tempo_array: Array dei tempi (s)
        densita: Densità del mezzo (default: acqua)
        velocita_suono: Velocità del suono (default: acqua)

    Ritorna:
        PII in mJ/mm²
    """
    rho = (densita or PhysicalConstants.DENSITA_ACQUA).to("kg/m^3").magnitude
    c = (velocita_suono or PhysicalConstants.VELOCITA_SUONO_ACQUA).to("m/s").magnitude

    # Integrazione numerica di p²
    p_squared = pressione_array**2
    integral = np.trapezoid(p_squared, tempo_array)

    # PII in J/m²
    pii_j_m2 = integral / (rho * c)

    # Converti a mJ/mm²
    pii_mj_mm2 = pii_j_m2 * 1e-3  # J → mJ, m² → mm² si cancella

    return Q_(pii_mj_mm2, "mJ/mm^2")


def stima_energia_da_pressione(
    pressione_picco: "Q_",
    distanza: "Q_",
    k: float = None,
    alpha: float = None,
) -> "Q_":
    """
    Stima l'energia dalla pressione di picco (inverso di Chen Eq. 3).

    E_B = (P_0 / k)^(1/α)

    Utile per stimare l'energia necessaria per ottenere una certa pressione.

    Parametri:
        pressione_picco: Pressione desiderata (MPa)
        distanza: Distanza dal target (cm)
        k: Coefficiente empirico
        alpha: Esponente empirico

    Ritorna:
        Energia stimata in Joule
    """
    p0 = pressione_picco.to("MPa").magnitude
    d = distanza.to("cm").magnitude

    if k is None:
        k = 5.2 / d  # Coerente con calcola_pressione_picco_chen

    if alpha is None:
        alpha = 0.4

    e = (p0 / k) ** (1 / alpha)
    return Q_(e, "J")


def calcola_efficienza_scarica(
    energia_iniziale: "Q_",
    energia_plasma: "Q_",
    energia_shockwave: "Q_" = None,
) -> dict:
    """
    Calcola l'efficienza della scarica.

    Parametri:
        energia_iniziale: Energia immagazzinata nel condensatore
        energia_plasma: Energia dissipata nel plasma
        energia_shockwave: Energia convertita in onda d'urto (opzionale)

    Ritorna:
        Dizionario con le efficienze calcolate
    """
    e_in = energia_iniziale.to("J").magnitude
    e_plasma = energia_plasma.to("J").magnitude

    efficienza_plasma = e_plasma / e_in if e_in > 0 else 0

    result = {
        "energia_iniziale_J": e_in,
        "energia_plasma_J": e_plasma,
        "efficienza_plasma": efficienza_plasma,
    }

    if energia_shockwave:
        e_shock = energia_shockwave.to("J").magnitude
        result["energia_shockwave_J"] = e_shock
        result["efficienza_shockwave"] = e_shock / e_in if e_in > 0 else 0

    return result
