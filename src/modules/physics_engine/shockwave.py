# Propagazione onde d'urto
"""
Modulo per la propagazione delle onde d'urto in mezzi acustici.

Implementa modelli di propagazione sferica con attenuazione geometrica
e assorbimento del mezzo.

Equazioni principali:
    - Attenuazione geometrica: P ∝ 1/r (onda sferica)
    - Attenuazione con assorbimento: P(r) = P_0 * (r_0/r)^n * exp(-α*r)
    - Rise time tipico: < 10 ns per onde d'urto ESWT

Riferimenti:
    - Ogden et al. (2001) - Principles of Shock Wave Therapy
    - Chen et al. (2010) - PAED shock wave propagation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants


@dataclass
class ShockwaveState:
    """
    Stato di un'onda d'urto in un punto dello spazio.

    Attributi:
        pressione_picco: Pressione di picco positiva (MPa)
        pressione_negativa: Pressione di picco negativa/tensile (MPa)
        rise_time: Tempo di salita (ns)
        durata_impulso: Durata totale dell'impulso (μs)
        distanza: Distanza dalla sorgente (mm)
    """

    pressione_picco: "Q_"
    pressione_negativa: "Q_"
    rise_time: "Q_"
    durata_impulso: "Q_"
    distanza: "Q_"

    @property
    def rapporto_pressioni(self) -> float:
        """Rapporto tra pressione positiva e negativa."""
        p_pos = abs(self.pressione_picco.to("MPa").magnitude)
        p_neg = abs(self.pressione_negativa.to("MPa").magnitude)
        return p_pos / p_neg if p_neg > 0 else float("inf")


@dataclass
class ShockwavePulse:
    """
    Rappresenta un impulso d'onda d'urto completo.

    Attributi:
        tempo: Array dei tempi (s)
        pressione: Array delle pressioni (Pa)
        energia_sorgente: Energia alla sorgente (J)
    """

    tempo: np.ndarray
    pressione: np.ndarray
    energia_sorgente: "Q_"

    @property
    def pressione_picco(self) -> "Q_":
        """Pressione di picco dell'impulso."""
        return Q_(np.max(self.pressione), "Pa")

    @property
    def pressione_minima(self) -> "Q_":
        """Pressione minima (tensile) dell'impulso."""
        return Q_(np.min(self.pressione), "Pa")

    @property
    def durata(self) -> "Q_":
        """Durata dell'impulso."""
        return Q_(self.tempo[-1] - self.tempo[0], "s")


def calcola_attenuazione(
    pressione_iniziale: "Q_",
    distanza_iniziale: "Q_",
    distanza_finale: "Q_",
    esponente_geometrico: float = 1.0,
    coefficiente_assorbimento: "Q_" = None,
) -> "Q_":
    """
    Calcola l'attenuazione di un'onda d'urto con la distanza.

    P(r) = P_0 * (r_0/r)^n * exp(-α*(r-r_0))

    Parametri:
        pressione_iniziale: Pressione alla distanza iniziale (Pa o MPa)
        distanza_iniziale: Distanza iniziale dalla sorgente (mm)
        distanza_finale: Distanza finale dalla sorgente (mm)
        esponente_geometrico: Esponente n (1.0 per sferica, 0.5 per cilindrica)
        coefficiente_assorbimento: Coefficiente α (1/m), None per trascurarlo

    Ritorna:
        Pressione alla distanza finale
    """
    p0 = pressione_iniziale.to("Pa").magnitude
    r0 = distanza_iniziale.to("m").magnitude
    r = distanza_finale.to("m").magnitude

    if r <= 0:
        raise ValueError("La distanza finale deve essere positiva")

    # Attenuazione geometrica
    fattore_geom = (r0 / r) ** esponente_geometrico

    # Attenuazione per assorbimento
    if coefficiente_assorbimento is not None:
        alpha = coefficiente_assorbimento.to("1/m").magnitude
        fattore_assorbimento = np.exp(-alpha * (r - r0))
    else:
        fattore_assorbimento = 1.0

    p_finale = p0 * fattore_geom * fattore_assorbimento
    return Q_(p_finale, "Pa")


def propaga_onda_sferica(
    pressione_sorgente: "Q_",
    distanza_sorgente: "Q_",
    distanze: np.ndarray,
    coefficiente_assorbimento: "Q_" = None,
) -> np.ndarray:
    """
    Propaga un'onda sferica a diverse distanze.

    Parametri:
        pressione_sorgente: Pressione alla sorgente (Pa o MPa)
        distanza_sorgente: Raggio della sorgente (mm)
        distanze: Array delle distanze (mm)
        coefficiente_assorbimento: Coefficiente α (1/m)

    Ritorna:
        Array delle pressioni alle diverse distanze (Pa)
    """
    p0 = pressione_sorgente.to("Pa").magnitude
    r0 = distanza_sorgente.to("m").magnitude

    # Converti distanze a metri
    r = distanze * 1e-3  # mm -> m

    # Evita divisione per zero
    r = np.maximum(r, r0)

    # Attenuazione geometrica (sferica)
    pressioni = p0 * (r0 / r)

    # Assorbimento
    if coefficiente_assorbimento is not None:
        alpha = coefficiente_assorbimento.to("1/m").magnitude
        pressioni *= np.exp(-alpha * (r - r0))

    return pressioni


def calcola_rise_time(pressione_picco: "Q_") -> "Q_":
    """
    Stima il rise time di un'onda d'urto dalla pressione di picco.

    Per onde d'urto ESWT, il rise time è tipicamente < 10 ns.
    Onde più intense tendono ad avere rise time più brevi.

    Parametri:
        pressione_picco: Pressione di picco (MPa)

    Ritorna:
        Rise time stimato (ns)
    """
    p = pressione_picco.to("MPa").magnitude

    # Modello empirico semplificato
    # Rise time diminuisce con l'aumentare della pressione
    # Tipicamente 5-10 ns per ESWT
    if p > 100:
        rise_time = 2.0  # ns per pressioni molto alte
    elif p > 50:
        rise_time = 5.0
    elif p > 20:
        rise_time = 8.0
    else:
        rise_time = 10.0

    return Q_(rise_time, "ns")


def genera_profilo_impulso(
    pressione_picco: "Q_",
    durata_positiva: "Q_" = None,
    durata_negativa: "Q_" = None,
    rise_time: "Q_" = None,
    n_punti: int = 1000,
) -> ShockwavePulse:
    """
    Genera un profilo temporale di impulso d'onda d'urto.

    Il profilo ha una fase positiva rapida seguita da una fase
    negativa (tensile) più lenta, tipico delle onde ESWT.

    Parametri:
        pressione_picco: Pressione di picco positiva (MPa)
        durata_positiva: Durata fase positiva (default: 1 μs)
        durata_negativa: Durata fase negativa (default: 5 μs)
        rise_time: Tempo di salita (default: calcolato)
        n_punti: Numero di punti temporali

    Ritorna:
        ShockwavePulse con il profilo temporale
    """
    p_max = pressione_picco.to("Pa").magnitude

    # Valori default
    if rise_time is None:
        rise_time = calcola_rise_time(pressione_picco)
    t_rise = rise_time.to("s").magnitude

    if durata_positiva is None:
        durata_positiva = Q_(1, "us")
    t_pos = durata_positiva.to("s").magnitude

    if durata_negativa is None:
        durata_negativa = Q_(5, "us")
    t_neg = durata_negativa.to("s").magnitude

    # Crea array temporale
    t_totale = t_rise + t_pos + t_neg
    tempo = np.linspace(0, t_totale, n_punti)

    # Genera profilo
    pressione = np.zeros_like(tempo)

    # Fase di salita (esponenziale rapida)
    mask_rise = tempo < t_rise
    pressione[mask_rise] = p_max * (1 - np.exp(-5 * tempo[mask_rise] / t_rise))

    # Fase positiva (decadimento esponenziale)
    mask_pos = (tempo >= t_rise) & (tempo < t_rise + t_pos)
    t_rel = tempo[mask_pos] - t_rise
    pressione[mask_pos] = p_max * np.exp(-3 * t_rel / t_pos)

    # Fase negativa (tensile)
    mask_neg = tempo >= t_rise + t_pos
    t_rel = tempo[mask_neg] - t_rise - t_pos
    p_neg = -0.1 * p_max  # Pressione negativa ~10% della positiva
    pressione[mask_neg] = p_neg * np.sin(np.pi * t_rel / t_neg) * np.exp(-2 * t_rel / t_neg)

    return ShockwavePulse(
        tempo=tempo,
        pressione=pressione,
        energia_sorgente=Q_(0, "J"),  # Da calcolare separatamente
    )


def calcola_velocita_onda(pressione: "Q_", mezzo: str = "acqua") -> "Q_":
    """
    Calcola la velocità dell'onda d'urto considerando effetti non lineari.

    Per onde d'urto intense, la velocità aumenta con la pressione.

    Parametri:
        pressione: Pressione dell'onda (MPa)
        mezzo: Tipo di mezzo ("acqua", "tessuto")

    Ritorna:
        Velocità dell'onda (m/s)
    """
    p = pressione.to("Pa").magnitude

    # Velocità base nel mezzo
    if mezzo == "acqua":
        c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude
        B_A = 5.0  # Parametro non lineare per acqua
    elif mezzo == "tessuto":
        c0 = 1540  # m/s tipico per tessuti molli
        B_A = 6.0
    else:
        c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude
        B_A = 5.0

    # Correzione non lineare (per pressioni elevate)
    # c = c0 * (1 + B/2A * p/ρc0²)
    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    correzione = 1 + (B_A / 2) * p / (rho * c0**2)

    # Limita la correzione per evitare valori irrealistici
    correzione = min(correzione, 1.5)

    return Q_(c0 * correzione, "m/s")


# =============================================================================
# PROPAGAZIONE NON LINEARE (FASE 2-3)
# =============================================================================

# Dati OssaTron da Ogden et al. (2001) Table 1 per calibrazione
OGDEN_DATA = {
    14: {"p_max": 40.6, "efd": 0.105, "d_lat": 6.8, "l_ax": 44.1, "rise_time_ns": 100},
    20: {"p_max": 45.6, "efd": 0.255, "d_lat": 6.4, "l_ax": 59.0, "rise_time_ns": 80},
    28: {"p_max": 71.9, "efd": 0.370, "d_lat": 8.7, "l_ax": 67.6, "rise_time_ns": 50},
}


def propaga_onda_nonlineare(
    pressione_iniziale: "Q_",
    distanza_iniziale: "Q_",
    distanza_finale: "Q_",
    beta: float = 3.5,
    coefficiente_assorbimento: "Q_" = None,
) -> "Q_":
    """
    Propaga un'onda d'urto con effetti non lineari (modello Burgers semplificato).

    Formula (weak shock approximation):
    P(r) = P₀·(r₀/r)·exp(-α·(r-r₀)) / [1 + β·P₀·(r-r₀)/(2ρc₀³)]

    Il termine non lineare causa:
    - Steepening del fronte d'onda (aumento gradiente)
    - Dissipazione aggiuntiva per onde intense
    - Formazione di shock discontinui

    Parametri:
        pressione_iniziale: Pressione alla distanza iniziale (MPa)
        distanza_iniziale: Distanza iniziale dalla sorgente (mm)
        distanza_finale: Distanza finale dalla sorgente (mm)
        beta: Parametro di non-linearità (default: 3.5 per acqua, β = 1 + B/2A)
        coefficiente_assorbimento: Coefficiente α (1/m)

    Ritorna:
        Pressione alla distanza finale (MPa)

    Riferimenti:
        - Blackstock (1966) - Weak shock propagation
        - Hamilton & Blackstock (1998) - Nonlinear Acoustics
    """
    p0 = pressione_iniziale.to("Pa").magnitude
    r0 = distanza_iniziale.to("m").magnitude
    r = distanza_finale.to("m").magnitude

    if r <= r0:
        return pressione_iniziale

    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    # Attenuazione geometrica (sferica)
    fattore_geom = r0 / r

    # Attenuazione per assorbimento
    if coefficiente_assorbimento is not None:
        alpha = coefficiente_assorbimento.to("1/m").magnitude
        fattore_assorbimento = np.exp(-alpha * (r - r0))
    else:
        fattore_assorbimento = 1.0

    # Termine non lineare (Burgers/weak shock)
    # Causa dissipazione aggiuntiva per onde intense
    delta_r = r - r0
    denominatore_nl = 1 + beta * p0 * delta_r / (2 * rho * c0**3)

    # Evita divisione per valori troppo piccoli
    denominatore_nl = max(denominatore_nl, 0.1)

    # Pressione finale
    p_finale = p0 * fattore_geom * fattore_assorbimento / denominatore_nl

    return Q_(p_finale / 1e6, "MPa")


def propaga_onda_nonlineare_array(
    pressione_sorgente: "Q_",
    distanza_sorgente: "Q_",
    distanze: np.ndarray,
    beta: float = 3.5,
    coefficiente_assorbimento: "Q_" = None,
) -> np.ndarray:
    """
    Propaga un'onda non lineare a diverse distanze.

    Parametri:
        pressione_sorgente: Pressione alla sorgente (MPa)
        distanza_sorgente: Raggio della sorgente (mm)
        distanze: Array delle distanze (mm)
        beta: Parametro di non-linearità
        coefficiente_assorbimento: Coefficiente α (1/m)

    Ritorna:
        Array delle pressioni alle diverse distanze (MPa)
    """
    p0 = pressione_sorgente.to("Pa").magnitude
    r0 = distanza_sorgente.to("m").magnitude
    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    # Converti distanze a metri
    r = distanze * 1e-3  # mm -> m

    # Evita divisione per zero
    r = np.maximum(r, r0)

    # Attenuazione geometrica
    fattore_geom = r0 / r

    # Assorbimento
    if coefficiente_assorbimento is not None:
        alpha = coefficiente_assorbimento.to("1/m").magnitude
        fattore_assorbimento = np.exp(-alpha * (r - r0))
    else:
        fattore_assorbimento = np.ones_like(r)

    # Termine non lineare
    delta_r = r - r0
    denominatore_nl = 1 + beta * p0 * delta_r / (2 * rho * c0**3)
    denominatore_nl = np.maximum(denominatore_nl, 0.1)

    # Pressioni in Pa, poi converti a MPa
    pressioni = p0 * fattore_geom * fattore_assorbimento / denominatore_nl

    return pressioni / 1e6  # MPa


def calcola_rise_time_calibrato(
    pressione_picco: "Q_",
    distanza: "Q_" = None,
) -> "Q_":
    """
    Calcola il rise time calibrato su dati sperimentali Ogden.

    Correlazione empirica basata su OssaTron (Ogden 2001 Table 1):
    - 14 kV: P=40.6 MPa, t_rise ≈ 100 ns
    - 20 kV: P=45.6 MPa, t_rise ≈ 80 ns
    - 28 kV: P=71.9 MPa, t_rise ≈ 50 ns

    Fitting: t_rise ∝ 1/sqrt(P_peak)
    t_rise = k / sqrt(P) con k calibrato sui dati

    Parametri:
        pressione_picco: Pressione di picco (MPa)
        distanza: Distanza dalla sorgente (mm) - per correzione propagazione

    Ritorna:
        Rise time calibrato (ns)
    """
    p = pressione_picco.to("MPa").magnitude

    if p <= 0:
        return Q_(100, "ns")

    # Coefficiente calibrato su dati Ogden
    # t_rise = k / sqrt(P)
    # Da P=40.6 MPa → t=100ns: k = 100 * sqrt(40.6) ≈ 637
    # Da P=71.9 MPa → t=50ns: k = 50 * sqrt(71.9) ≈ 424
    # Media: k ≈ 530
    k = 530

    t_rise = k / np.sqrt(p)

    # Limiti fisici
    t_rise = max(t_rise, 5)    # Min 5 ns (limite fisico)
    t_rise = min(t_rise, 200)  # Max 200 ns

    # Correzione per propagazione (rise time aumenta con distanza)
    if distanza is not None:
        d = distanza.to("mm").magnitude
        # Correzione empirica: +0.5 ns per mm di propagazione
        t_rise += 0.5 * max(0, d - 50)  # Oltre 50 mm dalla sorgente

    return Q_(t_rise, "ns")


def calcola_distanza_formazione_shock(
    pressione_iniziale: "Q_",
    frequenza: "Q_" = None,
    beta: float = 3.5,
) -> "Q_":
    """
    Calcola la distanza di formazione dello shock (shock formation distance).

    x_shock = ρ·c₀³ / (β·ω·P₀)

    Questa è la distanza a cui un'onda sinusoidale si trasforma in shock.

    Parametri:
        pressione_iniziale: Ampiezza pressione (MPa)
        frequenza: Frequenza caratteristica (default: 1 MHz)
        beta: Parametro non-linearità

    Ritorna:
        Distanza di formazione shock (mm)
    """
    p0 = pressione_iniziale.to("Pa").magnitude
    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    if frequenza is None:
        frequenza = Q_(1, "MHz")
    omega = 2 * np.pi * frequenza.to("Hz").magnitude

    if p0 <= 0 or omega <= 0:
        return Q_(float("inf"), "mm")

    x_shock = rho * c0**3 / (beta * omega * p0)

    return Q_(x_shock * 1000, "mm")  # m → mm


def calcola_attenuazione_shock(
    pressione_iniziale: "Q_",
    distanza: "Q_",
    beta: float = 3.5,
) -> float:
    """
    Calcola il fattore di attenuazione dovuto agli effetti non lineari.

    Ritorna il rapporto P/P₀ dovuto solo alla dissipazione non lineare
    (esclusa l'attenuazione geometrica).

    Parametri:
        pressione_iniziale: Pressione iniziale (MPa)
        distanza: Distanza percorsa (mm)
        beta: Parametro non-linearità

    Ritorna:
        Fattore di attenuazione (0-1, adimensionale)
    """
    p0 = pressione_iniziale.to("Pa").magnitude
    x = distanza.to("m").magnitude

    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c0 = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    denominatore = 1 + beta * p0 * x / (2 * rho * c0**3)

    if denominatore <= 0:
        return 0.0

    return 1.0 / denominatore


def genera_profilo_impulso_calibrato(
    pressione_picco: "Q_",
    tensione_kv: int = None,
    n_punti: int = 1000,
) -> ShockwavePulse:
    """
    Genera un profilo d'impulso calibrato su dati OssaTron.

    Usa i parametri di Ogden Table 1 per generare profili realistici.

    Parametri:
        pressione_picco: Pressione di picco positiva (MPa)
        tensione_kv: Tensione del dispositivo (14, 20, o 28 kV) per calibrazione
        n_punti: Numero di punti temporali

    Ritorna:
        ShockwavePulse con profilo calibrato
    """
    p_max = pressione_picco.to("Pa").magnitude

    # Seleziona parametri da Ogden se tensione specificata
    if tensione_kv is not None and tensione_kv in OGDEN_DATA:
        data = OGDEN_DATA[tensione_kv]
        rise_time = Q_(data["rise_time_ns"], "ns")
    else:
        rise_time = calcola_rise_time_calibrato(pressione_picco)

    t_rise = rise_time.to("s").magnitude

    # Durate tipiche ESWT (da letteratura)
    t_pos = 1e-6   # 1 μs fase positiva
    t_neg = 5e-6   # 5 μs fase negativa

    # Pressione negativa tipica: 10-20% della positiva
    p_neg_ratio = 0.15

    # Crea array temporale
    t_totale = t_rise + t_pos + t_neg
    tempo = np.linspace(0, t_totale, n_punti)

    # Genera profilo
    pressione = np.zeros_like(tempo)

    # Fase di salita (molto rapida, quasi discontinua)
    mask_rise = tempo < t_rise
    pressione[mask_rise] = p_max * (1 - np.exp(-5 * tempo[mask_rise] / t_rise))

    # Fase positiva (decadimento esponenziale)
    mask_pos = (tempo >= t_rise) & (tempo < t_rise + t_pos)
    t_rel = tempo[mask_pos] - t_rise
    pressione[mask_pos] = p_max * np.exp(-3 * t_rel / t_pos)

    # Fase negativa (tensile)
    mask_neg = tempo >= t_rise + t_pos
    t_rel = tempo[mask_neg] - t_rise - t_pos
    p_neg = -p_neg_ratio * p_max
    pressione[mask_neg] = p_neg * np.sin(np.pi * t_rel / t_neg) * np.exp(-2 * t_rel / t_neg)

    return ShockwavePulse(
        tempo=tempo,
        pressione=pressione,
        energia_sorgente=Q_(0, "J"),
    )
