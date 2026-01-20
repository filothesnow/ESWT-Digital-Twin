# Modello di cavitazione per onde d'urto
"""
Modulo per la modellazione degli effetti di cavitazione in ESWT.

La cavitazione si verifica quando la fase di tensione (negativa) dell'onda
supera la resistenza tensile dinamica dell'acqua (~10 MPa).

Equazioni principali:
    - Condizione attivazione: |p_neg| > σ_tensile (~10 MPa)
    - Raggio massimo bolla: R_max = R₀·(P₀/P_amb)^(1/3)
    - Energia al collasso: E_collapse ∝ P_amb·R_max³

Riferimenti:
    - Ogden et al. (2001) - Principles of Shock Wave Therapy
    - Rayleigh (1917) - Pressure during collapse of spherical cavity
    - Plesset & Prosperetti (1977) - Bubble dynamics
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants


# Costanti per cavitazione
TENSIONE_ROTTURA_ACQUA = Q_(10, "MPa")  # Resistenza tensile dinamica acqua
RAGGIO_NUCLEI_TIPICO = Q_(1, "um")  # Raggio tipico nuclei di cavitazione


@dataclass
class CavitationBubble:
    """
    Rappresenta una bolla di cavitazione.

    Attributi:
        raggio_iniziale: Raggio iniziale del nucleo (μm)
        raggio_massimo: Raggio massimo raggiunto (μm)
        pressione_attivazione: Pressione negativa che ha attivato la bolla (MPa)
        energia_collasso: Energia rilasciata al collasso (mJ)
        tempo_vita: Tempo di vita della bolla (μs)
    """

    raggio_iniziale: "Q_"
    raggio_massimo: "Q_"
    pressione_attivazione: "Q_"
    energia_collasso: "Q_"
    tempo_vita: "Q_"


@dataclass
class CavitationResult:
    """
    Risultato dell'analisi di cavitazione.

    Attributi:
        cavitazione_attiva: True se la cavitazione si verifica
        pressione_negativa: Pressione negativa dell'impulso (MPa)
        soglia_superata: Di quanto la soglia è superata (MPa)
        raggio_bolla_max: Raggio massimo delle bolle (μm)
        energia_cavitazione: Energia totale della cavitazione (mJ)
        densita_bolle: Densità stimata di bolle (bolle/mm³)
    """

    cavitazione_attiva: bool
    pressione_negativa: "Q_"
    soglia_superata: "Q_"
    raggio_bolla_max: "Q_"
    energia_cavitazione: "Q_"
    densita_bolle: float


def verifica_cavitazione(
    pressione_negativa: "Q_",
    soglia: "Q_" = None,
) -> Tuple[bool, "Q_"]:
    """
    Verifica se la cavitazione si attiva.

    La cavitazione si verifica quando la pressione negativa (tensile)
    supera la resistenza tensile dinamica del liquido.

    Parametri:
        pressione_negativa: Pressione negativa dell'impulso (MPa, valore assoluto o negativo)
        soglia: Soglia di attivazione (default: 10 MPa per acqua)

    Ritorna:
        Tupla (attiva: bool, eccesso: Q_)
        - attiva: True se cavitazione presente
        - eccesso: Di quanto la soglia è superata (MPa)
    """
    if soglia is None:
        soglia = TENSIONE_ROTTURA_ACQUA

    p_neg = abs(pressione_negativa.to("MPa").magnitude)
    p_soglia = soglia.to("MPa").magnitude

    attiva = p_neg > p_soglia
    eccesso = max(0, p_neg - p_soglia)

    return attiva, Q_(eccesso, "MPa")


def calcola_raggio_bolla_max(
    pressione_picco: "Q_",
    raggio_iniziale: "Q_" = None,
    pressione_ambiente: "Q_" = None,
) -> "Q_":
    """
    Calcola il raggio massimo di una bolla di cavitazione.

    Durante la fase di tensione, i nuclei di gas pre-esistenti
    si espandono. Il raggio massimo dipende dal rapporto tra
    la pressione di picco e la pressione ambiente.

    Formula (Rayleigh):
    R_max = R₀ · (P_picco / P_ambiente)^(1/3)

    Parametri:
        pressione_picco: Pressione di picco positiva (MPa)
        raggio_iniziale: Raggio del nucleo (default: 1 μm)
        pressione_ambiente: Pressione ambiente (default: 1 atm)

    Ritorna:
        Raggio massimo della bolla (μm)
    """
    if raggio_iniziale is None:
        raggio_iniziale = RAGGIO_NUCLEI_TIPICO
    if pressione_ambiente is None:
        pressione_ambiente = PhysicalConstants.PRESSIONE_ATMOSFERICA

    r0 = raggio_iniziale.to("um").magnitude
    p_picco = pressione_picco.to("Pa").magnitude
    p_amb = pressione_ambiente.to("Pa").magnitude

    # R_max = R₀ · (P/P_amb)^(1/3)
    r_max = r0 * (p_picco / p_amb) ** (1 / 3)

    return Q_(r_max, "um")


def calcola_raggio_da_tensione(
    pressione_negativa: "Q_",
    raggio_iniziale: "Q_" = None,
    pressione_ambiente: "Q_" = None,
) -> "Q_":
    """
    Calcola il raggio massimo dalla pressione negativa (tensile).

    Durante la fase negativa, la bolla si espande fino al punto
    in cui la pressione interna eguaglia la tensione esterna.

    Parametri:
        pressione_negativa: Pressione negativa (MPa, valore assoluto)
        raggio_iniziale: Raggio del nucleo (default: 1 μm)
        pressione_ambiente: Pressione ambiente (default: 1 atm)

    Ritorna:
        Raggio massimo della bolla (μm)
    """
    if raggio_iniziale is None:
        raggio_iniziale = RAGGIO_NUCLEI_TIPICO
    if pressione_ambiente is None:
        pressione_ambiente = PhysicalConstants.PRESSIONE_ATMOSFERICA

    r0 = raggio_iniziale.to("um").magnitude
    p_neg = abs(pressione_negativa.to("Pa").magnitude)
    p_amb = pressione_ambiente.to("Pa").magnitude

    # La bolla si espande fino a quando P_gas ≈ |P_negativa|
    # Per gas ideale: P·V = cost → P·R³ = cost
    # R_max = R₀ · (P_amb / |P_neg|)^(1/3) se P_neg > P_amb
    # Altrimenti non c'è espansione significativa

    if p_neg > p_amb:
        # La bolla cresce quando la pressione negativa "tira"
        r_max = r0 * ((p_neg + p_amb) / p_amb) ** (1 / 3)
    else:
        r_max = r0

    return Q_(r_max, "um")


def stima_energia_collasso(
    raggio_max: "Q_",
    pressione_ambiente: "Q_" = None,
) -> "Q_":
    """
    Stima l'energia rilasciata al collasso di una bolla.

    L'energia è approssimativamente pari al lavoro fatto dalla
    pressione ambiente per comprimere la bolla:

    E = (4/3)·π·R_max³·P_ambiente

    Questa energia viene rilasciata come:
    - Onde d'urto secondarie
    - Microgetti (microjet)
    - Riscaldamento locale

    Parametri:
        raggio_max: Raggio massimo della bolla (μm)
        pressione_ambiente: Pressione ambiente (default: 1 atm)

    Ritorna:
        Energia di collasso (mJ)
    """
    if pressione_ambiente is None:
        pressione_ambiente = PhysicalConstants.PRESSIONE_ATMOSFERICA

    r = raggio_max.to("m").magnitude
    p = pressione_ambiente.to("Pa").magnitude

    # E = (4/3)·π·R³·P
    energia_j = (4 / 3) * np.pi * r**3 * p

    return Q_(energia_j * 1000, "mJ")  # J → mJ


def calcola_tempo_rayleigh_collasso(
    raggio: "Q_",
    pressione_ambiente: "Q_" = None,
    densita: "Q_" = None,
) -> "Q_":
    """
    Calcola il tempo di Rayleigh per il collasso di una bolla.

    t_R = 0.915 · R · sqrt(ρ / P)

    Questo è il tempo caratteristico per il collasso completo
    di una bolla sferica vuota.

    Parametri:
        raggio: Raggio della bolla (μm)
        pressione_ambiente: Pressione di collasso (default: 1 atm)
        densita: Densità del liquido (default: acqua)

    Ritorna:
        Tempo di Rayleigh (μs)
    """
    if pressione_ambiente is None:
        pressione_ambiente = PhysicalConstants.PRESSIONE_ATMOSFERICA
    if densita is None:
        densita = PhysicalConstants.DENSITA_ACQUA

    r = raggio.to("m").magnitude
    p = pressione_ambiente.to("Pa").magnitude
    rho = densita.to("kg/m^3").magnitude

    t_r = 0.915 * r * np.sqrt(rho / p)

    return Q_(t_r * 1e6, "us")  # s → μs


def calcola_pressione_collasso(
    raggio_max: "Q_",
    raggio_min: "Q_" = None,
    pressione_ambiente: "Q_" = None,
) -> "Q_":
    """
    Stima la pressione generata al collasso della bolla.

    Al collasso, la bolla raggiunge un raggio minimo e genera
    una pressione molto elevata. Per collasso adiabatico:

    P_collasso ≈ P_ambiente · (R_max / R_min)^(3γ)

    Parametri:
        raggio_max: Raggio massimo della bolla (μm)
        raggio_min: Raggio minimo al collasso (default: R_max/100)
        pressione_ambiente: Pressione ambiente (default: 1 atm)

    Ritorna:
        Pressione stimata al collasso (MPa)
    """
    if pressione_ambiente is None:
        pressione_ambiente = PhysicalConstants.PRESSIONE_ATMOSFERICA

    r_max = raggio_max.to("um").magnitude

    if raggio_min is None:
        r_min = r_max / 100  # Rapporto tipico
    else:
        r_min = raggio_min.to("um").magnitude

    p_amb = pressione_ambiente.to("Pa").magnitude
    gamma = 1.4  # Gas ideale

    # P = P_amb · (R_max/R_min)^(3γ)
    rapporto = r_max / max(r_min, 0.01)  # Evita divisione per zero
    p_collasso = p_amb * (rapporto ** (3 * gamma))

    # Limita a valori fisicamente ragionevoli
    p_collasso = min(p_collasso, 1e11)  # Max ~100 GPa

    return Q_(p_collasso / 1e6, "MPa")


def calcola_velocita_microjet(
    pressione_collasso: "Q_",
    densita: "Q_" = None,
) -> "Q_":
    """
    Stima la velocità del microjet al collasso asimmetrico.

    Quando una bolla collassa vicino a una superficie, si forma
    un getto di liquido (microjet) che può raggiungere alte velocità.

    v_jet ≈ sqrt(2·P_collasso / ρ)

    Parametri:
        pressione_collasso: Pressione al collasso (MPa)
        densita: Densità del liquido (default: acqua)

    Ritorna:
        Velocità del microjet (m/s)
    """
    if densita is None:
        densita = PhysicalConstants.DENSITA_ACQUA

    p = pressione_collasso.to("Pa").magnitude
    rho = densita.to("kg/m^3").magnitude

    v = np.sqrt(2 * p / rho)

    return Q_(v, "m/s")


def stima_densita_bolle(
    pressione_negativa: "Q_",
    soglia: "Q_" = None,
) -> float:
    """
    Stima la densità di bolle di cavitazione.

    La densità di nuclei che si attivano dipende dall'eccesso
    di pressione negativa sopra la soglia.

    Modello empirico:
    n = n₀ · exp(k · (|P_neg| - P_soglia))

    Parametri:
        pressione_negativa: Pressione negativa (MPa)
        soglia: Soglia di attivazione (default: 10 MPa)

    Ritorna:
        Densità stimata di bolle (bolle/mm³)
    """
    if soglia is None:
        soglia = TENSIONE_ROTTURA_ACQUA

    p_neg = abs(pressione_negativa.to("MPa").magnitude)
    p_soglia = soglia.to("MPa").magnitude

    if p_neg <= p_soglia:
        return 0.0

    # Densità base di nuclei pre-esistenti
    n0 = 1e3  # bolle/mm³ (stima conservativa)

    # Coefficiente di crescita esponenziale
    k = 0.1  # 1/MPa

    # Densità
    eccesso = p_neg - p_soglia
    n = n0 * np.exp(k * eccesso)

    # Limita a valori ragionevoli
    n = min(n, 1e7)  # Max 10^7 bolle/mm³

    return n


def analizza_cavitazione(
    pressione_picco: "Q_",
    pressione_negativa: "Q_",
    raggio_nucleo: "Q_" = None,
) -> CavitationResult:
    """
    Analisi completa della cavitazione per un impulso d'onda.

    Parametri:
        pressione_picco: Pressione di picco positiva (MPa)
        pressione_negativa: Pressione negativa (MPa, valore assoluto o negativo)
        raggio_nucleo: Raggio dei nuclei pre-esistenti (default: 1 μm)

    Ritorna:
        CavitationResult con tutti i parametri calcolati
    """
    if raggio_nucleo is None:
        raggio_nucleo = RAGGIO_NUCLEI_TIPICO

    # Verifica attivazione
    attiva, eccesso = verifica_cavitazione(pressione_negativa)

    if not attiva:
        return CavitationResult(
            cavitazione_attiva=False,
            pressione_negativa=Q_(abs(pressione_negativa.to("MPa").magnitude), "MPa"),
            soglia_superata=Q_(0, "MPa"),
            raggio_bolla_max=raggio_nucleo,
            energia_cavitazione=Q_(0, "mJ"),
            densita_bolle=0.0,
        )

    # Calcola parametri cavitazione
    r_max = calcola_raggio_da_tensione(pressione_negativa, raggio_nucleo)
    energia = stima_energia_collasso(r_max)
    densita = stima_densita_bolle(pressione_negativa)

    # Energia totale = energia per bolla × densità × volume tipico
    # Assumiamo volume zona focale ~1 mm³
    energia_totale = energia.to("mJ").magnitude * densita * 1e-3  # per mm³

    return CavitationResult(
        cavitazione_attiva=True,
        pressione_negativa=Q_(abs(pressione_negativa.to("MPa").magnitude), "MPa"),
        soglia_superata=eccesso,
        raggio_bolla_max=r_max,
        energia_cavitazione=Q_(energia_totale, "mJ"),
        densita_bolle=densita,
    )


def calcola_potenziale_danno_cavitazione(
    cavitation_result: CavitationResult,
) -> dict:
    """
    Valuta il potenziale di danno della cavitazione.

    Ritorna un dizionario con indicatori del potenziale distruttivo.

    Parametri:
        cavitation_result: Risultato dell'analisi cavitazione

    Ritorna:
        Dizionario con indicatori di danno
    """
    if not cavitation_result.cavitazione_attiva:
        return {
            "livello_danno": "nessuno",
            "score": 0,
            "velocita_microjet_m_s": 0,
            "pressione_collasso_mpa": 0,
        }

    # Calcola pressione al collasso
    p_collasso = calcola_pressione_collasso(cavitation_result.raggio_bolla_max)

    # Velocità microjet
    v_jet = calcola_velocita_microjet(p_collasso)

    # Score di danno (0-100)
    # Basato su: raggio bolla, energia, densità
    r_score = min(cavitation_result.raggio_bolla_max.to("um").magnitude / 100, 1) * 30
    e_score = min(cavitation_result.energia_cavitazione.to("mJ").magnitude, 1) * 40
    d_score = min(cavitation_result.densita_bolle / 1e5, 1) * 30

    score = r_score + e_score + d_score

    # Livello danno
    if score < 20:
        livello = "basso"
    elif score < 50:
        livello = "moderato"
    elif score < 80:
        livello = "alto"
    else:
        livello = "molto_alto"

    return {
        "livello_danno": livello,
        "score": score,
        "velocita_microjet_m_s": v_jet.to("m/s").magnitude,
        "pressione_collasso_mpa": p_collasso.to("MPa").magnitude,
    }
