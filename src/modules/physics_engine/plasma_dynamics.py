# Dinamica del plasma e formazione onda d'urto
"""
Modulo per la modellazione della dinamica del plasma e formazione dell'onda d'urto.

Implementa i modelli per la transizione:
    Energia elettrica → Plasma → Espansione bolla → Onda d'urto

Equazioni principali:
    - Bilancio energetico plasma: dT/dt = (P_in - P_rad) / (ρ·V·Cv)
    - Espansione bolla (Rayleigh-Plesset semplificato): R·R'' + (3/2)·R'² = (P_p - P_amb) / ρ
    - Pressione da espansione: P(r) = P_plasma · (R/r)³

Riferimenti:
    - Rayleigh (1917) - Collapse of spherical cavity
    - Plesset (1949) - Dynamics of cavitation bubbles
    - Chen et al. (2010) - PAED shock wave propagation
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants, ChenModelParameters


@dataclass
class PlasmaState:
    """
    Stato del plasma durante la scarica.

    Attributi:
        temperatura: Temperatura del plasma (K)
        raggio: Raggio del canale/bolla di plasma (mm)
        pressione: Pressione interna del plasma (MPa)
        energia_termica: Energia termica accumulata (J)
    """

    temperatura: "Q_"
    raggio: "Q_"
    pressione: "Q_"
    energia_termica: "Q_"


@dataclass
class BubbleExpansionResult:
    """
    Risultato della simulazione di espansione della bolla.

    Attributi:
        tempo: Array dei tempi (s)
        raggio: Array dei raggi della bolla (m)
        velocita_parete: Array delle velocità della parete (m/s)
        pressione_bolla: Pressione iniziale nella bolla (MPa)
        raggio_massimo: Raggio massimo raggiunto (mm)
        tempo_espansione: Tempo per raggiungere raggio massimo (μs)
        pressione_onda: Pressione dell'onda d'urto alla parete (MPa)
    """

    tempo: np.ndarray
    raggio: np.ndarray
    velocita_parete: np.ndarray
    pressione_bolla: "Q_"
    raggio_massimo: "Q_"
    tempo_espansione: "Q_"
    pressione_onda: "Q_"


@dataclass
class PlasmaExpansionResult:
    """
    Risultato completo della simulazione plasma → onda d'urto.

    Attributi:
        pressione_iniziale: Pressione generata dal plasma (MPa)
        raggio_bolla: Raggio della bolla di plasma (mm)
        tempo_espansione: Tempo caratteristico di espansione (μs)
        temperatura_plasma: Temperatura stimata del plasma (K)
        energia_meccanica: Energia convertita in onda (J)
    """

    pressione_iniziale: "Q_"
    raggio_bolla: "Q_"
    tempo_espansione: "Q_"
    temperatura_plasma: "Q_"
    energia_meccanica: "Q_"


def calcola_pressione_plasma(
    energia: "Q_",
    volume: "Q_",
    gamma: float = 1.4,
) -> "Q_":
    """
    Calcola la pressione del plasma assumendo gas ideale.

    P = (γ - 1) · E / V

    Per plasma ad alta temperatura, γ ≈ 1.2-1.4.

    Parametri:
        energia: Energia termica nel plasma (J)
        volume: Volume del canale di plasma (mm³)
        gamma: Rapporto calori specifici (default: 1.4)

    Ritorna:
        Pressione del plasma (MPa)
    """
    e = energia.to("J").magnitude
    v = volume.to("m^3").magnitude

    # P = (γ - 1) · E / V
    p = (gamma - 1) * e / v

    return Q_(p / 1e6, "MPa")  # Converti Pa → MPa


def calcola_temperatura_plasma(
    energia: "Q_",
    massa: "Q_",
    cv: "Q_" = None,
) -> "Q_":
    """
    Stima la temperatura del plasma dal bilancio energetico.

    T = E / (m · Cv)

    Parametri:
        energia: Energia termica (J)
        massa: Massa del plasma/vapore (kg)
        cv: Calore specifico a volume costante (default: vapore acqueo)

    Ritorna:
        Temperatura stimata (K)
    """
    e = energia.to("J").magnitude
    m = massa.to("kg").magnitude

    # Cv vapore acqueo ≈ 1400 J/(kg·K) a temperature elevate
    cv_val = cv.to("J/(kg*K)").magnitude if cv else 1400

    if m <= 0:
        return Q_(10000, "K")  # Valore limite per massa zero

    t = e / (m * cv_val)

    # Limita a valori fisicamente ragionevoli per plasma ESWT
    t = min(t, 50000)  # Max ~50000 K
    t = max(t, 300)  # Min temperatura ambiente

    return Q_(t, "K")


def calcola_raggio_iniziale_bolla(
    energia: "Q_",
    pressione_plasma: "Q_",
) -> "Q_":
    """
    Calcola il raggio iniziale della bolla di plasma.

    Assumendo che l'energia sia contenuta in una sfera a pressione P:
    E = P · V = P · (4/3)πr³
    r = (3E / 4πP)^(1/3)

    Parametri:
        energia: Energia nel plasma (J)
        pressione_plasma: Pressione del plasma (MPa)

    Ritorna:
        Raggio iniziale della bolla (mm)
    """
    e = energia.to("J").magnitude
    p = pressione_plasma.to("Pa").magnitude

    if p <= 0:
        raise ValueError("La pressione deve essere positiva")

    r = (3 * e / (4 * np.pi * p)) ** (1 / 3)

    return Q_(r * 1000, "mm")  # Converti m → mm


def rayleigh_plesset_semplificato(
    t: float,
    y: np.ndarray,
    p_bolla: float,
    p_ambiente: float,
    rho: float,
) -> np.ndarray:
    """
    Equazione di Rayleigh-Plesset semplificata per espansione bolla.

    R·R'' + (3/2)·R'² = (P_bolla - P_ambiente) / ρ

    Forma come sistema del primo ordine:
        dR/dt = v
        dv/dt = [(P_bolla - P_ambiente)/ρ - (3/2)v²] / R

    Parametri:
        t: Tempo (s)
        y: Vettore stato [R, v] (m, m/s)
        p_bolla: Pressione nella bolla (Pa)
        p_ambiente: Pressione ambiente (Pa)
        rho: Densità liquido (kg/m³)

    Ritorna:
        Derivate [dR/dt, dv/dt]
    """
    R, v = y

    # Evita divisione per zero
    R = max(R, 1e-9)

    # Pressione nella bolla decresce con espansione (gas ideale)
    # P_bolla(R) = P_bolla_0 · (R_0/R)^(3γ) ≈ P_bolla_0 · (R_0/R)^3
    # Per semplicità, usiamo pressione costante nella fase iniziale

    # Derivate
    dR_dt = v
    dv_dt = ((p_bolla - p_ambiente) / rho - 1.5 * v**2) / R

    return np.array([dR_dt, dv_dt])


def simula_espansione_bolla(
    pressione_bolla: "Q_",
    raggio_iniziale: "Q_",
    tempo_simulazione: "Q_" = None,
    n_punti: int = 1000,
) -> BubbleExpansionResult:
    """
    Simula l'espansione della bolla di plasma usando Rayleigh-Plesset.

    Parametri:
        pressione_bolla: Pressione iniziale nella bolla (MPa)
        raggio_iniziale: Raggio iniziale della bolla (mm)
        tempo_simulazione: Durata simulazione (default: 10 μs)
        n_punti: Numero di punti temporali

    Ritorna:
        BubbleExpansionResult con la dinamica della bolla
    """
    p_bolla = pressione_bolla.to("Pa").magnitude
    r0 = raggio_iniziale.to("m").magnitude
    p_amb = PhysicalConstants.PRESSIONE_ATMOSFERICA.to("Pa").magnitude
    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude

    if tempo_simulazione is None:
        tempo_simulazione = Q_(10, "us")
    t_max = tempo_simulazione.to("s").magnitude

    # Condizioni iniziali [R, v]
    y0 = np.array([r0, 0.0])  # Velocità iniziale nulla

    # Risolvi ODE
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_punti)

    sol = solve_ivp(
        lambda t, y: rayleigh_plesset_semplificato(t, y, p_bolla, p_amb, rho),
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        max_step=t_max / 100,
    )

    raggio = sol.y[0]
    velocita = sol.y[1]

    # Trova raggio massimo e tempo
    idx_max = np.argmax(raggio)
    r_max = raggio[idx_max]
    t_expansion = sol.t[idx_max]

    # Pressione dell'onda d'urto alla parete della bolla
    # P_onda ≈ ρ · c · v_parete (per velocità moderata)
    # Per alte velocità: P_onda ≈ ρ · v² (pressione dinamica)
    c = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude
    v_max = np.max(np.abs(velocita))

    # Usa formula per shock: P ≈ ρ·c·v per v << c, ρ·v² per v ~ c
    if v_max < c / 10:
        p_onda = rho * c * v_max
    else:
        p_onda = rho * v_max**2

    return BubbleExpansionResult(
        tempo=sol.t,
        raggio=raggio,
        velocita_parete=velocita,
        pressione_bolla=pressione_bolla,
        raggio_massimo=Q_(r_max * 1000, "mm"),
        tempo_espansione=Q_(t_expansion * 1e6, "us"),
        pressione_onda=Q_(p_onda / 1e6, "MPa"),
    )


def calcola_pressione_da_espansione(
    pressione_bolla: "Q_",
    raggio_bolla: "Q_",
    distanza: "Q_",
) -> "Q_":
    """
    Calcola la pressione dell'onda a una certa distanza dalla bolla.

    Per espansione sferica rapida:
    P(r) = P_bolla · (R/r)³  per r > R (zona vicina)
    P(r) = P_bolla · (R/r)   per r >> R (zona lontana, onda sferica)

    Questa funzione usa un modello interpolato.

    Parametri:
        pressione_bolla: Pressione nella bolla (MPa)
        raggio_bolla: Raggio della bolla (mm)
        distanza: Distanza dal centro (mm)

    Ritorna:
        Pressione all'onda a quella distanza (MPa)
    """
    p0 = pressione_bolla.to("MPa").magnitude
    r_bolla = raggio_bolla.to("mm").magnitude
    r = distanza.to("mm").magnitude

    if r <= r_bolla:
        # Dentro la bolla
        return pressione_bolla

    # Rapporto distanza/raggio
    ratio = r / r_bolla

    # Modello interpolato:
    # - Zona vicina (r < 3R): attenuazione ∝ 1/r³
    # - Zona intermedia (3R < r < 10R): transizione
    # - Zona lontana (r > 10R): attenuazione ∝ 1/r (onda sferica)

    if ratio < 3:
        # Zona vicina: P ∝ 1/r³
        p = p0 * (r_bolla / r) ** 3
    elif ratio < 10:
        # Zona intermedia: interpolazione
        # Transizione graduale da 1/r³ a 1/r
        f = (ratio - 3) / 7  # 0 a r=3R, 1 a r=10R
        exp = 3 - 2 * f  # da 3 a 1
        p = p0 * (r_bolla / r) ** exp
    else:
        # Zona lontana: P ∝ 1/r
        # Normalizzato per continuità a r=10R
        p_at_10R = p0 * (r_bolla / (10 * r_bolla)) ** 1
        p = p_at_10R * (10 * r_bolla / r)

    return Q_(p, "MPa")


def calcola_espansione_da_energia(
    energia_plasma: "Q_",
    gap_elettrodi: "Q_",
    efficienza: float = None,
) -> PlasmaExpansionResult:
    """
    Calcola l'espansione del plasma dall'energia della scarica.

    Pipeline completa:
    1. Calcola energia per onda d'urto (90% da Chen)
    2. Stima volume iniziale (dal gap elettrodi)
    3. Calcola pressione plasma
    4. Simula espansione bolla
    5. Determina pressione onda risultante

    Parametri:
        energia_plasma: Energia dissipata nel plasma (J)
        gap_elettrodi: Distanza inter-elettrodo (mm)
        efficienza: Frazione energia → onda (default: 0.9)

    Ritorna:
        PlasmaExpansionResult con tutti i parametri calcolati
    """
    if efficienza is None:
        efficienza = ChenModelParameters.FRAZIONE_ENERGIA_SHOCKWAVE

    # Energia disponibile per l'onda d'urto
    e_plasma = energia_plasma.to("J").magnitude
    e_onda = e_plasma * efficienza

    # Volume iniziale (sfera di diametro = gap)
    gap = gap_elettrodi.to("mm").magnitude
    r0 = gap / 2  # mm
    volume_mm3 = (4 / 3) * np.pi * r0**3
    volume = Q_(volume_mm3, "mm^3")

    # Pressione iniziale del plasma
    pressione = calcola_pressione_plasma(Q_(e_onda, "J"), volume)

    # Limita pressione a valori ragionevoli (evita overflow numerici)
    p_val = pressione.to("MPa").magnitude
    p_val = min(p_val, 10000)  # Max 10 GPa
    pressione = Q_(p_val, "MPa")

    # Stima temperatura
    # Massa vapore ≈ volume · densità vapore (~1 kg/m³ a pressione)
    massa_vapore = Q_(volume_mm3 * 1e-9 * 1, "kg")  # Stima approssimata
    temperatura = calcola_temperatura_plasma(Q_(e_onda, "J"), massa_vapore)

    # Simula espansione bolla
    raggio_iniziale = Q_(r0, "mm")
    espansione = simula_espansione_bolla(pressione, raggio_iniziale)

    return PlasmaExpansionResult(
        pressione_iniziale=espansione.pressione_onda,
        raggio_bolla=espansione.raggio_massimo,
        tempo_espansione=espansione.tempo_espansione,
        temperatura_plasma=temperatura,
        energia_meccanica=Q_(e_onda, "J"),
    )


def calcola_velocita_espansione_iniziale(
    pressione_bolla: "Q_",
    densita_liquido: "Q_" = None,
) -> "Q_":
    """
    Calcola la velocità iniziale di espansione della parete della bolla.

    v = sqrt(2 · (P_bolla - P_amb) / ρ)

    Parametri:
        pressione_bolla: Pressione nella bolla (MPa)
        densita_liquido: Densità del liquido (default: acqua)

    Ritorna:
        Velocità iniziale della parete (m/s)
    """
    p_bolla = pressione_bolla.to("Pa").magnitude
    p_amb = PhysicalConstants.PRESSIONE_ATMOSFERICA.to("Pa").magnitude
    rho = (densita_liquido or PhysicalConstants.DENSITA_ACQUA).to("kg/m^3").magnitude

    delta_p = p_bolla - p_amb
    if delta_p <= 0:
        return Q_(0, "m/s")

    v = np.sqrt(2 * delta_p / rho)
    return Q_(v, "m/s")


def tempo_rayleigh(
    raggio: "Q_",
    pressione_collasso: "Q_" = None,
    densita: "Q_" = None,
) -> "Q_":
    """
    Calcola il tempo di Rayleigh per collasso/espansione di una bolla.

    t_R = 0.915 · R · sqrt(ρ / P)

    Questo tempo caratterizza la scala temporale della dinamica della bolla.

    Parametri:
        raggio: Raggio della bolla (mm)
        pressione_collasso: Pressione di collasso (default: 1 atm)
        densita: Densità del liquido (default: acqua)

    Ritorna:
        Tempo di Rayleigh (μs)
    """
    r = raggio.to("m").magnitude
    p = (pressione_collasso or PhysicalConstants.PRESSIONE_ATMOSFERICA).to("Pa").magnitude
    rho = (densita or PhysicalConstants.DENSITA_ACQUA).to("kg/m^3").magnitude

    t_r = 0.915 * r * np.sqrt(rho / p)

    return Q_(t_r * 1e6, "us")
