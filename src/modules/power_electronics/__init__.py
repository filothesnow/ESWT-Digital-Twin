# Modulo Power Electronics
"""
Simulazione dell'elettronica di potenza per dispositivi ESWT.

Questo modulo implementa:
    - Modello di scarica del condensatore
    - Resistenza dinamica del canale di plasma
    - Calcolo dell'energia depositata
    - Simulazione del circuito RLC con plasma

Equazioni principali (da Chen et al. 2010):
    - E(t) = 0.5 * C * (U_M² - U(t)²)  # Energia iniettata
    - P_0 = k * E_B^α                   # Pressione picco empirica

Moduli:
    - capacitor: Modello condensatore
    - plasma_channel: Resistenza plasma R(t)
    - discharge: Solver ODE per scarica
    - energy: Calcolo energia plasma
"""

from .capacitor import Capacitor, CapacitorState, calcola_capacita_da_energia
from .plasma_channel import (
    PlasmaChannel,
    PlasmaModel,
    PlasmaState,
    crea_modello_rompe_weizel,
    crea_modello_costante,
)
from .discharge import DischargeSimulator, DischargeResult, simula_scarica_rlc_analitica
from .energy import (
    EnergyBreakdown,
    calcola_energia_condensatore,
    calcola_energia_rilasciata,
    calcola_pressione_picco_chen,
    calcola_ripartizione_energia,
    calcola_pulse_integral_intensity,
)

__all__ = [
    # Capacitor
    "Capacitor",
    "CapacitorState",
    "calcola_capacita_da_energia",
    # Plasma
    "PlasmaChannel",
    "PlasmaModel",
    "PlasmaState",
    "crea_modello_rompe_weizel",
    "crea_modello_costante",
    # Discharge
    "DischargeSimulator",
    "DischargeResult",
    "simula_scarica_rlc_analitica",
    # Energy
    "EnergyBreakdown",
    "calcola_energia_condensatore",
    "calcola_energia_rilasciata",
    "calcola_pressione_picco_chen",
    "calcola_ripartizione_energia",
    "calcola_pulse_integral_intensity",
]
