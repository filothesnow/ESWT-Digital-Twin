# Modulo Degradation - Erosione elettrodi e degradazione componenti
"""
Simulazione della degradazione componenti per dispositivi ESWT.

Questo modulo implementa:
    - Erosione degli elettrodi
    - Variazione del gap inter-elettrodo
    - Efficienza vs numero di impulsi
    - Chimica dell'acqua (detriti, conducibilità)
    - Proprietà fisiche dell'acqua
    - Assorbimento gas da palladio

Equazioni principali:
    - Tasso erosione: dm/dt = k * I^n * f(T)
    - Gap variabile: d(gap)/dt = (dm/dt) / (ρ * A_tip)
    - Efficienza: η(N) = η_0 * exp(-λ*N)
    - Conducibilità: σ = σ₀ + k_σ * c_detriti
    - Breakdown: V_bd = f(gap, σ, T)

Moduli:
    - electrode: Modello erosione elettrodo
    - efficiency: Calcolo efficienza vs impulsi
    - water_chemistry: Chimica acqua e conducibilità
    - water_properties: Proprietà fisiche acqua
    - gas_absorption: Assorbimento gas da palladio

Esempio d'uso:
    >>> from modules.degradation import (
    ...     ElectrodeState, crea_stato_elettrodo,
    ...     WaterChemistryModel, WaterPropertiesModel
    ... )
    >>> stato = crea_stato_elettrodo("tungsteno", gap_iniziale=Q_(5, "mm"))
    >>> water = WaterChemistryModel()
"""

from .electrode import (
    ElectrodeState,
    ElectrodeErosionModel,
    calcola_erosione_impulso,
    calcola_vita_residua,
    simula_degradazione,
    crea_stato_elettrodo,
)

from .efficiency import (
    EfficiencyModel,
    calcola_efficienza,
    stima_impulsi_rimanenti,
    genera_curva_efficienza,
)

from .water_chemistry import (
    WaterChemistryState,
    WaterChemistryConfig,
    WaterChemistryModel,
)

from .water_properties import (
    WaterPropertiesState,
    WaterPropertiesModel,
)

from .gas_absorption import (
    GasAbsorptionState,
    GasAbsorptionConfig,
    GasAbsorptionModel,
)

__all__ = [
    # Electrode
    "ElectrodeState",
    "ElectrodeErosionModel",
    "calcola_erosione_impulso",
    "calcola_vita_residua",
    "simula_degradazione",
    "crea_stato_elettrodo",
    # Efficiency
    "EfficiencyModel",
    "calcola_efficienza",
    "stima_impulsi_rimanenti",
    "genera_curva_efficienza",
    # Water Chemistry
    "WaterChemistryState",
    "WaterChemistryConfig",
    "WaterChemistryModel",
    # Water Properties
    "WaterPropertiesState",
    "WaterPropertiesModel",
    # Gas Absorption
    "GasAbsorptionState",
    "GasAbsorptionConfig",
    "GasAbsorptionModel",
]
