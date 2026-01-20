# Modulo Physics Engine - Propagazione onde d'urto e focalizzazione
"""
Simulazione della fisica delle onde d'urto per dispositivi ESWT.

Questo modulo implementa:
    - Formazione del plasma e espansione bolla
    - Propagazione dell'onda d'urto (lineare e non lineare)
    - Geometria del riflettore (ellisse e parabola)
    - Focalizzazione e calcolo pressione focale
    - Calcoli di impedenza acustica
    - Effetti di cavitazione
    - Simulazione integrata

Equazioni principali (da Ogden et al. 2001, Chen et al. 2010):
    - Rayleigh-Plesset: R·R'' + (3/2)·R'² = (P_p - P_amb) / ρ
    - Burgers (weak shock): P(r) = P₀·(r₀/r)·exp(-αr) / [1 + β·P₀·r/(2ρc₀³)]
    - Attenuazione lineare: P(r) = P_0 * (r_0/r)^n * exp(-α*r)
    - Z = ρ × c (impedenza acustica)
    - PII = (1/ρc) ∫ p²(t) dt (energy flux density)

Moduli:
    - plasma_dynamics: Formazione plasma, espansione bolla
    - shockwave: Propagazione onda d'urto (lineare e non lineare)
    - reflector: Geometria riflettore (ellisse, parabola)
    - focusing: Focalizzazione
    - impedance: Calcoli impedenza acustica
    - cavitation: Effetti di cavitazione
    - simulation: Integrazione completa
"""

# Plasma dynamics
from .plasma_dynamics import (
    PlasmaState,
    PlasmaExpansionResult,
    BubbleExpansionResult,
    calcola_pressione_plasma,
    calcola_temperatura_plasma,
    calcola_espansione_da_energia,
    simula_espansione_bolla,
    tempo_rayleigh,
)

# Shockwave - base
from .shockwave import (
    ShockwaveState,
    ShockwavePulse,
    propaga_onda_sferica,
    calcola_attenuazione,
    calcola_rise_time,
    genera_profilo_impulso,
    calcola_velocita_onda,
)

# Shockwave - propagazione non lineare
from .shockwave import (
    propaga_onda_nonlineare,
    propaga_onda_nonlineare_array,
    calcola_rise_time_calibrato,
    calcola_distanza_formazione_shock,
    calcola_attenuazione_shock,
    genera_profilo_impulso_calibrato,
    OGDEN_DATA,
)

# Reflector
from .reflector import (
    ReflectorType,
    ReflectorGeometry,
    BaseReflector,
    EllipticalReflector,
    ParabolicReflector,
    crea_riflettore,
)

# Focusing
from .focusing import (
    FocalZone,
    FocusingResult,
    calcola_zona_focale,
    calcola_pressione_focale,
    calcola_pressione_focale_completa,
    calcola_energy_flux_density,
    confronta_con_ogden,
    calcola_profilo_pressione_assiale,
    calcola_profilo_pressione_laterale,
)

# Impedance
from .impedance import (
    calcola_impedenza,
    coefficiente_riflessione,
    coefficiente_trasmissione,
    pressione_trasmessa,
    pressione_riflessa,
    calcola_trasmissione_multistrato,
)

# Cavitation
from .cavitation import (
    CavitationBubble,
    CavitationResult,
    verifica_cavitazione,
    calcola_raggio_bolla_max,
    stima_energia_collasso,
    calcola_tempo_rayleigh_collasso,
    analizza_cavitazione,
    TENSIONE_ROTTURA_ACQUA,
)

# Simulation
from .simulation import (
    ShockwaveSimulationConfig,
    ShockwaveSimulationResult,
    simula_impulso_completo,
    simula_serie_impulsi,
    crea_riflettore_ossatron,
    valida_simulazione_vs_ogden,
    report_simulazione,
)

__all__ = [
    # Plasma dynamics
    "PlasmaState",
    "PlasmaExpansionResult",
    "BubbleExpansionResult",
    "calcola_pressione_plasma",
    "calcola_temperatura_plasma",
    "calcola_espansione_da_energia",
    "simula_espansione_bolla",
    "tempo_rayleigh",
    # Shockwave - base
    "ShockwaveState",
    "ShockwavePulse",
    "propaga_onda_sferica",
    "calcola_attenuazione",
    "calcola_rise_time",
    "genera_profilo_impulso",
    "calcola_velocita_onda",
    # Shockwave - non lineare
    "propaga_onda_nonlineare",
    "propaga_onda_nonlineare_array",
    "calcola_rise_time_calibrato",
    "calcola_distanza_formazione_shock",
    "calcola_attenuazione_shock",
    "genera_profilo_impulso_calibrato",
    "OGDEN_DATA",
    # Reflector
    "ReflectorType",
    "ReflectorGeometry",
    "BaseReflector",
    "EllipticalReflector",
    "ParabolicReflector",
    "crea_riflettore",
    # Focusing
    "FocalZone",
    "FocusingResult",
    "calcola_zona_focale",
    "calcola_pressione_focale",
    "calcola_pressione_focale_completa",
    "calcola_energy_flux_density",
    "confronta_con_ogden",
    "calcola_profilo_pressione_assiale",
    "calcola_profilo_pressione_laterale",
    # Impedance
    "calcola_impedenza",
    "coefficiente_riflessione",
    "coefficiente_trasmissione",
    "pressione_trasmessa",
    "pressione_riflessa",
    "calcola_trasmissione_multistrato",
    # Cavitation
    "CavitationBubble",
    "CavitationResult",
    "verifica_cavitazione",
    "calcola_raggio_bolla_max",
    "stima_energia_collasso",
    "calcola_tempo_rayleigh_collasso",
    "analizza_cavitazione",
    "TENSIONE_ROTTURA_ACQUA",
    # Simulation
    "ShockwaveSimulationConfig",
    "ShockwaveSimulationResult",
    "simula_impulso_completo",
    "simula_serie_impulsi",
    "crea_riflettore_ossatron",
    "valida_simulazione_vs_ogden",
    "report_simulazione",
]
