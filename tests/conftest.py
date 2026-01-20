# Configurazione pytest e fixture comuni
"""
Fixture e configurazione per i test ESWT Digital Twin.
"""

import pytest
import sys
from pathlib import Path

# Aggiungi la directory src al path per gli import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def unita():
    """Fixture per il registry delle unità Pint."""
    from src.core.units import ureg, Q_

    return ureg, Q_


@pytest.fixture
def condensatore_standard(unita):
    """
    Fixture per un condensatore standard ESWT.

    C = 1 μF, V = 20 kV → E = 200 J
    """
    _, Q_ = unita
    from src.modules.power_electronics import Capacitor

    return Capacitor(Q_(1, "uF"), Q_(20, "kV"))


@pytest.fixture
def plasma_standard(unita):
    """
    Fixture per un canale di plasma standard.

    Gap = 5 mm, modello Rompe-Weizel
    """
    _, Q_ = unita
    from src.modules.power_electronics import PlasmaChannel

    return PlasmaChannel(gap=Q_(5, "mm"))


@pytest.fixture
def plasma_costante(unita):
    """
    Fixture per un canale di plasma a resistenza costante.

    R = 0.5 Ω (per test con soluzioni analitiche)
    """
    _, Q_ = unita
    from src.modules.power_electronics import crea_modello_costante

    return crea_modello_costante(Q_(0.5, "ohm"))


@pytest.fixture
def simulatore_standard(condensatore_standard, plasma_costante, unita):
    """
    Fixture per un simulatore di scarica standard.

    C = 1 μF, V = 20 kV, L = 5 μH, R = 0.5 Ω
    """
    _, Q_ = unita
    from src.modules.power_electronics import DischargeSimulator

    return DischargeSimulator(condensatore_standard, plasma_costante, Q_(5, "uH"))


@pytest.fixture
def dati_chen_2010():
    """
    Fixture con i dati sperimentali di Chen et al. (2010).

    Ritorna lista di dizionari con i parametri dei test.
    """
    return [
        {"test": 1, "energia_J": 3300, "distanza_cm": 17.5, "gap_mm": 10, "pressione_MPa": 8.0},
        {"test": 2, "energia_J": 600, "distanza_cm": 17.5, "gap_mm": 5, "pressione_MPa": 4.4},
        {"test": 3, "energia_J": 31, "distanza_cm": 9, "gap_mm": 5, "pressione_MPa": 2.8},
        {"test": 4, "energia_J": 20, "distanza_cm": 9, "gap_mm": 5, "pressione_MPa": 2.0},
    ]


@pytest.fixture
def dati_ogden_2001():
    """
    Fixture con i dati dell'OssaTron da Ogden et al. (2001).

    Ritorna dizionario con parametri per diversi livelli di tensione.
    """
    return {
        "14kV": {
            "max_pressure_MPa": 40.6,
            "efd_mJ_mm2": 0.09,
            "6dB_lateral_mm": 6.8,
            "6dB_axial_mm": 44.1,
        },
        "20kV": {
            "max_pressure_MPa": 45.6,
            "efd_mJ_mm2": 0.24,
            "6dB_lateral_mm": 6.4,
            "6dB_axial_mm": 59.0,
        },
        "28kV": {
            "max_pressure_MPa": 71.9,
            "efd_mJ_mm2": 0.34,
            "6dB_lateral_mm": 8.7,
            "6dB_axial_mm": 67.6,
        },
    }
