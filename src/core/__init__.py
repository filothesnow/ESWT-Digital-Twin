# Modulo core - Costanti fisiche, unità di misura, proprietà materiali
"""
Modulo core del simulatore ESWT.

Contiene:
    - units: Sistema di unità di misura basato su Pint
    - constants: Costanti fisiche fondamentali
    - materials: Proprietà dei materiali (elettrodi, acqua, etc.)
"""

from .units import ureg, Q_
from .constants import (
    PhysicalConstants,
    ESWTParameters,
    AcousticImpedance,
    ElectrodeProperties,
    ChenModelParameters,
    calcola_impedenza_acustica,
    coefficiente_riflessione,
    coefficiente_trasmissione,
)
from .materials import MaterialsDatabase, ElectrodeMaterial, WaterSolution

__all__ = [
    "ureg",
    "Q_",
    "PhysicalConstants",
    "ESWTParameters",
    "AcousticImpedance",
    "ElectrodeProperties",
    "ChenModelParameters",
    "calcola_impedenza_acustica",
    "coefficiente_riflessione",
    "coefficiente_trasmissione",
    "MaterialsDatabase",
    "ElectrodeMaterial",
    "WaterSolution",
]
