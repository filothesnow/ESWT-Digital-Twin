# Calcoli impedenza acustica
"""
Modulo per i calcoli di impedenza acustica e trasmissione/riflessione.

L'impedenza acustica è fondamentale per capire come le onde d'urto
interagiscono con le interfacce tra tessuti diversi.

Equazioni principali:
    - Impedenza: Z = ρ × c
    - Coefficiente riflessione: R = (Z2-Z1)/(Z2+Z1)
    - Coefficiente trasmissione: T = 2*Z2/(Z2+Z1)
    - Intensità riflessa: I_R/I = R²
    - Intensità trasmessa: I_T/I = 4*Z1*Z2/(Z1+Z2)²

Riferimenti:
    - Ogden et al. (2001) - Table 3: Acoustic impedances
"""

from typing import Optional, Tuple
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants, AcousticImpedance


def calcola_impedenza(
    densita: "Q_",
    velocita_suono: "Q_",
) -> "Q_":
    """
    Calcola l'impedenza acustica di un mezzo.

    Z = ρ × c

    Parametri:
        densita: Densità del mezzo (kg/m³ o g/cm³)
        velocita_suono: Velocità del suono nel mezzo (m/s)

    Ritorna:
        Impedenza acustica (kg/(m²·s) = rayl)
    """
    rho = densita.to("kg/m^3").magnitude
    c = velocita_suono.to("m/s").magnitude

    z = rho * c
    return Q_(z, "kg/(m^2*s)")


def impedenza_acqua() -> "Q_":
    """Ritorna l'impedenza acustica dell'acqua."""
    return calcola_impedenza(
        PhysicalConstants.DENSITA_ACQUA,
        PhysicalConstants.VELOCITA_SUONO_ACQUA,
    )


def impedenza_tessuto(tipo: str = "muscolo") -> "Q_":
    """
    Ritorna l'impedenza acustica di un tessuto biologico.

    Parametri:
        tipo: Tipo di tessuto ("muscolo", "grasso", "osso_corticale",
              "osso_spongioso", "polmone", "fegato")

    Ritorna:
        Impedenza acustica del tessuto
    """
    # Dati da Ogden et al. (2001) Table 3 e letteratura
    tessuti = {
        "muscolo": {"rho": 1060, "c": 1630},  # kg/m³, m/s
        "grasso": {"rho": 920, "c": 1450},
        "osso_corticale": {"rho": 1800, "c": 4100},
        "osso_spongioso": {"rho": 1100, "c": 2300},
        "polmone": {"rho": 300, "c": 650},
        "fegato": {"rho": 1060, "c": 1590},
        "rene": {"rho": 1050, "c": 1560},
        "pelle": {"rho": 1100, "c": 1730},
    }

    if tipo not in tessuti:
        raise ValueError(f"Tessuto '{tipo}' non riconosciuto. Disponibili: {list(tessuti.keys())}")

    props = tessuti[tipo]
    return calcola_impedenza(Q_(props["rho"], "kg/m^3"), Q_(props["c"], "m/s"))


def coefficiente_riflessione(
    z1: "Q_",
    z2: "Q_",
) -> float:
    """
    Calcola il coefficiente di riflessione in ampiezza.

    R = (Z2 - Z1) / (Z2 + Z1)

    Un valore positivo indica riflessione senza inversione di fase,
    negativo indica inversione di fase.

    Parametri:
        z1: Impedenza del primo mezzo (incidente)
        z2: Impedenza del secondo mezzo (trasmesso)

    Ritorna:
        Coefficiente di riflessione (-1 a +1)
    """
    z1_val = z1.to("kg/(m^2*s)").magnitude
    z2_val = z2.to("kg/(m^2*s)").magnitude

    return (z2_val - z1_val) / (z2_val + z1_val)


def coefficiente_trasmissione(
    z1: "Q_",
    z2: "Q_",
) -> float:
    """
    Calcola il coefficiente di trasmissione in ampiezza.

    T = 2*Z2 / (Z2 + Z1)

    Parametri:
        z1: Impedenza del primo mezzo
        z2: Impedenza del secondo mezzo

    Ritorna:
        Coefficiente di trasmissione (0 a 2)
    """
    z1_val = z1.to("kg/(m^2*s)").magnitude
    z2_val = z2.to("kg/(m^2*s)").magnitude

    return 2 * z2_val / (z2_val + z1_val)


def intensita_riflessa_relativa(
    z1: "Q_",
    z2: "Q_",
) -> float:
    """
    Calcola la frazione di intensità riflessa.

    I_R/I = R² = ((Z2-Z1)/(Z2+Z1))²

    Parametri:
        z1: Impedenza del primo mezzo
        z2: Impedenza del secondo mezzo

    Ritorna:
        Frazione di intensità riflessa (0 a 1)
    """
    R = coefficiente_riflessione(z1, z2)
    return R**2


def intensita_trasmessa_relativa(
    z1: "Q_",
    z2: "Q_",
) -> float:
    """
    Calcola la frazione di intensità trasmessa.

    I_T/I = 4*Z1*Z2 / (Z1+Z2)²

    Parametri:
        z1: Impedenza del primo mezzo
        z2: Impedenza del secondo mezzo

    Ritorna:
        Frazione di intensità trasmessa (0 a 1)
    """
    z1_val = z1.to("kg/(m^2*s)").magnitude
    z2_val = z2.to("kg/(m^2*s)").magnitude

    return 4 * z1_val * z2_val / (z1_val + z2_val) ** 2


def pressione_trasmessa(
    pressione_incidente: "Q_",
    z1: "Q_",
    z2: "Q_",
) -> "Q_":
    """
    Calcola la pressione trasmessa attraverso un'interfaccia.

    P_T = T × P_I = 2*Z2/(Z1+Z2) × P_I

    Parametri:
        pressione_incidente: Pressione dell'onda incidente
        z1: Impedenza del primo mezzo
        z2: Impedenza del secondo mezzo

    Ritorna:
        Pressione trasmessa
    """
    p_in = pressione_incidente.to("Pa").magnitude
    T = coefficiente_trasmissione(z1, z2)
    return Q_(T * p_in, "Pa")


def pressione_riflessa(
    pressione_incidente: "Q_",
    z1: "Q_",
    z2: "Q_",
) -> "Q_":
    """
    Calcola la pressione riflessa da un'interfaccia.

    P_R = R × P_I

    Parametri:
        pressione_incidente: Pressione dell'onda incidente
        z1: Impedenza del primo mezzo
        z2: Impedenza del secondo mezzo

    Ritorna:
        Pressione riflessa (può essere negativa se R < 0)
    """
    p_in = pressione_incidente.to("Pa").magnitude
    R = coefficiente_riflessione(z1, z2)
    return Q_(R * p_in, "Pa")


def calcola_trasmissione_multistrato(
    pressione_iniziale: "Q_",
    impedenze: list,
) -> Tuple["Q_", list]:
    """
    Calcola la trasmissione attraverso più strati di tessuto.

    Parametri:
        pressione_iniziale: Pressione incidente sul primo strato
        impedenze: Lista di impedenze dei vari strati [Z1, Z2, Z3, ...]

    Ritorna:
        Tuple (pressione_finale, lista_pressioni_intermedie)
    """
    if len(impedenze) < 2:
        return pressione_iniziale, [pressione_iniziale]

    pressioni = [pressione_iniziale.to("Pa").magnitude]
    p_corrente = pressione_iniziale.to("Pa").magnitude

    for i in range(len(impedenze) - 1):
        T = coefficiente_trasmissione(impedenze[i], impedenze[i + 1])
        p_corrente = T * p_corrente
        pressioni.append(p_corrente)

    return Q_(p_corrente, "Pa"), [Q_(p, "Pa") for p in pressioni]


def analisi_interfaccia_acqua_tessuto(tessuto: str) -> dict:
    """
    Analizza la trasmissione/riflessione all'interfaccia acqua-tessuto.

    Parametri:
        tessuto: Tipo di tessuto

    Ritorna:
        Dizionario con analisi completa dell'interfaccia
    """
    z_acqua = impedenza_acqua()
    z_tessuto = impedenza_tessuto(tessuto)

    R = coefficiente_riflessione(z_acqua, z_tessuto)
    T = coefficiente_trasmissione(z_acqua, z_tessuto)

    return {
        "tessuto": tessuto,
        "impedenza_acqua_Mrayl": z_acqua.to("kg/(m^2*s)").magnitude / 1e6,
        "impedenza_tessuto_Mrayl": z_tessuto.to("kg/(m^2*s)").magnitude / 1e6,
        "coefficiente_riflessione": R,
        "coefficiente_trasmissione": T,
        "intensita_riflessa_percentuale": intensita_riflessa_relativa(z_acqua, z_tessuto) * 100,
        "intensita_trasmessa_percentuale": intensita_trasmessa_relativa(z_acqua, z_tessuto) * 100,
    }


def tabella_impedenze_tessuti() -> dict:
    """
    Ritorna una tabella con le impedenze di tutti i tessuti disponibili.

    Ritorna:
        Dizionario {tessuto: impedenza_Mrayl}
    """
    tessuti = [
        "muscolo",
        "grasso",
        "osso_corticale",
        "osso_spongioso",
        "polmone",
        "fegato",
        "rene",
        "pelle",
    ]

    tabella = {"acqua": impedenza_acqua().to("kg/(m^2*s)").magnitude / 1e6}

    for t in tessuti:
        tabella[t] = impedenza_tessuto(t).to("kg/(m^2*s)").magnitude / 1e6

    return tabella
