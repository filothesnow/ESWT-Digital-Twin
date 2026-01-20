# Focalizzazione e calcolo pressione focale
"""
Modulo per il calcolo della focalizzazione e pressione nella zona focale.

Implementa:
    - Calcolo dimensioni zona focale
    - Pressione di picco al fuoco
    - Energy flux density (EFD / PII)

Riferimenti:
    - Ogden et al. (2001) - Table 1 dati OssaTron
    - Slezak et al. (2022) - EFD e parametri clinici
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants, AcousticImpedance
from .reflector import BaseReflector, EllipticalReflector, ParabolicReflector


@dataclass
class FocalZone:
    """
    Caratteristiche della zona focale.

    Attributi:
        diametro_laterale_6dB: Diametro laterale a -6dB (mm)
        lunghezza_assiale_6dB: Lunghezza assiale a -6dB (mm)
        volume_6dB: Volume della zona focale a -6dB (mm³)
        posizione_fuoco: Posizione del centro focale (mm)
    """

    diametro_laterale_6dB: "Q_"
    lunghezza_assiale_6dB: "Q_"
    volume_6dB: "Q_"
    posizione_fuoco: Tuple["Q_", "Q_", "Q_"]

    @property
    def area_sezione_focale(self) -> "Q_":
        """Area della sezione trasversale al fuoco."""
        r = self.diametro_laterale_6dB.to("mm").magnitude / 2
        return Q_(np.pi * r**2, "mm^2")


@dataclass
class FocusingResult:
    """
    Risultato completo della simulazione di focalizzazione.

    Attributi:
        pressione_picco: Pressione di picco al fuoco (MPa)
        pressione_negativa: Pressione tensile di picco (MPa)
        energy_flux_density: EFD al fuoco (mJ/mm²)
        zona_focale: Caratteristiche della zona focale
        efficienza_focalizzazione: Rapporto energia_focale/energia_sorgente
    """

    pressione_picco: "Q_"
    pressione_negativa: "Q_"
    energy_flux_density: "Q_"
    zona_focale: FocalZone
    efficienza_focalizzazione: float


def calcola_zona_focale(riflettore: BaseReflector) -> FocalZone:
    """
    Calcola le caratteristiche della zona focale per un riflettore.

    Parametri:
        riflettore: Istanza di riflettore (ellittico o parabolico)

    Ritorna:
        FocalZone con le dimensioni della zona focale
    """
    # Ottieni dimensioni 6dB dal riflettore
    d_lat, l_ax = riflettore.calcola_zona_focale_6dB()

    d = d_lat.to("mm").magnitude
    l = l_ax.to("mm").magnitude

    # Volume zona focale (approssimazione ellissoide prolato)
    # V = (4/3) * π * r_lat² * (l_ax/2)
    r = d / 2
    volume = (4 / 3) * np.pi * r**2 * (l / 2)

    # Posizione fuoco
    pos_fuoco = riflettore.posizione_fuoco_secondario()

    return FocalZone(
        diametro_laterale_6dB=d_lat,
        lunghezza_assiale_6dB=l_ax,
        volume_6dB=Q_(volume, "mm^3"),
        posizione_fuoco=pos_fuoco,
    )


def calcola_pressione_focale(
    pressione_sorgente: "Q_",
    riflettore: BaseReflector,
    perdite_riflessione: float = 0.95,
    perdite_propagazione: float = 0.90,
) -> "Q_":
    """
    Calcola la pressione di picco al fuoco.

    La pressione focale dipende da:
    - Pressione alla sorgente
    - Guadagno geometrico del riflettore
    - Perdite per riflessione e propagazione

    Parametri:
        pressione_sorgente: Pressione generata dalla scarica (MPa)
        riflettore: Istanza di riflettore
        perdite_riflessione: Efficienza riflessione (0-1)
        perdite_propagazione: Efficienza propagazione (0-1)

    Ritorna:
        Pressione di picco al fuoco (MPa)
    """
    p_src = pressione_sorgente.to("MPa").magnitude

    # Guadagno geometrico
    G = riflettore.guadagno_geometrico()

    # Pressione focale (in ampiezza, non intensità)
    # P_focale = P_sorgente * sqrt(G) * η_riflessione * η_propagazione
    p_focale = p_src * np.sqrt(G) * perdite_riflessione * perdite_propagazione

    # Limita a valori realistici (max ~100 MPa per ESWT)
    p_focale = min(p_focale, 150)

    return Q_(p_focale, "MPa")


def calcola_pressione_focale_completa(
    energia_scarica: "Q_",
    riflettore: BaseReflector,
    gap_elettrodi: "Q_" = None,
    efficienza_conversione: float = 0.3,
) -> FocusingResult:
    """
    Calcola i parametri focali completi da energia di scarica.

    Modello integrato che tiene conto di:
    - Conversione energia → pressione sorgente (Chen)
    - Geometria riflettore
    - Attenuazione e perdite

    Parametri:
        energia_scarica: Energia della scarica (J)
        riflettore: Istanza di riflettore
        gap_elettrodi: Distanza inter-elettrodo (default: 5 mm)
        efficienza_conversione: Frazione energia → acustica

    Ritorna:
        FocusingResult con tutti i parametri focali
    """
    from ..power_electronics.energy import calcola_pressione_picco_chen

    if gap_elettrodi is None:
        gap_elettrodi = Q_(5, "mm")

    # Calcola pressione alla sorgente (vicino al gap)
    distanza_sorgente = gap_elettrodi.to("cm")
    p_sorgente = calcola_pressione_picco_chen(energia_scarica, distanza_sorgente)

    # Calcola zona focale
    zona_focale = calcola_zona_focale(riflettore)

    # Calcola pressione al fuoco
    p_focale = calcola_pressione_focale(p_sorgente, riflettore)

    # Pressione negativa (tipicamente 10-20% della positiva)
    p_neg = Q_(-0.15 * p_focale.to("MPa").magnitude, "MPa")

    # Energy flux density
    efd = calcola_energy_flux_density(
        p_focale, durata_impulso=Q_(5, "us"), area_focale=zona_focale.area_sezione_focale
    )

    # Efficienza focalizzazione
    # Rapporto tra energia nella zona focale e energia totale
    E_tot = energia_scarica.to("J").magnitude
    E_focale = efd.to("mJ/mm^2").magnitude * zona_focale.area_sezione_focale.to("mm^2").magnitude
    efficienza = min(E_focale / (E_tot * 1000), 1.0)  # mJ/J

    return FocusingResult(
        pressione_picco=p_focale,
        pressione_negativa=p_neg,
        energy_flux_density=efd,
        zona_focale=zona_focale,
        efficienza_focalizzazione=efficienza,
    )


def calcola_energy_flux_density(
    pressione_picco: "Q_",
    durata_impulso: "Q_" = None,
    area_focale: "Q_" = None,
) -> "Q_":
    """
    Calcola l'Energy Flux Density (EFD) / Pulse Integral Intensity (PII).

    EFD = (1/ρc) * ∫ p²(t) dt

    Per un impulso approssimato come triangolare:
    EFD ≈ p_max² * τ / (3 * ρ * c)

    Parametri:
        pressione_picco: Pressione di picco (MPa)
        durata_impulso: Durata dell'impulso (default: 5 μs)
        area_focale: Area della zona focale (per normalizzazione)

    Ritorna:
        EFD in mJ/mm²
    """
    p = pressione_picco.to("Pa").magnitude  # Pa

    if durata_impulso is None:
        durata_impulso = Q_(5, "us")
    tau = durata_impulso.to("s").magnitude

    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    # EFD per impulso approssimato
    # Integrale di p² per impulso triangolare: ∫p²dt ≈ p_max² * τ / 3
    efd_j_m2 = (p**2 * tau / 3) / (rho * c)

    # Converti a mJ/mm²
    # 1 J/m² = 1e-3 mJ / 1e6 mm² = 1e-9 mJ/mm²
    # Quindi: mJ/mm² = J/m² * 1e-3 (da J a mJ) / 1e-6 (da m² a mm²) = J/m² * 1e3
    # Correzione: 1 J/m² = 1e-3 J/mm² = 1 mJ/mm² * 1e-3 → no
    # 1 m² = 1e6 mm², quindi 1 J/m² = 1e-6 J/mm² = 1e-3 mJ/mm²
    efd_mj_mm2 = efd_j_m2 * 1e-3

    return Q_(efd_mj_mm2, "mJ/mm^2")


def stima_pressione_da_efd(
    efd: "Q_",
    durata_impulso: "Q_" = None,
) -> "Q_":
    """
    Stima la pressione di picco dall'EFD (operazione inversa).

    Parametri:
        efd: Energy flux density (mJ/mm²)
        durata_impulso: Durata dell'impulso (default: 5 μs)

    Ritorna:
        Pressione di picco stimata (MPa)
    """
    efd_val = efd.to("mJ/mm^2").magnitude
    # Converti a J/m²
    efd_j_m2 = efd_val * 1e3

    if durata_impulso is None:
        durata_impulso = Q_(5, "us")
    tau = durata_impulso.to("s").magnitude

    rho = PhysicalConstants.DENSITA_ACQUA.to("kg/m^3").magnitude
    c = PhysicalConstants.VELOCITA_SUONO_ACQUA.to("m/s").magnitude

    # p² = EFD * ρ * c * 3 / τ
    p_squared = efd_j_m2 * rho * c * 3 / tau
    p_pa = np.sqrt(max(p_squared, 0))

    return Q_(p_pa / 1e6, "MPa")


def calcola_profilo_pressione_assiale(
    riflettore: BaseReflector,
    pressione_focale: "Q_",
    n_punti: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola il profilo di pressione lungo l'asse del riflettore.

    Parametri:
        riflettore: Istanza di riflettore
        pressione_focale: Pressione di picco al fuoco
        n_punti: Numero di punti

    Ritorna:
        Tuple (posizioni_z_mm, pressioni_MPa)
    """
    # Ottieni posizione fuoco e dimensioni zona focale
    _, _, z_fuoco = riflettore.posizione_fuoco_secondario()
    z_f = z_fuoco.to("mm").magnitude

    _, l_ax = riflettore.calcola_zona_focale_6dB()
    l = l_ax.to("mm").magnitude

    # Range di posizioni
    z_min = z_f - l
    z_max = z_f + l
    z = np.linspace(z_min, z_max, n_punti)

    # Profilo gaussiano centrato sul fuoco
    p_max = pressione_focale.to("MPa").magnitude
    sigma = l / (2 * np.sqrt(2 * np.log(2)))  # sigma da FWHM
    pressione = p_max * np.exp(-((z - z_f) ** 2) / (2 * sigma**2))

    return z, pressione


def calcola_profilo_pressione_laterale(
    riflettore: BaseReflector,
    pressione_focale: "Q_",
    n_punti: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola il profilo di pressione trasversale al fuoco.

    Parametri:
        riflettore: Istanza di riflettore
        pressione_focale: Pressione di picco al fuoco
        n_punti: Numero di punti

    Ritorna:
        Tuple (posizioni_r_mm, pressioni_MPa)
    """
    d_lat, _ = riflettore.calcola_zona_focale_6dB()
    d = d_lat.to("mm").magnitude

    # Range di posizioni
    r_max = d * 1.5
    r = np.linspace(-r_max, r_max, n_punti)

    # Profilo gaussiano
    p_max = pressione_focale.to("MPa").magnitude
    sigma = d / (2 * 2 * np.sqrt(2 * np.log(2)))  # sigma da diametro 6dB
    pressione = p_max * np.exp(-(r**2) / (2 * sigma**2))

    return r, pressione


def confronta_con_ogden(
    risultato: FocusingResult,
    tensione_kV: float,
) -> dict:
    """
    Confronta i risultati con i dati di riferimento Ogden et al. (2001).

    Parametri:
        risultato: FocusingResult dalla simulazione
        tensione_kV: Tensione di scarica (14, 20, o 28 kV)

    Ritorna:
        Dizionario con confronto simulazione vs riferimento
    """
    # Dati OssaTron da Ogden Table 1
    ogden_data = {
        14: {"p_max": 40.6, "efd": 0.105, "d_lat": 6.8, "l_ax": 44.1},
        20: {"p_max": 45.6, "efd": 0.255, "d_lat": 6.4, "l_ax": 59.0},
        28: {"p_max": 71.9, "efd": 0.37, "d_lat": 8.7, "l_ax": 67.6},
    }

    # Trova il più vicino
    tensioni = list(ogden_data.keys())
    tensione_ref = min(tensioni, key=lambda x: abs(x - tensione_kV))
    ref = ogden_data[tensione_ref]

    # Valori simulati
    p_sim = risultato.pressione_picco.to("MPa").magnitude
    efd_sim = risultato.energy_flux_density.to("mJ/mm^2").magnitude
    d_sim = risultato.zona_focale.diametro_laterale_6dB.to("mm").magnitude
    l_sim = risultato.zona_focale.lunghezza_assiale_6dB.to("mm").magnitude

    return {
        "tensione_riferimento_kV": tensione_ref,
        "pressione": {
            "simulata_MPa": p_sim,
            "riferimento_MPa": ref["p_max"],
            "errore_percentuale": 100 * (p_sim - ref["p_max"]) / ref["p_max"],
        },
        "efd": {
            "simulata_mJ_mm2": efd_sim,
            "riferimento_mJ_mm2": ref["efd"],
            "errore_percentuale": 100 * (efd_sim - ref["efd"]) / ref["efd"],
        },
        "zona_focale_laterale": {
            "simulata_mm": d_sim,
            "riferimento_mm": ref["d_lat"],
            "errore_percentuale": 100 * (d_sim - ref["d_lat"]) / ref["d_lat"],
        },
        "zona_focale_assiale": {
            "simulata_mm": l_sim,
            "riferimento_mm": ref["l_ax"],
            "errore_percentuale": 100 * (l_sim - ref["l_ax"]) / ref["l_ax"],
        },
    }
