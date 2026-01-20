# Simulazione integrata onde d'urto ESWT
"""
Modulo per la simulazione completa del sistema ESWT.

Integra tutti i componenti del Physics Engine per fornire una
pipeline completa:

    Power Electronics → Plasma → Propagazione → Focalizzazione → Cavitazione

Questo modulo collega:
    - plasma_dynamics: Formazione del plasma e espansione bolla
    - shockwave: Propagazione non lineare
    - reflector: Geometria del riflettore
    - focusing: Focalizzazione e calcolo EFD
    - cavitation: Effetti di cavitazione
    - impedance: Trasmissione nei tessuti

Riferimenti:
    - Ogden et al. (2001) - Principles of Shock Wave Therapy
    - Chen et al. (2010) - PAED shock wave propagation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import PhysicalConstants, ChenModelParameters

# Import dai moduli del physics engine
from .plasma_dynamics import (
    calcola_espansione_da_energia,
    PlasmaExpansionResult,
)
from .shockwave import (
    propaga_onda_nonlineare,
    calcola_rise_time_calibrato,
    genera_profilo_impulso_calibrato,
    ShockwaveState,
    OGDEN_DATA,
)
from .reflector import EllipticalReflector, BaseReflector
from .focusing import (
    calcola_zona_focale,
    calcola_pressione_focale,
    calcola_energy_flux_density,
    confronta_con_ogden,
    FocalZone,
    FocusingResult,
)
from .cavitation import analizza_cavitazione, CavitationResult


@dataclass
class ShockwaveSimulationConfig:
    """
    Configurazione per la simulazione d'onda d'urto.

    Attributi:
        tensione_kv: Tensione del condensatore (kV)
        capacita_uf: Capacità del condensatore (μF)
        gap_elettrodi_mm: Distanza inter-elettrodo (mm)
        usa_propagazione_nonlineare: Abilita propagazione non lineare
        calcola_cavitazione: Abilita calcolo cavitazione
        beta_nonlinearita: Parametro β per propagazione non lineare
    """

    tensione_kv: float = 20.0
    capacita_uf: float = 1.0
    gap_elettrodi_mm: float = 5.0
    usa_propagazione_nonlineare: bool = True
    calcola_cavitazione: bool = True
    beta_nonlinearita: float = 3.5


@dataclass
class ShockwaveSimulationResult:
    """
    Risultato completo della simulazione d'onda d'urto.

    Attributi:
        # Sorgente
        energia_scarica: Energia rilasciata dal condensatore (J)
        pressione_sorgente: Pressione alla sorgente (MPa)

        # Plasma
        plasma_result: Risultato espansione plasma

        # Propagazione
        pressione_focale: Pressione al punto focale (MPa)
        pressione_negativa: Pressione negativa dell'impulso (MPa)
        rise_time: Tempo di salita (ns)

        # Zona focale
        zona_focale: Caratteristiche zona focale
        efd: Energy Flux Density (mJ/mm²)

        # Cavitazione
        cavitazione: Risultato analisi cavitazione

        # Validazione
        confronto_ogden: Dizionario con errori vs dati OssaTron
        configurazione: Configurazione usata per la simulazione
    """

    # Sorgente
    energia_scarica: "Q_"
    pressione_sorgente: "Q_"

    # Plasma
    plasma_result: PlasmaExpansionResult

    # Propagazione
    pressione_focale: "Q_"
    pressione_negativa: "Q_"
    rise_time: "Q_"

    # Zona focale
    zona_focale: FocalZone
    efd: "Q_"

    # Cavitazione
    cavitazione: CavitationResult

    # Validazione
    confronto_ogden: Dict[str, Any]
    configurazione: ShockwaveSimulationConfig


def calcola_energia_scarica(
    tensione_kv: float,
    capacita_uf: float,
) -> "Q_":
    """
    Calcola l'energia immagazzinata nel condensatore.

    E = 0.5 · C · V²

    Parametri:
        tensione_kv: Tensione (kV)
        capacita_uf: Capacità (μF)

    Ritorna:
        Energia (J)
    """
    c = capacita_uf * 1e-6  # μF → F
    v = tensione_kv * 1e3   # kV → V
    e = 0.5 * c * v**2
    return Q_(e, "J")


def simula_impulso_completo(
    riflettore: BaseReflector,
    config: ShockwaveSimulationConfig = None,
    energia_override: "Q_" = None,
) -> ShockwaveSimulationResult:
    """
    Esegue una simulazione completa di un impulso d'onda d'urto.

    Pipeline:
    1. Calcola energia dalla configurazione
    2. Modella espansione plasma (energia → pressione sorgente)
    3. Propaga onda (lineare o non lineare)
    4. Focalizza con riflettore
    5. Calcola EFD e zona focale
    6. Analizza cavitazione
    7. Confronta con dati Ogden

    Parametri:
        riflettore: Riflettore da utilizzare (EllipticalReflector)
        config: Configurazione simulazione (default: 20 kV, 1 μF)
        energia_override: Energia da usare invece di calcolarla (opzionale)

    Ritorna:
        ShockwaveSimulationResult con tutti i risultati
    """
    if config is None:
        config = ShockwaveSimulationConfig()

    # STEP 1: Energia
    if energia_override is not None:
        energia = energia_override
    else:
        energia = calcola_energia_scarica(config.tensione_kv, config.capacita_uf)

    # STEP 2: Espansione plasma
    gap = Q_(config.gap_elettrodi_mm, "mm")
    plasma_result = calcola_espansione_da_energia(energia, gap)
    pressione_sorgente = plasma_result.pressione_iniziale

    # STEP 3: Propagazione
    # Distanza dalla sorgente (F1) alla parete del riflettore
    # Approssimiamo con la metà della distanza focale
    distanza_riflettore = riflettore.distanza_focale() / 2

    if config.usa_propagazione_nonlineare:
        pressione_riflettore = propaga_onda_nonlineare(
            pressione_sorgente,
            gap,  # distanza iniziale ≈ gap
            distanza_riflettore,
            beta=config.beta_nonlinearita,
        )
    else:
        # Propagazione lineare sferica
        from .shockwave import calcola_attenuazione
        pressione_riflettore = calcola_attenuazione(
            pressione_sorgente,
            gap,
            distanza_riflettore,
        )

    # STEP 4: Focalizzazione
    pressione_focale = calcola_pressione_focale(
        pressione_riflettore,
        riflettore,
        perdite_riflessione=0.9,
    )

    # Limita a valori realistici
    p_focale_val = pressione_focale.to("MPa").magnitude
    p_focale_val = min(p_focale_val, 150)  # Max 150 MPa (realistico ESWT)
    pressione_focale = Q_(p_focale_val, "MPa")

    # STEP 5: Zona focale e EFD
    zona_focale = calcola_zona_focale(riflettore)

    # Rise time calibrato
    rise_time = calcola_rise_time_calibrato(pressione_focale)

    # EFD (stima da pressione e durata)
    durata_impulso = Q_(1, "us")
    efd = calcola_energy_flux_density(pressione_focale, durata_impulso)

    # Pressione negativa (tipicamente 10-20% della positiva)
    pressione_negativa = Q_(p_focale_val * 0.15, "MPa")

    # STEP 6: Cavitazione
    if config.calcola_cavitazione:
        cavitazione = analizza_cavitazione(pressione_focale, pressione_negativa)
    else:
        cavitazione = CavitationResult(
            cavitazione_attiva=False,
            pressione_negativa=pressione_negativa,
            soglia_superata=Q_(0, "MPa"),
            raggio_bolla_max=Q_(1, "um"),
            energia_cavitazione=Q_(0, "mJ"),
            densita_bolle=0.0,
        )

    # STEP 7: Confronto con Ogden
    # Trova la tensione Ogden più vicina
    tensione_int = int(round(config.tensione_kv))
    tensioni_ogden = list(OGDEN_DATA.keys())
    tensione_ogden = min(tensioni_ogden, key=lambda x: abs(x - tensione_int))

    focusing_result = FocusingResult(
        pressione_picco=pressione_focale,
        pressione_negativa=pressione_negativa,
        energy_flux_density=efd,
        zona_focale=zona_focale,
        efficienza_focalizzazione=riflettore.guadagno_geometrico(),
    )

    confronto = confronta_con_ogden(focusing_result, tensione_ogden)

    return ShockwaveSimulationResult(
        energia_scarica=energia,
        pressione_sorgente=pressione_sorgente,
        plasma_result=plasma_result,
        pressione_focale=pressione_focale,
        pressione_negativa=pressione_negativa,
        rise_time=rise_time,
        zona_focale=zona_focale,
        efd=efd,
        cavitazione=cavitazione,
        confronto_ogden=confronto,
        configurazione=config,
    )


def simula_serie_impulsi(
    riflettore: BaseReflector,
    n_impulsi: int,
    config: ShockwaveSimulationConfig = None,
    variabilita_percentuale: float = 10.0,
) -> Dict[str, Any]:
    """
    Simula una serie di impulsi con variabilità.

    I dispositivi elettroidraulici hanno variabilità pulse-to-pulse
    del 30-40% (Ogden). Questa funzione simula una serie di impulsi
    con variabilità gaussiana.

    Parametri:
        riflettore: Riflettore da utilizzare
        n_impulsi: Numero di impulsi da simulare
        config: Configurazione base
        variabilita_percentuale: Variabilità energia (default: 10%)

    Ritorna:
        Dizionario con statistiche della serie
    """
    if config is None:
        config = ShockwaveSimulationConfig()

    energia_base = calcola_energia_scarica(config.tensione_kv, config.capacita_uf)
    e_base = energia_base.to("J").magnitude

    # Array per raccolta risultati
    pressioni = []
    efds = []

    for i in range(n_impulsi):
        # Variabilità gaussiana sull'energia
        e_var = e_base * (1 + np.random.normal(0, variabilita_percentuale / 100))
        e_var = max(e_var, 0.1)  # Evita energie negative

        result = simula_impulso_completo(
            riflettore,
            config,
            energia_override=Q_(e_var, "J"),
        )

        pressioni.append(result.pressione_focale.to("MPa").magnitude)
        efds.append(result.efd.to("mJ/mm^2").magnitude)

    pressioni = np.array(pressioni)
    efds = np.array(efds)

    return {
        "n_impulsi": n_impulsi,
        "pressione_media_mpa": np.mean(pressioni),
        "pressione_std_mpa": np.std(pressioni),
        "pressione_min_mpa": np.min(pressioni),
        "pressione_max_mpa": np.max(pressioni),
        "efd_media_mj_mm2": np.mean(efds),
        "efd_std_mj_mm2": np.std(efds),
        "variabilita_pressione_pct": np.std(pressioni) / np.mean(pressioni) * 100,
        "variabilita_efd_pct": np.std(efds) / np.mean(efds) * 100,
    }


def crea_riflettore_ossatron() -> EllipticalReflector:
    """
    Crea un riflettore con geometria tipica OssaTron.

    Parametri approssimati dai dati Ogden (2001).

    Ritorna:
        EllipticalReflector configurato
    """
    # Parametri tipici OssaTron (stimati da zona focale)
    # Apertura ~80 mm, distanza focale ~150 mm
    return EllipticalReflector(
        apertura=Q_(80, "mm"),
        distanza_fuochi=Q_(150, "mm"),
        profondita=Q_(40, "mm"),
    )


def valida_simulazione_vs_ogden(
    tensione_kv: int = 20,
) -> Dict[str, Any]:
    """
    Valida la simulazione contro i dati OssaTron di Ogden.

    Esegue una simulazione con i parametri corrispondenti e
    confronta i risultati con i dati sperimentali.

    Parametri:
        tensione_kv: Tensione da validare (14, 20, o 28 kV)

    Ritorna:
        Dizionario con risultati simulazione e errori
    """
    if tensione_kv not in OGDEN_DATA:
        raise ValueError(f"Tensione {tensione_kv} kV non disponibile. Usa 14, 20, o 28.")

    dati_ogden = OGDEN_DATA[tensione_kv]

    # Crea riflettore OssaTron
    riflettore = crea_riflettore_ossatron()

    # Configura simulazione
    # Stima capacità da energia tipica
    # E = 0.5 * C * V² → C = 2E/V²
    # Per OssaTron: ~1 μF tipico
    config = ShockwaveSimulationConfig(
        tensione_kv=float(tensione_kv),
        capacita_uf=1.0,
        gap_elettrodi_mm=5.0,
        usa_propagazione_nonlineare=True,
    )

    # Esegui simulazione
    result = simula_impulso_completo(riflettore, config)

    # Calcola errori
    p_sim = result.pressione_focale.to("MPa").magnitude
    p_ogden = dati_ogden["p_max"]
    errore_pressione = abs(p_sim - p_ogden) / p_ogden * 100

    efd_sim = result.efd.to("mJ/mm^2").magnitude
    efd_ogden = dati_ogden["efd"]
    errore_efd = abs(efd_sim - efd_ogden) / efd_ogden * 100

    return {
        "tensione_kv": tensione_kv,
        "simulazione": {
            "pressione_mpa": p_sim,
            "efd_mj_mm2": efd_sim,
            "rise_time_ns": result.rise_time.to("ns").magnitude,
        },
        "ogden": {
            "pressione_mpa": p_ogden,
            "efd_mj_mm2": efd_ogden,
            "d_lat_mm": dati_ogden["d_lat"],
            "l_ax_mm": dati_ogden["l_ax"],
        },
        "errori": {
            "pressione_pct": errore_pressione,
            "efd_pct": errore_efd,
        },
        "validazione_ok": errore_pressione < 30 and errore_efd < 50,
    }


def report_simulazione(result: ShockwaveSimulationResult) -> str:
    """
    Genera un report testuale della simulazione.

    Parametri:
        result: Risultato della simulazione

    Ritorna:
        Stringa con il report formattato
    """
    lines = [
        "=" * 60,
        "REPORT SIMULAZIONE ONDA D'URTO ESWT",
        "=" * 60,
        "",
        "CONFIGURAZIONE:",
        f"  Tensione:        {result.configurazione.tensione_kv:.1f} kV",
        f"  Capacità:        {result.configurazione.capacita_uf:.2f} μF",
        f"  Gap elettrodi:   {result.configurazione.gap_elettrodi_mm:.1f} mm",
        f"  Non-linearità:   {'Sì' if result.configurazione.usa_propagazione_nonlineare else 'No'}",
        "",
        "ENERGIA:",
        f"  Energia scarica: {result.energia_scarica.to('J').magnitude:.2f} J",
        "",
        "PLASMA:",
        f"  Pressione iniziale: {result.plasma_result.pressione_iniziale.to('MPa').magnitude:.1f} MPa",
        f"  Raggio bolla:       {result.plasma_result.raggio_bolla.to('mm').magnitude:.2f} mm",
        f"  Temperatura:        {result.plasma_result.temperatura_plasma.to('K').magnitude:.0f} K",
        "",
        "ZONA FOCALE:",
        f"  Pressione picco:    {result.pressione_focale.to('MPa').magnitude:.1f} MPa",
        f"  Pressione negativa: {result.pressione_negativa.to('MPa').magnitude:.1f} MPa",
        f"  Rise time:          {result.rise_time.to('ns').magnitude:.0f} ns",
        f"  EFD:                {result.efd.to('mJ/mm^2').magnitude:.3f} mJ/mm²",
        f"  Diametro laterale:  {result.zona_focale.diametro_laterale.to('mm').magnitude:.1f} mm",
        f"  Lunghezza assiale:  {result.zona_focale.lunghezza_assiale.to('mm').magnitude:.1f} mm",
        "",
        "CAVITAZIONE:",
        f"  Attiva:             {'Sì' if result.cavitazione.cavitazione_attiva else 'No'}",
    ]

    if result.cavitazione.cavitazione_attiva:
        lines.extend([
            f"  Soglia superata:    {result.cavitazione.soglia_superata.to('MPa').magnitude:.1f} MPa",
            f"  Raggio bolla max:   {result.cavitazione.raggio_bolla_max.to('um').magnitude:.1f} μm",
            f"  Densità bolle:      {result.cavitazione.densita_bolle:.0f} bolle/mm³",
        ])

    lines.extend([
        "",
        "VALIDAZIONE vs OGDEN:",
    ])

    if result.confronto_ogden:
        for param, data in result.confronto_ogden.items():
            if isinstance(data, dict) and "errore_percentuale" in data:
                lines.append(f"  {param}: errore {data['errore_percentuale']:.1f}%")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
