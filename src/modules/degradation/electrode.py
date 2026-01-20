# Modello erosione elettrodo
"""
Modello per l'erosione degli elettrodi nei dispositivi ESWT.

L'erosione degli elettrodi è un processo critico che:
- Aumenta il gap inter-elettrodo nel tempo
- Riduce l'efficienza di generazione dell'onda d'urto
- Richiede interventi del motore di avvicinamento

Equazioni principali:
    - Massa erosa per impulso: Δm = k * Q^α (Q = carica trasferita)
    - Variazione gap: Δgap = Δm / (ρ * A_punta)
    - Volume eroso: ΔV = Δm / ρ

Il tasso di erosione dipende da:
- Materiale dell'elettrodo (tungsteno > rame > acciaio)
- Energia della scarica
- Polarità (anodo si erode più del catodo)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np

from ...core.units import ureg, Q_
from ...core.materials import MaterialsDatabase, ElectrodeMaterial


@dataclass
class ElectrodeState:
    """
    Stato corrente di un elettrodo.

    Attributi:
        materiale: Materiale dell'elettrodo
        massa_iniziale: Massa iniziale (g)
        massa_attuale: Massa attuale (g)
        gap_iniziale: Gap inter-elettrodo iniziale (mm)
        gap_attuale: Gap attuale (mm)
        numero_impulsi: Numero totale di impulsi erogati
        area_punta: Area della punta dell'elettrodo (mm²)
    """

    materiale: str
    massa_iniziale: "Q_"
    massa_attuale: "Q_"
    gap_iniziale: "Q_"
    gap_attuale: "Q_"
    numero_impulsi: int = 0
    area_punta: "Q_" = field(default_factory=lambda: Q_(1, "mm^2"))

    @property
    def massa_erosa(self) -> "Q_":
        """Massa totale erosa."""
        m_i = self.massa_iniziale.to("g").magnitude
        m_a = self.massa_attuale.to("g").magnitude
        return Q_(m_i - m_a, "g")

    @property
    def percentuale_erosione(self) -> float:
        """Percentuale di massa erosa."""
        m_i = self.massa_iniziale.to("g").magnitude
        m_a = self.massa_attuale.to("g").magnitude
        return 100 * (m_i - m_a) / m_i if m_i > 0 else 0

    @property
    def variazione_gap(self) -> "Q_":
        """Variazione del gap rispetto al valore iniziale."""
        g_i = self.gap_iniziale.to("mm").magnitude
        g_a = self.gap_attuale.to("mm").magnitude
        return Q_(g_a - g_i, "mm")


class ElectrodeErosionModel:
    """
    Modello di erosione per elettrodi ESWT.

    Il modello calcola l'erosione basandosi su:
    - Carica trasferita per impulso
    - Proprietà del materiale
    - Coefficienti empirici

    Parametri:
        materiale: Nome del materiale o istanza ElectrodeMaterial
        coefficiente_erosione: k nel modello Δm = k * Q^α (mg/C^α)
        esponente_carica: α nel modello (tipicamente 1.0-1.5)
    """

    def __init__(
        self,
        materiale: str = "tungsteno",
        coefficiente_erosione: float = None,
        esponente_carica: float = 1.2,
    ):
        self.materiale_nome = materiale

        # Carica proprietà materiale
        try:
            db = MaterialsDatabase()
            self.materiale = db.elettrodo(materiale)
            self.densita = self.materiale.densita
        except (FileNotFoundError, KeyError):
            # Fallback a valori default
            self.materiale = None
            self.densita = Q_(19300, "kg/m^3")  # Tungsteno default

        # Coefficienti erosione empirici per materiale
        # Valori tipici in mg/C^α
        coefficienti_default = {
            "tungsteno": 0.05,  # Più resistente
            "rame": 0.15,
            "acciaio_inox": 0.08,
        }

        if coefficiente_erosione is None:
            self.k = coefficienti_default.get(materiale, 0.1)
        else:
            self.k = coefficiente_erosione

        self.alpha = esponente_carica

    def calcola_erosione(
        self,
        carica_trasferita: "Q_",
    ) -> "Q_":
        """
        Calcola la massa erosa per una singola scarica.

        Δm = k * Q^α

        Parametri:
            carica_trasferita: Carica totale della scarica (C o mC)

        Ritorna:
            Massa erosa (mg)
        """
        q = carica_trasferita.to("C").magnitude

        # Modello: massa erosa proporzionale a Q^α
        delta_m = self.k * (q ** self.alpha)

        return Q_(delta_m, "mg")

    def calcola_variazione_gap(
        self,
        massa_erosa: "Q_",
        area_punta: "Q_",
    ) -> "Q_":
        """
        Calcola la variazione del gap dovuta all'erosione.

        Δgap = Δm / (ρ * A)

        Assunzione: erosione uniforme sulla punta (approssimazione).
        In realtà l'erosione crea crateri e la geometria è più complessa.

        Parametri:
            massa_erosa: Massa erosa (mg o g)
            area_punta: Area della punta dell'elettrodo (mm²)

        Ritorna:
            Variazione del gap (mm) - per ciascun elettrodo
        """
        m = massa_erosa.to("kg").magnitude
        A = area_punta.to("m^2").magnitude
        rho = self.densita.to("kg/m^3").magnitude

        # Volume eroso
        V = m / rho

        # Altezza erosa (assumendo erosione uniforme)
        h = V / A

        return Q_(h * 1000, "mm")  # Converti m -> mm


def calcola_erosione_impulso(
    energia_scarica: "Q_",
    tensione: "Q_",
    materiale: str = "tungsteno",
) -> "Q_":
    """
    Calcola l'erosione per un singolo impulso data l'energia.

    Stima la carica trasferita dall'energia e tensione,
    poi calcola l'erosione.

    Parametri:
        energia_scarica: Energia della scarica (J)
        tensione: Tensione di scarica (kV)
        materiale: Materiale dell'elettrodo

    Ritorna:
        Massa erosa per l'impulso (mg)
    """
    # Stima carica: E = 0.5 * C * V² → C = 2*E/V²
    # Q = C * V = 2*E/V
    E = energia_scarica.to("J").magnitude
    V = tensione.to("V").magnitude

    # Carica approssimata
    Q_coulomb = 2 * E / V if V > 0 else 0

    model = ElectrodeErosionModel(materiale=materiale)
    return model.calcola_erosione(Q_(Q_coulomb, "C"))


def calcola_vita_residua(
    stato: ElectrodeState,
    energia_media: "Q_",
    tensione: "Q_",
    soglia_gap_max: "Q_" = None,
) -> int:
    """
    Stima il numero di impulsi rimanenti prima della sostituzione.

    Parametri:
        stato: Stato attuale dell'elettrodo
        energia_media: Energia media per impulso
        tensione: Tensione di scarica
        soglia_gap_max: Gap massimo accettabile (default: 2x iniziale)

    Ritorna:
        Numero stimato di impulsi rimanenti
    """
    if soglia_gap_max is None:
        soglia_gap_max = Q_(stato.gap_iniziale.to("mm").magnitude * 2, "mm")

    gap_max = soglia_gap_max.to("mm").magnitude
    gap_attuale = stato.gap_attuale.to("mm").magnitude

    if gap_attuale >= gap_max:
        return 0

    # Calcola erosione per impulso
    erosione_per_impulso = calcola_erosione_impulso(
        energia_media, tensione, stato.materiale
    )

    # Calcola variazione gap per impulso
    model = ElectrodeErosionModel(materiale=stato.materiale)
    delta_gap = model.calcola_variazione_gap(
        erosione_per_impulso, stato.area_punta
    )

    # Per due elettrodi (anodo + catodo), il gap aumenta del doppio
    delta_gap_totale = delta_gap.to("mm").magnitude * 2

    if delta_gap_totale <= 0:
        return float("inf")

    impulsi_rimanenti = int((gap_max - gap_attuale) / delta_gap_totale)
    return max(impulsi_rimanenti, 0)


def simula_degradazione(
    stato_iniziale: ElectrodeState,
    numero_impulsi: int,
    energia_per_impulso: "Q_",
    tensione: "Q_",
    intervallo_compensazione: int = 1000,
    compensazione_gap: "Q_" = None,
) -> Tuple[ElectrodeState, List[dict]]:
    """
    Simula la degradazione dell'elettrodo su più impulsi.

    Include la compensazione automatica del gap (motore avvicinamento).

    Parametri:
        stato_iniziale: Stato iniziale dell'elettrodo
        numero_impulsi: Numero di impulsi da simulare
        energia_per_impulso: Energia di ogni impulso
        tensione: Tensione di scarica
        intervallo_compensazione: Ogni quanti impulsi compensare il gap
        compensazione_gap: Entità della compensazione (default: auto)

    Ritorna:
        Tuple (stato_finale, storico) dove storico è una lista di dizionari
        con l'evoluzione dei parametri
    """
    model = ElectrodeErosionModel(materiale=stato_iniziale.materiale)

    # Copia stato
    massa = stato_iniziale.massa_attuale.to("g").magnitude
    gap = stato_iniziale.gap_attuale.to("mm").magnitude
    n_impulsi = stato_iniziale.numero_impulsi

    # Calcola erosione per singolo impulso
    E = energia_per_impulso.to("J").magnitude
    V = tensione.to("V").magnitude
    Q_impulso = 2 * E / V if V > 0 else 0
    erosione_impulso = model.calcola_erosione(Q_(Q_impulso, "C"))
    delta_gap_impulso = model.calcola_variazione_gap(
        erosione_impulso, stato_iniziale.area_punta
    ).to("mm").magnitude * 2  # x2 per due elettrodi

    # Compensazione default
    if compensazione_gap is None:
        # Compensa per mantenere gap circa costante
        compensazione_gap = Q_(delta_gap_impulso * intervallo_compensazione * 0.9, "mm")

    comp_mm = compensazione_gap.to("mm").magnitude

    storico = []
    interventi_motore = 0

    for i in range(numero_impulsi):
        # Erosione
        massa -= erosione_impulso.to("g").magnitude
        gap += delta_gap_impulso
        n_impulsi += 1

        # Compensazione periodica
        if (i + 1) % intervallo_compensazione == 0:
            gap -= comp_mm
            gap = max(gap, stato_iniziale.gap_iniziale.to("mm").magnitude * 0.5)
            interventi_motore += 1

        # Log periodico
        if (i + 1) % (numero_impulsi // 10 + 1) == 0:
            storico.append({
                "impulso": n_impulsi,
                "massa_g": massa,
                "gap_mm": gap,
                "interventi_motore": interventi_motore,
            })

    # Stato finale
    stato_finale = ElectrodeState(
        materiale=stato_iniziale.materiale,
        massa_iniziale=stato_iniziale.massa_iniziale,
        massa_attuale=Q_(massa, "g"),
        gap_iniziale=stato_iniziale.gap_iniziale,
        gap_attuale=Q_(gap, "mm"),
        numero_impulsi=n_impulsi,
        area_punta=stato_iniziale.area_punta,
    )

    return stato_finale, storico


def crea_stato_elettrodo(
    materiale: str = "tungsteno",
    gap_iniziale: "Q_" = None,
    diametro_punta: "Q_" = None,
) -> ElectrodeState:
    """
    Crea uno stato elettrodo iniziale con valori di default.

    Parametri:
        materiale: Nome del materiale
        gap_iniziale: Gap inter-elettrodo (default: 5 mm)
        diametro_punta: Diametro della punta (default: 2 mm)

    Ritorna:
        ElectrodeState inizializzato
    """
    if gap_iniziale is None:
        gap_iniziale = Q_(5, "mm")

    if diametro_punta is None:
        diametro_punta = Q_(2, "mm")

    # Area punta (circolare)
    r = diametro_punta.to("mm").magnitude / 2
    area = np.pi * r ** 2

    # Massa tipica elettrodo (stima)
    try:
        db = MaterialsDatabase()
        mat = db.elettrodo(materiale)
        rho = mat.densita.to("g/cm^3").magnitude
    except:
        rho = 19.3  # Tungsteno default

    # Volume approssimato elettrodo (cilindro 2mm x 10mm)
    volume_cm3 = np.pi * (0.1) ** 2 * 1  # r=1mm, h=10mm
    massa = rho * volume_cm3

    return ElectrodeState(
        materiale=materiale,
        massa_iniziale=Q_(massa, "g"),
        massa_attuale=Q_(massa, "g"),
        gap_iniziale=gap_iniziale,
        gap_attuale=gap_iniziale,
        numero_impulsi=0,
        area_punta=Q_(area, "mm^2"),
    )
