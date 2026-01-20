# Monitor di feedback elettrico per il sistema di controllo ESWT
"""
Modulo per il monitoraggio del feedback elettrico dalla scarica.

Questo modulo implementa:
    - ElectricalFeedbackMonitor: Monitoraggio e analisi feedback
    - Stima gap da resistenza plasma
    - Calcolo efficienza scarica

Il feedback elettrico è fondamentale per il controllo PID del gap:
    - Se I_picco < I_target → gap troppo grande
    - Se I_picco > I_target → gap troppo piccolo

Riferimenti:
    - Prompt.md sezione 2.D
    - Chen et al. (2010) - Modello resistenza plasma
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np

from ...core.units import ureg, Q_
from .data_structures import FeedbackMeasurement, ControlStatus


@dataclass
class FeedbackConfig:
    """
    Configurazione del monitor di feedback.

    Attributi:
        gap_reference: Gap di riferimento per calibrazione (mm)
        R_reference: Resistenza di riferimento al gap nominale (Ω)
        k_gap_resistance: Coefficiente gap-resistenza (mm/Ω)
        I_nominal: Corrente nominale attesa (kA)
        noise_std: Deviazione standard del rumore di misura (A)
        enable_filtering: Abilita filtro passa-basso
        filter_alpha: Costante filtro EMA (0-1)
    """

    gap_reference: "Q_" = None
    R_reference: "Q_" = None
    k_gap_resistance: float = 25.0  # mm/Ω (empirico)
    I_nominal: "Q_" = None
    noise_std: "Q_" = None
    enable_filtering: bool = True
    filter_alpha: float = 0.3  # Filtro passa-basso

    def __post_init__(self):
        if self.gap_reference is None:
            self.gap_reference = Q_(5, "mm")
        if self.R_reference is None:
            self.R_reference = Q_(0.2, "ohm")
        if self.I_nominal is None:
            self.I_nominal = Q_(10, "kA")
        if self.noise_std is None:
            self.noise_std = Q_(50, "A")


class ElectricalFeedbackMonitor:
    """
    Monitor di feedback elettrico per controllo gap ESWT.

    Analizza i risultati della scarica per estrarre:
    - Corrente di picco (feedback principale per PID)
    - Resistenza plasma media (stima indiretta del gap)
    - Efficienza energetica

    La relazione gap-resistenza è basata su:
        R_plasma ∝ gap^β  (β ~ 1 per plasma in acqua)

    Quindi:
        gap_estimated = gap_ref * (R_measured / R_ref)^(1/β)

    Esempio:
        >>> monitor = ElectricalFeedbackMonitor()
        >>> feedback = monitor.measure(discharge_result, electrode_state)
        >>> print(f"Gap stimato: {feedback.gap_estimated}")
    """

    def __init__(self, config: FeedbackConfig = None):
        """
        Inizializza il monitor di feedback.

        Parametri:
            config: Configurazione del monitor (opzionale)
        """
        self.config = config or FeedbackConfig()

        # Storico misure per filtering
        self._history: List[FeedbackMeasurement] = []
        self._filtered_I: Optional["Q_"] = None
        self._filtered_R: Optional["Q_"] = None

        # Statistiche
        self._total_measurements = 0
        self._anomaly_count = 0

    def measure(
        self,
        discharge_result,  # DischargeResult
        gap_measured: "Q_" = None,
        energia_iniziale: "Q_" = None,
    ) -> FeedbackMeasurement:
        """
        Estrae misure di feedback dal risultato della scarica.

        Parametri:
            discharge_result: Risultato dalla simulazione scarica
            gap_measured: Gap noto/misurato (opzionale)
            energia_iniziale: Energia iniziale condensatore (opzionale)

        Ritorna:
            FeedbackMeasurement con tutti i dati di feedback
        """
        timestamp = datetime.now()

        # Estrai corrente di picco
        I_picco = discharge_result.corrente_picco.to("kA")

        # Calcola resistenza media del plasma
        R_plasma_avg = self._calcola_resistenza_media(discharge_result)

        # Energia rilasciata
        E_rilasciata = discharge_result.energia_rilasciata.to("J")

        # Stima gap dalla resistenza
        gap_estimated = self._estimate_gap_from_resistance(
            R_plasma_avg,
            self.config.gap_reference
        )

        # Se gap_measured non fornito, usa la stima
        if gap_measured is None:
            gap_measured = gap_estimated

        # Calcola efficienza
        if energia_iniziale is not None:
            E_iniziale = energia_iniziale.to("J").magnitude
            E_plasma = discharge_result.energia_plasma.to("J").magnitude
            efficienza = E_plasma / E_iniziale if E_iniziale > 0 else 0.0
        else:
            # Stima efficienza dal rapporto energia plasma/rilasciata
            E_plasma = discharge_result.energia_plasma.to("J").magnitude
            E_rilasciata_val = E_rilasciata.magnitude
            efficienza = E_plasma / E_rilasciata_val if E_rilasciata_val > 0 else 0.0

        # Applica filtering se abilitato
        if self.config.enable_filtering and self._filtered_I is not None:
            alpha = self.config.filter_alpha
            I_picco_filtered = Q_(
                alpha * I_picco.magnitude + (1 - alpha) * self._filtered_I.magnitude,
                "kA"
            )
            R_filtered = Q_(
                alpha * R_plasma_avg.magnitude + (1 - alpha) * self._filtered_R.magnitude,
                "ohm"
            )
        else:
            I_picco_filtered = I_picco
            R_filtered = R_plasma_avg

        # Aggiorna stati filtrati
        self._filtered_I = I_picco_filtered
        self._filtered_R = R_filtered

        # Crea misura
        measurement = FeedbackMeasurement(
            timestamp=timestamp,
            I_picco=I_picco_filtered,
            R_plasma_avg=R_filtered,
            E_rilasciata=E_rilasciata,
            gap_estimated=gap_estimated,
            gap_measured=gap_measured,
            efficienza=float(np.clip(efficienza, 0, 1)),
        )

        # Aggiorna statistiche
        self._total_measurements += 1
        self._history.append(measurement)

        # Limita storico a ultimi 1000 campioni
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return measurement

    def _calcola_resistenza_media(self, discharge_result) -> "Q_":
        """
        Calcola la resistenza media del plasma durante la scarica.

        Usa l'integrale della resistenza pesato per la corrente:
            R_avg = ∫R(t)·I(t)dt / ∫I(t)dt

        Parametri:
            discharge_result: Risultato scarica

        Ritorna:
            Resistenza media in Ohm
        """
        R_array = discharge_result.resistenza_plasma  # numpy array
        I_array = np.abs(discharge_result.corrente)   # numpy array

        # Evita divisione per zero
        I_integral = np.trapz(I_array, discharge_result.tempo)
        if I_integral < 1e-10:
            return self.config.R_reference

        RI_integral = np.trapz(R_array * I_array, discharge_result.tempo)
        R_avg = RI_integral / I_integral

        return Q_(float(R_avg), "ohm")

    def _estimate_gap_from_resistance(
        self,
        R_measured: "Q_",
        gap_reference: "Q_"
    ) -> "Q_":
        """
        Stima il gap inter-elettrodo dalla resistenza del plasma.

        Modello semplificato:
            R_plasma ∝ gap  (approssimativamente lineare)

        Quindi:
            gap_est = gap_ref * (R_measured / R_ref)

        Per modelli più accurati si può usare:
            gap_est = gap_ref * (R_measured / R_ref)^(1/β)

        dove β dipende dalla geometria e dalla temperatura del plasma.

        Parametri:
            R_measured: Resistenza misurata (Ω)
            gap_reference: Gap di riferimento (mm)

        Ritorna:
            Gap stimato in mm
        """
        R_meas = R_measured.to("ohm").magnitude
        R_ref = self.config.R_reference.to("ohm").magnitude
        gap_ref = gap_reference.to("mm").magnitude

        # Evita valori non fisici
        if R_meas <= 0 or R_ref <= 0:
            return gap_reference

        # Rapporto resistenze
        ratio = R_meas / R_ref

        # Limita il rapporto a valori ragionevoli (0.1 - 10)
        ratio = np.clip(ratio, 0.1, 10.0)

        # Stima lineare
        gap_estimated = gap_ref * ratio

        # Limita a range fisico (1-20 mm tipico per ESWT)
        gap_estimated = np.clip(gap_estimated, 1.0, 20.0)

        return Q_(float(gap_estimated), "mm")

    def diagnose_status(
        self,
        measurement: FeedbackMeasurement,
        I_target: "Q_",
        tolerance: float = 0.2
    ) -> Tuple[ControlStatus, str]:
        """
        Diagnostica lo stato del sistema basandosi sulla misura.

        Parametri:
            measurement: Ultima misura di feedback
            I_target: Corrente target
            tolerance: Tolleranza relativa (default 20%)

        Ritorna:
            Tuple (ControlStatus, messaggio diagnostico)
        """
        I_meas = measurement.I_picco.to("kA").magnitude
        I_tgt = I_target.to("kA").magnitude

        error_rel = abs(I_meas - I_tgt) / I_tgt if I_tgt > 0 else 0

        # Diagnosi basata sull'errore
        if error_rel < tolerance * 0.5:
            return ControlStatus.NOMINAL, "Sistema nominale"

        elif error_rel < tolerance:
            if I_meas < I_tgt:
                msg = f"Corrente bassa ({I_meas:.1f} kA < {I_tgt:.1f} kA), gap probabilmente alto"
            else:
                msg = f"Corrente alta ({I_meas:.1f} kA > {I_tgt:.1f} kA), gap probabilmente basso"
            return ControlStatus.WARNING, msg

        elif error_rel < tolerance * 2:
            if I_meas < I_tgt:
                msg = f"Corrente molto bassa ({I_meas:.1f} kA), verificare gap"
            else:
                msg = f"Corrente molto alta ({I_meas:.1f} kA), rischio arco"
            return ControlStatus.CRITICAL, msg

        else:
            return ControlStatus.ERROR, f"Errore critico: I={I_meas:.1f} kA fuori range"

    def get_trend(self, window: int = 10) -> dict:
        """
        Calcola il trend delle misure recenti.

        Parametri:
            window: Finestra di campioni per il calcolo

        Ritorna:
            Dizionario con statistiche del trend
        """
        if len(self._history) < 2:
            return {
                "samples": len(self._history),
                "I_trend": 0.0,
                "R_trend": 0.0,
                "gap_trend": 0.0,
            }

        # Prendi ultimi N campioni
        recent = self._history[-window:]

        # Calcola trend (pendenza)
        n = len(recent)
        x = np.arange(n)

        I_values = [m.I_picco.to("kA").magnitude for m in recent]
        R_values = [m.R_plasma_avg.to("ohm").magnitude for m in recent]
        gap_values = [m.gap_estimated.to("mm").magnitude for m in recent]

        # Regressione lineare semplice
        def linear_trend(y):
            if len(y) < 2:
                return 0.0
            slope = np.polyfit(x[:len(y)], y, 1)[0]
            return float(slope)

        return {
            "samples": n,
            "I_mean_kA": float(np.mean(I_values)),
            "I_std_kA": float(np.std(I_values)),
            "I_trend": linear_trend(I_values),  # kA/sample
            "R_mean_ohm": float(np.mean(R_values)),
            "R_trend": linear_trend(R_values),  # Ω/sample
            "gap_mean_mm": float(np.mean(gap_values)),
            "gap_trend": linear_trend(gap_values),  # mm/sample
            "eta_mean": float(np.mean([m.efficienza for m in recent])),
        }

    def reset(self):
        """Resetta lo stato del monitor."""
        self._history.clear()
        self._filtered_I = None
        self._filtered_R = None
        self._total_measurements = 0
        self._anomaly_count = 0

    @property
    def total_measurements(self) -> int:
        """Numero totale di misure effettuate."""
        return self._total_measurements

    @property
    def history(self) -> List[FeedbackMeasurement]:
        """Storico delle misure (ultimi 1000 campioni)."""
        return self._history.copy()

    def __repr__(self) -> str:
        return (
            f"ElectricalFeedbackMonitor("
            f"measurements={self._total_measurements}, "
            f"gap_ref={self.config.gap_reference:~.1fP})"
        )
