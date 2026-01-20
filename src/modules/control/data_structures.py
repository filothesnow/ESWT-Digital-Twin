# Strutture dati per il modulo Control
"""
Classi e strutture dati per il sistema di controllo ESWT.

Questo modulo definisce:
    - ControlState: Stato del sistema di controllo
    - MotorAction: Azione del motore di avvicinamento
    - FeedbackMeasurement: Misura di feedback elettrico
    - ControlMode: Modalità di controllo

Riferimenti:
    - Prompt.md sezione 2.D
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Tuple, Optional, Dict, Any

from ...core.units import ureg, Q_


class ControlMode(Enum):
    """Modalità di controllo disponibili."""

    OPEN_LOOP = "open_loop"    # Nessun feedback
    PID = "pid"                # Controllo PID classico
    FUZZY = "fuzzy"            # Logica Fuzzy
    HYBRID = "hybrid"          # Combinazione PID + Fuzzy


class ControlStatus(Enum):
    """Stati del sistema di controllo."""

    NOMINAL = "NOMINAL"        # Funzionamento normale
    WARNING = "WARNING"        # Avviso (gap vicino ai limiti)
    CRITICAL = "CRITICAL"      # Critico (gap fuori range)
    ERROR = "ERROR"            # Errore di sistema
    MAINTENANCE = "MAINTENANCE"  # Richiesta manutenzione


@dataclass
class ControlState:
    """
    Stato completo del sistema di controllo.

    Attributi:
        gap_target: Gap obiettivo (mm)
        gap_current: Gap attuale (mm)
        I_target: Corrente di picco obiettivo (kA)
        I_feedback: Corrente di picco misurata (kA)
        R_plasma: Resistenza plasma media (Ω)
        eta_system: Efficienza sistema (0-1)
        impulse_count: Numero impulsi eseguiti
        motor_interventions: Numero interventi motore
        mode: Modalità di controllo attiva
        status: Stato del sistema
        timestamp: Timestamp dello stato
    """

    gap_target: "Q_"
    gap_current: "Q_"
    I_target: "Q_"
    I_feedback: "Q_"
    R_plasma: "Q_"
    eta_system: float
    impulse_count: int
    motor_interventions: int
    mode: ControlMode = ControlMode.PID
    status: ControlStatus = ControlStatus.NOMINAL
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serializza lo stato in dizionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gap_target_mm": self.gap_target.to("mm").magnitude,
            "gap_current_mm": self.gap_current.to("mm").magnitude,
            "I_target_kA": self.I_target.to("kA").magnitude,
            "I_feedback_kA": self.I_feedback.to("kA").magnitude,
            "R_plasma_ohm": self.R_plasma.to("ohm").magnitude,
            "eta_system": self.eta_system,
            "impulse_count": self.impulse_count,
            "motor_interventions": self.motor_interventions,
            "mode": self.mode.value,
            "status": self.status.value,
        }


@dataclass
class MotorAction:
    """
    Azione del motore di avvicinamento.

    Registra ogni intervento del motore per il log richiesto
    dal prompt.md: "log degli interventi del motore di avvicinamento"

    Attributi:
        impulse_num: Numero impulso corrente
        timestamp: Timestamp dell'azione
        gap_before: Gap prima dell'intervento (mm)
        gap_after: Gap dopo l'intervento (mm)
        delta_gap: Variazione di gap (mm)
        current_feedback: Corrente di feedback che ha causato l'azione (kA)
        error_signal: Errore rispetto al setpoint (A)
        pid_output: Componenti PID (P, I, D)
        status: Stato dell'azione ("OK", "LIMITE_MAX", "LIMITE_MIN", "ERRORE")
    """

    impulse_num: int
    timestamp: datetime
    gap_before: "Q_"
    gap_after: "Q_"
    delta_gap: "Q_"
    current_feedback: "Q_"
    error_signal: "Q_"
    pid_output: Tuple[float, float, float]  # (u_p, u_i, u_d)
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializza l'azione in dizionario per export."""
        return {
            "impulse": self.impulse_num,
            "timestamp": self.timestamp.isoformat(),
            "gap_before_mm": self.gap_before.to("mm").magnitude,
            "gap_after_mm": self.gap_after.to("mm").magnitude,
            "delta_gap_mm": self.delta_gap.to("mm").magnitude,
            "I_feedback_kA": self.current_feedback.to("kA").magnitude,
            "error_A": self.error_signal.to("A").magnitude,
            "pid_p": self.pid_output[0],
            "pid_i": self.pid_output[1],
            "pid_d": self.pid_output[2],
            "status": self.status,
        }


@dataclass
class FeedbackMeasurement:
    """
    Misura di feedback elettrico dalla scarica.

    Attributi:
        timestamp: Timestamp della misura
        I_picco: Corrente di picco (kA)
        R_plasma_avg: Resistenza plasma media (Ω)
        E_rilasciata: Energia rilasciata (J)
        gap_estimated: Gap stimato dal feedback (mm)
        gap_measured: Gap misurato/noto (mm)
        efficienza: Efficienza della scarica (0-1)
    """

    timestamp: datetime
    I_picco: "Q_"
    R_plasma_avg: "Q_"
    E_rilasciata: "Q_"
    gap_estimated: "Q_"
    gap_measured: "Q_"
    efficienza: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializza la misura in dizionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "I_picco_kA": self.I_picco.to("kA").magnitude,
            "R_plasma_avg_ohm": self.R_plasma_avg.to("ohm").magnitude,
            "E_rilasciata_J": self.E_rilasciata.to("J").magnitude,
            "gap_estimated_mm": self.gap_estimated.to("mm").magnitude,
            "gap_measured_mm": self.gap_measured.to("mm").magnitude,
            "efficienza": self.efficienza,
        }


@dataclass
class PIDParameters:
    """
    Parametri del controllore PID.

    Attributi:
        Kp: Guadagno proporzionale (mm/kA)
        Ki: Guadagno integrale (mm/(kA·s))
        Kd: Guadagno derivativo (mm·s/kA)
        I_target: Corrente obiettivo (kA)
        gap_min: Gap minimo consentito (mm)
        gap_max: Gap massimo consentito (mm)
        dt: Intervallo di campionamento (s)
    """

    Kp: float = 0.5       # mm/kA
    Ki: float = 0.01      # mm/(kA·s)
    Kd: float = 0.1       # mm·s/kA
    I_target: "Q_" = field(default_factory=lambda: Q_(10, "kA"))
    gap_min: "Q_" = field(default_factory=lambda: Q_(3, "mm"))
    gap_max: "Q_" = field(default_factory=lambda: Q_(15, "mm"))
    dt: "Q_" = field(default_factory=lambda: Q_(1, "ms"))

    def to_dict(self) -> Dict[str, Any]:
        """Serializza i parametri."""
        return {
            "Kp_mm_per_kA": self.Kp,
            "Ki_mm_per_kAs": self.Ki,
            "Kd_mm_s_per_kA": self.Kd,
            "I_target_kA": self.I_target.to("kA").magnitude,
            "gap_min_mm": self.gap_min.to("mm").magnitude,
            "gap_max_mm": self.gap_max.to("mm").magnitude,
            "dt_ms": self.dt.to("ms").magnitude,
        }
