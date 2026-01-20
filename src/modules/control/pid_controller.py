# Controllore PID per gap elettrodi ESWT
"""
Controllore PID per il meccanismo di avvicinamento automatico delle punte.

Implementa l'algoritmo richiesto dal prompt.md sezione 2.D:
"Sviluppare un algoritmo PID per il meccanismo di avvicinamento automatico
delle punte basato sul monitoraggio del feedback elettrico."

Dinamica del controllo:
    - Se I_picco < I_target → gap troppo grande → avvicina punte
    - Se I_picco > I_target → gap troppo piccolo → allontana punte

Equazione PID:
    u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(de/dt)
    dove:
        e(t) = I_target - I_feedback
        u(t) = Δgap (movimento motore)

Riferimenti:
    - Prompt.md sezione 2.D
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Optional
import numpy as np

from ...core.units import ureg, Q_
from .data_structures import MotorAction, PIDParameters, ControlStatus


class PIDGapController:
    """
    Controllore PID per mantenere il gap inter-elettrodo ottimale.

    Il controllore monitora la corrente di picco e regola il gap
    per mantenere la corrente vicina al setpoint.

    Parametri tuning tipici ESWT:
        - Kp = 0.5 mm/kA (proporzionale)
        - Ki = 0.01 mm/(kA·s) (integrale)
        - Kd = 0.1 mm·s/kA (derivativo)
        - I_target = 10 kA (dipende dalla configurazione)
    """

    def __init__(
        self,
        params: PIDParameters = None,
        anti_windup_limit: float = 5.0,
    ):
        """
        Inizializza il controllore PID.

        Parametri:
            params: Parametri PID (default: valori tipici ESWT)
            anti_windup_limit: Limite per anti-windup integrale (mm)
        """
        self.params = params or PIDParameters()

        # Guadagni
        self.Kp = self.params.Kp
        self.Ki = self.params.Ki
        self.Kd = self.params.Kd

        # Limiti
        self.gap_min = self.params.gap_min.to("mm").magnitude
        self.gap_max = self.params.gap_max.to("mm").magnitude
        self.dt = self.params.dt.to("s").magnitude
        self.anti_windup = anti_windup_limit

        # Setpoint
        self.I_target = self.params.I_target.to("A").magnitude

        # Stato interno
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.motor_log: List[MotorAction] = []

        # Statistiche
        self.total_interventions = 0
        self.successful_interventions = 0

    def reset(self):
        """Resetta lo stato interno del controllore."""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.motor_log = []
        self.total_interventions = 0
        self.successful_interventions = 0

    def step(
        self,
        I_feedback: "Q_",
        gap_current: "Q_",
        impulse_num: int,
    ) -> Tuple["Q_", MotorAction]:
        """
        Calcola il comando di movimento per il motore.

        Parametri:
            I_feedback: Corrente di picco misurata
            gap_current: Gap attuale
            impulse_num: Numero impulso corrente

        Ritorna:
            Tupla (gap_nuovo, azione_motore)
        """
        # Converti a magnitudini
        I_fb = I_feedback.to("A").magnitude
        gap = gap_current.to("mm").magnitude

        # Errore di controllo (positivo se corrente bassa = gap grande)
        error = self.I_target - I_fb  # [A]

        # Termine proporzionale
        u_p = self.Kp * error / 1000  # Converti A → kA per scaling

        # Termine integrale con anti-windup
        self.integral_error += error * self.dt / 1000
        self.integral_error = np.clip(
            self.integral_error,
            -self.anti_windup,
            self.anti_windup
        )
        u_i = self.Ki * self.integral_error

        # Termine derivativo
        derivative = (error - self.prev_error) / self.dt / 1000
        u_d = self.Kd * derivative
        self.prev_error = error

        # Output PID totale [mm]
        delta_gap = u_p + u_i + u_d

        # Calcola nuovo gap con saturazione
        gap_new = gap + delta_gap
        gap_new_saturated = np.clip(gap_new, self.gap_min, self.gap_max)

        # Determina status
        if gap_new_saturated >= self.gap_max:
            status = "LIMITE_MAX"
        elif gap_new_saturated <= self.gap_min:
            status = "LIMITE_MIN"
        else:
            status = "OK"
            self.successful_interventions += 1

        self.total_interventions += 1

        # Crea azione motore
        action = MotorAction(
            impulse_num=impulse_num,
            timestamp=datetime.now(),
            gap_before=gap_current,
            gap_after=Q_(gap_new_saturated, "mm"),
            delta_gap=Q_(gap_new_saturated - gap, "mm"),
            current_feedback=I_feedback,
            error_signal=Q_(error, "A"),
            pid_output=(u_p, u_i, u_d),
            status=status,
        )

        self.motor_log.append(action)

        return Q_(gap_new_saturated, "mm"), action

    def set_target_current(self, I_target: "Q_"):
        """Imposta la corrente obiettivo."""
        self.I_target = I_target.to("A").magnitude

    def set_gains(self, Kp: float = None, Ki: float = None, Kd: float = None):
        """Aggiorna i guadagni PID."""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd

    def get_status(self) -> ControlStatus:
        """Ritorna lo stato del controllore."""
        if not self.motor_log:
            return ControlStatus.NOMINAL

        last_action = self.motor_log[-1]
        if last_action.status == "LIMITE_MAX":
            return ControlStatus.CRITICAL
        elif last_action.status == "LIMITE_MIN":
            return ControlStatus.WARNING
        elif last_action.status == "ERRORE":
            return ControlStatus.ERROR
        else:
            return ControlStatus.NOMINAL

    def get_statistics(self) -> dict:
        """Ritorna statistiche del controllore."""
        if not self.motor_log:
            return {
                "total_interventions": 0,
                "successful_rate": 0.0,
            }

        gap_changes = [
            a.delta_gap.to("mm").magnitude
            for a in self.motor_log
        ]

        return {
            "total_interventions": self.total_interventions,
            "successful_interventions": self.successful_interventions,
            "successful_rate": self.successful_interventions / max(1, self.total_interventions),
            "total_gap_change_mm": sum(gap_changes),
            "avg_gap_change_mm": np.mean(gap_changes) if gap_changes else 0,
            "max_gap_change_mm": max(gap_changes) if gap_changes else 0,
            "min_gap_change_mm": min(gap_changes) if gap_changes else 0,
            "integral_error": self.integral_error,
        }


def calcola_guadagni_ziegler_nichols(
    K_u: float,
    T_u: float,
    tipo: str = "classico",
) -> Tuple[float, float, float]:
    """
    Calcola i guadagni PID usando il metodo Ziegler-Nichols.

    Parametri:
        K_u: Guadagno ultimo (gain at oscillation)
        T_u: Periodo ultimo (period of oscillation) in secondi
        tipo: "classico", "no_overshoot", "some_overshoot"

    Ritorna:
        Tupla (Kp, Ki, Kd)
    """
    if tipo == "classico":
        Kp = 0.6 * K_u
        Ki = 2 * Kp / T_u
        Kd = Kp * T_u / 8
    elif tipo == "no_overshoot":
        Kp = 0.2 * K_u
        Ki = 2 * Kp / T_u
        Kd = Kp * T_u / 3
    elif tipo == "some_overshoot":
        Kp = 0.33 * K_u
        Ki = 2 * Kp / T_u
        Kd = Kp * T_u / 3
    else:
        raise ValueError(f"Tipo non riconosciuto: {tipo}")

    return Kp, Ki, Kd


def simula_risposta_pid(
    controller: PIDGapController,
    n_impulsi: int,
    gap_iniziale: "Q_",
    erosione_per_impulso: "Q_",
    rumore_corrente: float = 0.05,
) -> List[MotorAction]:
    """
    Simula la risposta del controllore PID.

    Utile per tuning e validazione del controllore.

    Parametri:
        controller: Istanza del controllore PID
        n_impulsi: Numero di impulsi da simulare
        gap_iniziale: Gap iniziale
        erosione_per_impulso: Erosione per impulso (disturbo)
        rumore_corrente: Rumore percentuale sulla corrente (0-1)

    Ritorna:
        Lista di azioni motore
    """
    gap = gap_iniziale.to("mm").magnitude
    I_target = controller.I_target

    actions = []

    for i in range(n_impulsi):
        # Simula corrente con relazione inversa al gap
        # I ∝ 1/gap (approssimazione)
        I_base = I_target * (5.0 / gap)  # Normalizzato a gap=5mm

        # Aggiungi rumore
        I_noise = I_base * (1 + np.random.normal(0, rumore_corrente))
        I_feedback = Q_(max(I_noise, 100), "A")  # Min 100 A

        # Step del controllore
        gap_new, action = controller.step(
            I_feedback,
            Q_(gap, "mm"),
            impulse_num=i,
        )

        actions.append(action)

        # Aggiorna gap con erosione (disturbo)
        gap = gap_new.to("mm").magnitude
        gap += erosione_per_impulso.to("mm").magnitude * 2  # Due elettrodi

    return actions
