# Stimatore del gap inter-elettrodo con filtro di Kalman
"""
Modulo per la stima robusta del gap inter-elettrodo.

Questo modulo implementa:
    - GapEstimator: Filtro di Kalman per stima gap
    - Fusione sensori (resistenza plasma + encoder motore)
    - Predizione stato futuro

Il gap varia per:
    1. Erosione elettrodi (lenta, ~μm/impulso)
    2. Movimento motore (veloce, comandato)
    3. Vibrazioni/deriva (disturbi)

Modello di stato:
    x = [gap, gap_dot, erosion_rate]ᵀ

Riferimenti:
    - Kalman, R. E. (1960)
    - Prompt.md sezione 2.D
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from ...core.units import ureg, Q_


@dataclass
class EstimatorConfig:
    """
    Configurazione del filtro di Kalman.

    Attributi:
        gap_initial: Gap iniziale stimato (mm)
        gap_uncertainty: Incertezza iniziale gap (mm)
        process_noise_gap: Rumore processo gap (mm²)
        process_noise_velocity: Rumore processo velocità (mm²/s²)
        process_noise_erosion: Rumore processo erosione (mm²/impulso²)
        measurement_noise_resistance: Rumore misura da resistenza (mm²)
        measurement_noise_encoder: Rumore misura da encoder (mm²)
        dt: Intervallo temporale tra impulsi (s)
    """

    gap_initial: "Q_" = None
    gap_uncertainty: float = 1.0          # mm
    process_noise_gap: float = 0.001      # mm²
    process_noise_velocity: float = 0.01  # mm²/s²
    process_noise_erosion: float = 1e-8   # mm²/impulso²
    measurement_noise_resistance: float = 0.5   # mm²
    measurement_noise_encoder: float = 0.01     # mm²
    dt: float = 0.001                     # s (1 ms tra impulsi)

    def __post_init__(self):
        if self.gap_initial is None:
            self.gap_initial = Q_(5, "mm")


@dataclass
class GapEstimate:
    """
    Risultato della stima del gap.

    Attributi:
        gap: Gap stimato (mm)
        gap_uncertainty: Incertezza 1σ (mm)
        gap_velocity: Velocità variazione gap (mm/s)
        erosion_rate: Tasso erosione stimato (mm/impulso)
        confidence: Confidenza stima (0-1)
    """

    gap: "Q_"
    gap_uncertainty: float
    gap_velocity: float
    erosion_rate: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "gap_mm": self.gap.to("mm").magnitude,
            "uncertainty_mm": self.gap_uncertainty,
            "velocity_mm_s": self.gap_velocity,
            "erosion_rate_mm_per_pulse": self.erosion_rate,
            "confidence": self.confidence,
        }


class GapEstimator:
    """
    Stimatore di Kalman per il gap inter-elettrodo.

    Modello di stato a 3 dimensioni:
        x = [gap, gap_dot, erosion_rate]ᵀ

    Dinamica:
        gap[k+1] = gap[k] + gap_dot[k]*dt + erosion_rate[k] + u[k]
        gap_dot[k+1] = gap_dot[k]
        erosion_rate[k+1] = erosion_rate[k]

    dove u[k] è il comando del motore.

    Misure:
        z1 = gap (da resistenza plasma, rumorosa)
        z2 = gap (da encoder motore, precisa)

    Esempio:
        >>> estimator = GapEstimator()
        >>> for i in range(1000):
        ...     estimate = estimator.update(
        ...         gap_from_resistance=Q_(5.1, "mm"),
        ...         motor_command=Q_(0.01, "mm")
        ...     )
        >>> print(f"Gap: {estimate.gap} ± {estimate.gap_uncertainty} mm")
    """

    def __init__(self, config: EstimatorConfig = None):
        """
        Inizializza il filtro di Kalman.

        Parametri:
            config: Configurazione del filtro
        """
        self.config = config or EstimatorConfig()

        # Stato: [gap, gap_dot, erosion_rate]
        gap_init = self.config.gap_initial.to("mm").magnitude
        self._x = np.array([gap_init, 0.0, 1e-6])  # State vector

        # Covarianza iniziale
        self._P = np.diag([
            self.config.gap_uncertainty ** 2,
            0.1,      # Incertezza velocità iniziale
            1e-10,    # Incertezza erosione iniziale
        ])

        # Matrice di transizione stato (aggiornata in predict)
        dt = self.config.dt
        self._F = np.array([
            [1, dt, 1],    # gap = gap + gap_dot*dt + erosion
            [0, 1,  0],    # gap_dot costante
            [0, 0,  1],    # erosion_rate costante
        ])

        # Matrice di controllo (movimento motore)
        self._B = np.array([1, 0, 0])  # Solo gap influenzato

        # Rumore di processo
        self._Q = np.diag([
            self.config.process_noise_gap,
            self.config.process_noise_velocity,
            self.config.process_noise_erosion,
        ])

        # Matrici di osservazione
        self._H_resistance = np.array([[1, 0, 0]])  # Misura gap
        self._H_encoder = np.array([[1, 0, 0]])     # Misura gap

        # Rumore di misura
        self._R_resistance = np.array([[self.config.measurement_noise_resistance]])
        self._R_encoder = np.array([[self.config.measurement_noise_encoder]])

        # Storico per diagnostica
        self._history: List[GapEstimate] = []
        self._impulse_count = 0

    def predict(self, motor_command: "Q_" = None) -> GapEstimate:
        """
        Fase di predizione del filtro di Kalman.

        Parametri:
            motor_command: Comando motore (variazione gap in mm)

        Ritorna:
            GapEstimate con predizione a priori
        """
        # Input di controllo
        u = 0.0
        if motor_command is not None:
            u = motor_command.to("mm").magnitude

        # Predizione stato: x_pred = F*x + B*u
        self._x = self._F @ self._x + self._B * u

        # Limita gap a valori fisici
        self._x[0] = np.clip(self._x[0], 1.0, 20.0)

        # Predizione covarianza: P_pred = F*P*F' + Q
        self._P = self._F @ self._P @ self._F.T + self._Q

        return self._create_estimate()

    def update(
        self,
        gap_from_resistance: "Q_" = None,
        gap_from_encoder: "Q_" = None,
        motor_command: "Q_" = None,
    ) -> GapEstimate:
        """
        Ciclo completo: predizione + aggiornamento.

        Parametri:
            gap_from_resistance: Misura gap da resistenza plasma (mm)
            gap_from_encoder: Misura gap da encoder motore (mm)
            motor_command: Comando motore (mm)

        Ritorna:
            GapEstimate aggiornato
        """
        # Predizione
        self.predict(motor_command)

        # Aggiornamento con misura da resistenza
        if gap_from_resistance is not None:
            z = np.array([gap_from_resistance.to("mm").magnitude])
            self._kalman_update(z, self._H_resistance, self._R_resistance)

        # Aggiornamento con misura da encoder
        if gap_from_encoder is not None:
            z = np.array([gap_from_encoder.to("mm").magnitude])
            self._kalman_update(z, self._H_encoder, self._R_encoder)

        self._impulse_count += 1

        estimate = self._create_estimate()
        self._history.append(estimate)

        # Limita storico
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return estimate

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Aggiornamento di Kalman con una misura.

        Parametri:
            z: Vettore misura
            H: Matrice di osservazione
            R: Covarianza rumore misura
        """
        # Innovazione: y = z - H*x
        y = z - H @ self._x

        # Covarianza innovazione: S = H*P*H' + R
        S = H @ self._P @ H.T + R

        # Guadagno di Kalman: K = P*H'*S^(-1)
        K = self._P @ H.T @ np.linalg.inv(S)

        # Aggiornamento stato: x = x + K*y
        self._x = self._x + (K @ y).flatten()

        # Limita gap
        self._x[0] = np.clip(self._x[0], 1.0, 20.0)

        # Aggiornamento covarianza: P = (I - K*H)*P
        I = np.eye(len(self._x))
        self._P = (I - K @ H) @ self._P

    def _create_estimate(self) -> GapEstimate:
        """Crea un GapEstimate dallo stato corrente."""
        gap_mm = float(self._x[0])
        gap_uncertainty = float(np.sqrt(self._P[0, 0]))
        gap_velocity = float(self._x[1])
        erosion_rate = float(self._x[2])

        # Confidenza basata sull'incertezza
        # Alta confidenza = bassa incertezza
        confidence = float(np.clip(1.0 - gap_uncertainty / 5.0, 0, 1))

        return GapEstimate(
            gap=Q_(gap_mm, "mm"),
            gap_uncertainty=gap_uncertainty,
            gap_velocity=gap_velocity,
            erosion_rate=erosion_rate,
            confidence=confidence,
        )

    def predict_future(self, steps: int) -> List[GapEstimate]:
        """
        Predice l'evoluzione del gap per N passi futuri.

        Utile per pianificare interventi di manutenzione.

        Parametri:
            steps: Numero di impulsi futuri da predire

        Ritorna:
            Lista di GapEstimate predetti
        """
        predictions = []

        # Salva stato corrente
        x_saved = self._x.copy()
        P_saved = self._P.copy()

        for _ in range(steps):
            # Predizione senza comando motore
            self._x = self._F @ self._x
            self._x[0] = np.clip(self._x[0], 1.0, 20.0)
            self._P = self._F @ self._P @ self._F.T + self._Q

            predictions.append(self._create_estimate())

        # Ripristina stato
        self._x = x_saved
        self._P = P_saved

        return predictions

    def estimate_time_to_maintenance(
        self,
        gap_threshold: "Q_"
    ) -> Optional[int]:
        """
        Stima il numero di impulsi prima della manutenzione.

        Basato sul tasso di erosione stimato.

        Parametri:
            gap_threshold: Gap massimo prima di richiedere manutenzione

        Ritorna:
            Numero di impulsi stimato, None se non raggiungibile
        """
        gap_current = self._x[0]
        erosion_rate = self._x[2]
        gap_max = gap_threshold.to("mm").magnitude

        if erosion_rate <= 0:
            return None

        # Tempo semplificato: (gap_max - gap_current) / erosion_rate
        delta_gap = gap_max - gap_current

        if delta_gap <= 0:
            return 0  # Già oltre la soglia

        impulses = int(delta_gap / erosion_rate)
        return impulses

    def reset(self, gap_initial: "Q_" = None):
        """
        Resetta il filtro allo stato iniziale.

        Parametri:
            gap_initial: Nuovo gap iniziale (opzionale)
        """
        if gap_initial:
            gap_mm = gap_initial.to("mm").magnitude
        else:
            gap_mm = self.config.gap_initial.to("mm").magnitude

        self._x = np.array([gap_mm, 0.0, 1e-6])
        self._P = np.diag([
            self.config.gap_uncertainty ** 2,
            0.1,
            1e-10,
        ])
        self._history.clear()
        self._impulse_count = 0

    @property
    def state(self) -> np.ndarray:
        """Stato corrente [gap, gap_dot, erosion_rate]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Matrice di covarianza corrente."""
        return self._P.copy()

    @property
    def impulse_count(self) -> int:
        """Numero di impulsi processati."""
        return self._impulse_count

    def get_diagnostics(self) -> dict:
        """Ritorna diagnostica del filtro."""
        return {
            "impulse_count": self._impulse_count,
            "state": {
                "gap_mm": float(self._x[0]),
                "gap_velocity_mm_s": float(self._x[1]),
                "erosion_rate_mm_per_pulse": float(self._x[2]),
            },
            "uncertainty": {
                "gap_mm": float(np.sqrt(self._P[0, 0])),
                "velocity_mm_s": float(np.sqrt(self._P[1, 1])),
                "erosion_mm_per_pulse": float(np.sqrt(self._P[2, 2])),
            },
            "history_length": len(self._history),
        }

    def __repr__(self) -> str:
        return (
            f"GapEstimator("
            f"gap={self._x[0]:.3f}mm ± {np.sqrt(self._P[0,0]):.3f}mm, "
            f"impulses={self._impulse_count})"
        )
