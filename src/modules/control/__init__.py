# Modulo Control Logic
"""
Logica di controllo per dispositivi ESWT.

Questo modulo implementa:
    - Controllore PID per gap elettrodi
    - Monitoraggio feedback elettrico
    - Stimatore gap con filtro di Kalman
    - Log interventi motore

Moduli:
    - data_structures: Strutture dati (ControlState, MotorAction, etc.)
    - pid_controller: Controllore PID per gap
    - feedback_monitor: Monitor feedback elettrico
    - gap_estimator: Filtro di Kalman per stima gap
    - motor_log: Logger interventi motore

Esempio d'uso:
    >>> from modules.control import PIDGapController, ElectricalFeedbackMonitor
    >>> controller = PIDGapController()
    >>> monitor = ElectricalFeedbackMonitor()
    >>> feedback = monitor.measure(discharge_result)
    >>> new_gap, action = controller.step(feedback.I_picco, current_gap, impulse)
"""

from .data_structures import (
    ControlMode,
    ControlStatus,
    ControlState,
    MotorAction,
    FeedbackMeasurement,
    PIDParameters,
)

from .pid_controller import PIDGapController

from .feedback_monitor import (
    FeedbackConfig,
    ElectricalFeedbackMonitor,
)

from .gap_estimator import (
    EstimatorConfig,
    GapEstimate,
    GapEstimator,
)

from .motor_log import (
    LogEntry,
    SessionSummary,
    MotorInterventionLogger,
)

__all__ = [
    # Data structures
    "ControlMode",
    "ControlStatus",
    "ControlState",
    "MotorAction",
    "FeedbackMeasurement",
    "PIDParameters",
    # PID Controller
    "PIDGapController",
    # Feedback Monitor
    "FeedbackConfig",
    "ElectricalFeedbackMonitor",
    # Gap Estimator
    "EstimatorConfig",
    "GapEstimate",
    "GapEstimator",
    # Motor Log
    "LogEntry",
    "SessionSummary",
    "MotorInterventionLogger",
]
