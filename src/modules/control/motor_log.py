# Logger degli interventi del motore di avvicinamento ESWT
"""
Modulo per il logging degli interventi del motore di avvicinamento.

Questo modulo implementa:
    - MotorInterventionLogger: Registrazione completa interventi
    - Export in formato CSV e JSON
    - Report di sintesi e statistiche

Output richiesto da prompt.md:
    "log degli interventi del motore di avvicinamento"

Riferimenti:
    - Prompt.md sezione 2.D
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv

from ...core.units import ureg, Q_
from .data_structures import MotorAction, ControlStatus


@dataclass
class LogEntry:
    """
    Entry singola nel log del motore.

    Estende MotorAction con metadati aggiuntivi per il logging.
    """

    action: MotorAction
    sequence_number: int
    session_id: str
    notes: str = ""


@dataclass
class SessionSummary:
    """
    Riepilogo di una sessione di trattamento.

    Attributi:
        session_id: Identificatore sessione
        start_time: Inizio sessione
        end_time: Fine sessione
        total_impulses: Numero totale impulsi
        total_interventions: Numero interventi motore
        gap_initial: Gap iniziale (mm)
        gap_final: Gap finale (mm)
        gap_delta_total: Variazione gap totale (mm)
        avg_current: Corrente media (kA)
        intervention_rate: Percentuale impulsi con intervento
    """

    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_impulses: int
    total_interventions: int
    gap_initial: "Q_"
    gap_final: "Q_"
    gap_delta_total: "Q_"
    avg_current: "Q_"
    intervention_rate: float


class MotorInterventionLogger:
    """
    Logger per interventi del motore di avvicinamento.

    Registra ogni intervento del motore che regola il gap
    inter-elettrodo basandosi sul feedback elettrico.

    Funzionalità:
    - Registrazione interventi con timestamp
    - Export CSV per analisi spreadsheet
    - Export JSON per integrazione software
    - Report statistiche e trend

    Esempio:
        >>> logger = MotorInterventionLogger()
        >>> logger.log_intervention(motor_action)
        >>> logger.export_csv("motor_log.csv")
        >>> summary = logger.get_session_summary()
    """

    def __init__(self, session_id: str = None):
        """
        Inizializza il logger.

        Parametri:
            session_id: ID sessione (generato automaticamente se non fornito)
        """
        self._session_id = session_id or self._generate_session_id()
        self._entries: List[LogEntry] = []
        self._start_time = datetime.now()
        self._sequence = 0

        # Statistiche running
        self._total_delta_gap = Q_(0, "mm")
        self._gap_history: List[float] = []
        self._current_history: List[float] = []

    def _generate_session_id(self) -> str:
        """Genera un ID sessione univoco."""
        return f"ESWT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_intervention(
        self,
        action: MotorAction,
        notes: str = ""
    ) -> LogEntry:
        """
        Registra un intervento del motore.

        Parametri:
            action: Azione del motore da registrare
            notes: Note aggiuntive opzionali

        Ritorna:
            LogEntry creata
        """
        self._sequence += 1

        entry = LogEntry(
            action=action,
            sequence_number=self._sequence,
            session_id=self._session_id,
            notes=notes,
        )

        self._entries.append(entry)

        # Aggiorna statistiche
        self._total_delta_gap += action.delta_gap
        self._gap_history.append(action.gap_after.to("mm").magnitude)
        self._current_history.append(action.current_feedback.to("kA").magnitude)

        return entry

    def export_csv(self, filename: str) -> Path:
        """
        Esporta il log in formato CSV.

        Formato CSV:
            seq,impulse,timestamp,gap_before_mm,gap_after_mm,delta_gap_mm,
            I_feedback_kA,error_A,pid_p,pid_i,pid_d,status,notes

        Parametri:
            filename: Nome file di output

        Ritorna:
            Path del file creato
        """
        filepath = Path(filename)

        # Assicura estensione .csv
        if filepath.suffix.lower() != ".csv":
            filepath = filepath.with_suffix(".csv")

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "sequence",
                "impulse",
                "timestamp",
                "gap_before_mm",
                "gap_after_mm",
                "delta_gap_mm",
                "I_feedback_kA",
                "error_A",
                "pid_p",
                "pid_i",
                "pid_d",
                "status",
                "notes",
            ])

            # Dati
            for entry in self._entries:
                a = entry.action
                writer.writerow([
                    entry.sequence_number,
                    a.impulse_num,
                    a.timestamp.isoformat(),
                    f"{a.gap_before.to('mm').magnitude:.4f}",
                    f"{a.gap_after.to('mm').magnitude:.4f}",
                    f"{a.delta_gap.to('mm').magnitude:.6f}",
                    f"{a.current_feedback.to('kA').magnitude:.3f}",
                    f"{a.error_signal.to('A').magnitude:.1f}",
                    f"{a.pid_output[0]:.6f}",
                    f"{a.pid_output[1]:.6f}",
                    f"{a.pid_output[2]:.6f}",
                    a.status,
                    entry.notes,
                ])

        return filepath

    def export_json(self, filename: str, indent: int = 2) -> Path:
        """
        Esporta il log in formato JSON.

        Struttura JSON:
        {
            "session_id": "...",
            "start_time": "...",
            "export_time": "...",
            "total_entries": N,
            "summary": {...},
            "entries": [...]
        }

        Parametri:
            filename: Nome file di output
            indent: Indentazione JSON (default 2)

        Ritorna:
            Path del file creato
        """
        filepath = Path(filename)

        # Assicura estensione .json
        if filepath.suffix.lower() != ".json":
            filepath = filepath.with_suffix(".json")

        # Costruisci struttura JSON
        data = {
            "session_id": self._session_id,
            "start_time": self._start_time.isoformat(),
            "export_time": datetime.now().isoformat(),
            "total_entries": len(self._entries),
            "summary": self.summary_report(),
            "entries": [self._entry_to_dict(e) for e in self._entries],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return filepath

    def _entry_to_dict(self, entry: LogEntry) -> Dict[str, Any]:
        """Converte un LogEntry in dizionario."""
        return {
            "sequence": entry.sequence_number,
            "session_id": entry.session_id,
            "notes": entry.notes,
            **entry.action.to_dict(),
        }

    def summary_report(self) -> Dict[str, Any]:
        """
        Genera un report di sintesi della sessione.

        Ritorna:
            Dizionario con statistiche riassuntive
        """
        if not self._entries:
            return {
                "session_id": self._session_id,
                "total_interventions": 0,
                "message": "Nessun intervento registrato",
            }

        # Estrai dati per statistiche
        gaps_before = [e.action.gap_before.to("mm").magnitude for e in self._entries]
        gaps_after = [e.action.gap_after.to("mm").magnitude for e in self._entries]
        delta_gaps = [e.action.delta_gap.to("mm").magnitude for e in self._entries]
        currents = [e.action.current_feedback.to("kA").magnitude for e in self._entries]
        errors = [e.action.error_signal.to("A").magnitude for e in self._entries]

        # Conta stati
        status_counts = {}
        for entry in self._entries:
            status = entry.action.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Calcola statistiche
        import numpy as np

        return {
            "session_id": self._session_id,
            "start_time": self._start_time.isoformat(),
            "total_interventions": len(self._entries),
            "first_impulse": self._entries[0].action.impulse_num,
            "last_impulse": self._entries[-1].action.impulse_num,
            "gap_statistics": {
                "initial_mm": gaps_before[0] if gaps_before else 0,
                "final_mm": gaps_after[-1] if gaps_after else 0,
                "delta_total_mm": sum(delta_gaps),
                "delta_mean_mm": float(np.mean(delta_gaps)),
                "delta_std_mm": float(np.std(delta_gaps)),
                "delta_max_mm": max(delta_gaps),
                "delta_min_mm": min(delta_gaps),
            },
            "current_statistics": {
                "mean_kA": float(np.mean(currents)),
                "std_kA": float(np.std(currents)),
                "max_kA": max(currents),
                "min_kA": min(currents),
            },
            "error_statistics": {
                "mean_A": float(np.mean(errors)),
                "std_A": float(np.std(errors)),
                "mae_A": float(np.mean(np.abs(errors))),  # Mean Absolute Error
            },
            "status_distribution": status_counts,
        }

    def get_session_summary(self) -> SessionSummary:
        """
        Ritorna il riepilogo strutturato della sessione.

        Ritorna:
            SessionSummary con dati aggregati
        """
        if not self._entries:
            return SessionSummary(
                session_id=self._session_id,
                start_time=self._start_time,
                end_time=datetime.now(),
                total_impulses=0,
                total_interventions=0,
                gap_initial=Q_(0, "mm"),
                gap_final=Q_(0, "mm"),
                gap_delta_total=Q_(0, "mm"),
                avg_current=Q_(0, "kA"),
                intervention_rate=0.0,
            )

        first_entry = self._entries[0]
        last_entry = self._entries[-1]

        total_impulses = last_entry.action.impulse_num
        intervention_rate = (
            len(self._entries) / total_impulses * 100
            if total_impulses > 0
            else 0.0
        )

        import numpy as np

        return SessionSummary(
            session_id=self._session_id,
            start_time=self._start_time,
            end_time=last_entry.action.timestamp,
            total_impulses=total_impulses,
            total_interventions=len(self._entries),
            gap_initial=first_entry.action.gap_before,
            gap_final=last_entry.action.gap_after,
            gap_delta_total=self._total_delta_gap,
            avg_current=Q_(float(np.mean(self._current_history)), "kA"),
            intervention_rate=intervention_rate,
        )

    def get_gap_evolution(self) -> Dict[str, List]:
        """
        Ritorna l'evoluzione del gap per plotting.

        Ritorna:
            Dizionario con liste di impulsi e gap
        """
        impulses = [e.action.impulse_num for e in self._entries]
        gaps = self._gap_history.copy()

        return {
            "impulses": impulses,
            "gaps_mm": gaps,
        }

    def get_intervention_events(self) -> List[Dict[str, Any]]:
        """
        Ritorna eventi di intervento per timeline.

        Ritorna:
            Lista di eventi con timestamp e dettagli
        """
        events = []
        for entry in self._entries:
            a = entry.action
            events.append({
                "impulse": a.impulse_num,
                "timestamp": a.timestamp.isoformat(),
                "delta_mm": a.delta_gap.to("mm").magnitude,
                "direction": "+" if a.delta_gap.magnitude > 0 else "-",
                "status": a.status,
            })
        return events

    def filter_by_status(self, status: str) -> List[LogEntry]:
        """
        Filtra entries per stato.

        Parametri:
            status: Stato da filtrare ("OK", "LIMITE_MAX", etc.)

        Ritorna:
            Lista di LogEntry con lo stato specificato
        """
        return [e for e in self._entries if e.action.status == status]

    def filter_by_impulse_range(
        self,
        start: int,
        end: int
    ) -> List[LogEntry]:
        """
        Filtra entries per range di impulsi.

        Parametri:
            start: Impulso iniziale (incluso)
            end: Impulso finale (incluso)

        Ritorna:
            Lista di LogEntry nel range
        """
        return [
            e for e in self._entries
            if start <= e.action.impulse_num <= end
        ]

    def clear(self):
        """Cancella tutti i log."""
        self._entries.clear()
        self._sequence = 0
        self._total_delta_gap = Q_(0, "mm")
        self._gap_history.clear()
        self._current_history.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"MotorInterventionLogger("
            f"session='{self._session_id}', "
            f"entries={len(self._entries)})"
        )
