# Gestione proprietà materiali
"""
Gestione delle proprietà dei materiali per il simulatore ESWT.

Questo modulo carica e gestisce le proprietà dei materiali da file JSON,
permettendo di personalizzare i parametri senza modificare il codice.

Uso tipico:
    from src.core.materials import MaterialsDatabase

    db = MaterialsDatabase()
    tungsteno = db.get_electrode("tungsteno")
    acqua = db.get_water_solution("acqua_distillata")
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .units import ureg, Q_


# Percorso base per i dati
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "materials"


@dataclass
class ElectrodeMaterial:
    """
    Proprietà di un materiale per elettrodi.

    Attributi:
        nome: Nome del materiale
        densita: Densità (kg/m³)
        punto_fusione: Temperatura di fusione (°C)
        conducibilita_termica: Conducibilità termica (W/(m·K))
        conducibilita_elettrica: Conducibilità elettrica (S/m)
        tasso_erosione: Tasso di erosione empirico (kg/(A·s))
    """

    nome: str
    densita: "Q_"
    punto_fusione: "Q_"
    conducibilita_termica: "Q_"
    conducibilita_elettrica: Optional["Q_"] = None
    tasso_erosione: Optional["Q_"] = None


@dataclass
class WaterSolution:
    """
    Proprietà di una soluzione acquosa.

    Attributi:
        nome: Nome della soluzione
        conducibilita: Conducibilità elettrica (S/m)
        densita: Densità (kg/m³)
        velocita_suono: Velocità del suono (m/s)
        tensione_breakdown: Tensione di rottura dielettrica (kV/mm)
        concentrazione_pd: Concentrazione Palladio (mg/L) - opzionale
    """

    nome: str
    conducibilita: "Q_"
    densita: "Q_"
    velocita_suono: "Q_"
    tensione_breakdown: Optional["Q_"] = None
    concentrazione_pd: Optional["Q_"] = None


class MaterialsDatabase:
    """
    Database delle proprietà dei materiali.

    Carica i dati da file JSON nella cartella data/materials/.
    Se i file non esistono, usa valori di default dalla letteratura.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Inizializza il database dei materiali.

        Parametri:
            data_dir: Directory contenente i file JSON (opzionale)
        """
        self.data_dir = data_dir or DATA_DIR
        self._electrodes: dict[str, ElectrodeMaterial] = {}
        self._water_solutions: dict[str, WaterSolution] = {}
        self._load_defaults()
        self._load_from_files()

    def _load_defaults(self):
        """Carica i valori di default dalla letteratura."""
        # Elettrodi di default
        self._electrodes["tungsteno"] = ElectrodeMaterial(
            nome="Tungsteno",
            densita=Q_(19300, "kg/m^3"),
            punto_fusione=Q_(3422, "degC"),
            conducibilita_termica=Q_(173, "W/(m*K)"),
            conducibilita_elettrica=Q_(1.89e7, "S/m"),
            tasso_erosione=Q_(1e-10, "kg/(A*s)"),  # Valore empirico tipico
        )

        self._electrodes["rame"] = ElectrodeMaterial(
            nome="Rame",
            densita=Q_(8960, "kg/m^3"),
            punto_fusione=Q_(1085, "degC"),
            conducibilita_termica=Q_(401, "W/(m*K)"),
            conducibilita_elettrica=Q_(5.96e7, "S/m"),
            tasso_erosione=Q_(5e-10, "kg/(A*s)"),
        )

        self._electrodes["acciaio_inox"] = ElectrodeMaterial(
            nome="Acciaio Inox",
            densita=Q_(8000, "kg/m^3"),
            punto_fusione=Q_(1400, "degC"),
            conducibilita_termica=Q_(16, "W/(m*K)"),
            conducibilita_elettrica=Q_(1.45e6, "S/m"),
        )

        # Soluzioni acquose di default
        self._water_solutions["acqua_distillata"] = WaterSolution(
            nome="Acqua Distillata",
            conducibilita=Q_(5e-4, "S/m"),  # Molto bassa
            densita=Q_(998.3, "kg/m^3"),
            velocita_suono=Q_(1492, "m/s"),
            tensione_breakdown=Q_(65, "kV/mm"),  # Tipico per acqua pura
        )

        self._water_solutions["acqua_tap"] = WaterSolution(
            nome="Acqua di Rubinetto",
            conducibilita=Q_(0.05, "S/m"),  # Tipica
            densita=Q_(998.3, "kg/m^3"),
            velocita_suono=Q_(1492, "m/s"),
            tensione_breakdown=Q_(30, "kV/mm"),  # Ridotta per impurità
        )

        self._water_solutions["acqua_pd"] = WaterSolution(
            nome="Acqua con Palladio",
            conducibilita=Q_(0.1, "S/m"),  # Aumentata per catalizzatore
            densita=Q_(1000, "kg/m^3"),
            velocita_suono=Q_(1492, "m/s"),
            tensione_breakdown=Q_(25, "kV/mm"),
            concentrazione_pd=Q_(100, "mg/L"),  # Concentrazione tipica
        )

    def _load_from_files(self):
        """Carica proprietà dai file JSON se esistono."""
        electrodes_file = self.data_dir / "electrodes.json"
        water_file = self.data_dir / "water_solutions.json"

        if electrodes_file.exists():
            try:
                with open(electrodes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, props in data.items():
                        self._electrodes[key] = self._parse_electrode(key, props)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Attenzione: errore caricamento {electrodes_file}: {e}")

        if water_file.exists():
            try:
                with open(water_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, props in data.items():
                        self._water_solutions[key] = self._parse_water_solution(key, props)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Attenzione: errore caricamento {water_file}: {e}")

    def _parse_electrode(self, key: str, props: dict) -> ElectrodeMaterial:
        """Converte dati JSON in ElectrodeMaterial."""
        return ElectrodeMaterial(
            nome=props.get("nome", key),
            densita=Q_(props["densita"]["valore"], props["densita"]["unita"]),
            punto_fusione=Q_(props["punto_fusione"]["valore"], props["punto_fusione"]["unita"]),
            conducibilita_termica=Q_(
                props["conducibilita_termica"]["valore"],
                props["conducibilita_termica"]["unita"],
            ),
            conducibilita_elettrica=(
                Q_(
                    props["conducibilita_elettrica"]["valore"],
                    props["conducibilita_elettrica"]["unita"],
                )
                if "conducibilita_elettrica" in props
                else None
            ),
            tasso_erosione=(
                Q_(props["tasso_erosione"]["valore"], props["tasso_erosione"]["unita"])
                if "tasso_erosione" in props
                else None
            ),
        )

    def _parse_water_solution(self, key: str, props: dict) -> WaterSolution:
        """Converte dati JSON in WaterSolution."""
        return WaterSolution(
            nome=props.get("nome", key),
            conducibilita=Q_(props["conducibilita"]["valore"], props["conducibilita"]["unita"]),
            densita=Q_(props["densita"]["valore"], props["densita"]["unita"]),
            velocita_suono=Q_(props["velocita_suono"]["valore"], props["velocita_suono"]["unita"]),
            tensione_breakdown=(
                Q_(props["tensione_breakdown"]["valore"], props["tensione_breakdown"]["unita"])
                if "tensione_breakdown" in props
                else None
            ),
            concentrazione_pd=(
                Q_(props["concentrazione_pd"]["valore"], props["concentrazione_pd"]["unita"])
                if "concentrazione_pd" in props
                else None
            ),
        )

    def get_electrode(self, nome: str) -> ElectrodeMaterial:
        """
        Ottiene le proprietà di un materiale per elettrodi.

        Parametri:
            nome: Nome del materiale (es. "tungsteno", "rame")

        Ritorna:
            ElectrodeMaterial con le proprietà

        Solleva:
            KeyError se il materiale non è trovato
        """
        if nome not in self._electrodes:
            disponibili = list(self._electrodes.keys())
            raise KeyError(f"Elettrodo '{nome}' non trovato. Disponibili: {disponibili}")
        return self._electrodes[nome]

    def get_water_solution(self, nome: str) -> WaterSolution:
        """
        Ottiene le proprietà di una soluzione acquosa.

        Parametri:
            nome: Nome della soluzione (es. "acqua_distillata", "acqua_pd")

        Ritorna:
            WaterSolution con le proprietà

        Solleva:
            KeyError se la soluzione non è trovata
        """
        if nome not in self._water_solutions:
            disponibili = list(self._water_solutions.keys())
            raise KeyError(f"Soluzione '{nome}' non trovata. Disponibili: {disponibili}")
        return self._water_solutions[nome]

    def list_electrodes(self) -> list[str]:
        """Ritorna la lista dei materiali per elettrodi disponibili."""
        return list(self._electrodes.keys())

    def list_water_solutions(self) -> list[str]:
        """Ritorna la lista delle soluzioni acquose disponibili."""
        return list(self._water_solutions.keys())
