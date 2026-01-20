# Modello assorbimento gas da palladio per ESWT
"""
Modello per l'assorbimento di gas (principalmente H₂) da parte del palladio.

Il palladio è un assorbitore di idrogeno noto. Nei sistemi ESWT:
    - L'elettrolisi produce H₂ e O₂
    - Le bolle di gas influenzano la cavitazione
    - Il palladio può assorbire H₂, riducendo la densità di bolle

Questo modulo implementa:
    - Modello cinetico di assorbimento H₂
    - Effetti sulla popolazione di bolle
    - Impatto sulla cavitazione

Riferimenti:
    - Prompt.md sezione 3
    - Alefeld & Völkl (1978) - Hydrogen in Metals
    - Manchester et al. (1994) - Pd-H System
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from ...core.units import ureg, Q_


@dataclass
class GasAbsorptionState:
    """
    Stato del sistema di assorbimento gas.

    Attributi:
        concentrazione_pd: Concentrazione palladio (mg/L)
        H2_assorbito: Moli H₂ assorbito per grammo Pd
        H2_saturo: Rapporto saturazione (0-1)
        densita_bolle: Densità bolle gas (#/mL)
        volume_bolle_totale: Volume totale bolle (μL/L)
    """

    concentrazione_pd: "Q_"         # mg/L
    H2_assorbito: float             # mol H₂ / g Pd
    H2_saturo: float                # Rapporto saturazione (0-1)
    densita_bolle: "Q_"             # #/mL
    volume_bolle_totale: "Q_"       # μL/L

    def to_dict(self) -> dict:
        return {
            "concentrazione_pd_mg_L": self.concentrazione_pd.to("mg/L").magnitude,
            "H2_assorbito_mol_per_g": self.H2_assorbito,
            "H2_saturo_frac": self.H2_saturo,
            "densita_bolle_per_mL": self.densita_bolle.to("1/mL").magnitude,
            "volume_bolle_uL_per_L": self.volume_bolle_totale.to("uL/L").magnitude,
        }


@dataclass
class GasAbsorptionConfig:
    """
    Configurazione del modello di assorbimento gas.

    Attributi:
        H2_max_per_Pd: Rapporto massimo H/Pd (adim., tipico ~0.7)
        k_assorbimento: Costante cinetica assorbimento (1/s)
        k_generazione_H2: Tasso generazione H₂ per impulso (μmol)
        raggio_bolla_medio: Raggio medio bolle (μm)
        densita_bolle_iniziale: Densità bolle iniziale (#/mL)
    """

    H2_max_per_Pd: float = 0.7           # Rapporto atomico H/Pd max
    k_assorbimento: float = 0.01         # 1/s - cinetica assorbimento
    k_generazione_H2: float = 0.1        # μmol H₂ per impulso
    raggio_bolla_medio: "Q_" = None      # μm
    densita_bolle_iniziale: "Q_" = None  # #/mL

    def __post_init__(self):
        if self.raggio_bolla_medio is None:
            self.raggio_bolla_medio = Q_(50, "um")  # 50 μm tipico
        if self.densita_bolle_iniziale is None:
            self.densita_bolle_iniziale = Q_(100, "1/mL")


class GasAbsorptionModel:
    """
    Modello per l'assorbimento di gas da parte del palladio in ESWT.

    Il palladio assorbe idrogeno con una cinetica:
        dC_H/dt = k · (C_H_max - C_H) · P_H2

    dove:
        - C_H = concentrazione H nel Pd
        - C_H_max = concentrazione a saturazione
        - P_H2 = pressione parziale H₂
        - k = costante cinetica

    L'assorbimento riduce la densità di bolle di H₂,
    che a sua volta influenza la cavitazione.

    Esempio:
        >>> model = GasAbsorptionModel()
        >>> state = model.get_initial_state(conc_pd=Q_(5, "mg/L"))
        >>> state = model.update_after_pulse(state, n_bolle_generate=1000)
    """

    # Costanti
    M_Pd = 106.42   # g/mol - massa molare palladio
    M_H2 = 2.016    # g/mol - massa molare H₂
    V_MOLARE_GAS = 24.5  # L/mol a 25°C, 1 atm

    def __init__(self, config: GasAbsorptionConfig = None):
        """
        Inizializza il modello.

        Parametri:
            config: Configurazione del modello
        """
        self.config = config or GasAbsorptionConfig()

        # Storico per analisi
        self._history: List[GasAbsorptionState] = []

    def get_initial_state(
        self,
        concentrazione_pd: "Q_" = None
    ) -> GasAbsorptionState:
        """
        Crea lo stato iniziale del sistema.

        Parametri:
            concentrazione_pd: Concentrazione Pd nel sistema (mg/L)

        Ritorna:
            GasAbsorptionState iniziale
        """
        if concentrazione_pd is None:
            concentrazione_pd = Q_(0, "mg/L")

        # Volume iniziale bolle
        n_bolle = self.config.densita_bolle_iniziale.to("1/mL").magnitude
        r = self.config.raggio_bolla_medio.to("um").magnitude
        V_bolla = (4/3) * np.pi * (r * 1e-4)**3  # cm³
        V_totale = n_bolle * V_bolla * 1000  # μL/L

        return GasAbsorptionState(
            concentrazione_pd=concentrazione_pd,
            H2_assorbito=0.0,
            H2_saturo=0.0,
            densita_bolle=self.config.densita_bolle_iniziale,
            volume_bolle_totale=Q_(V_totale, "uL/L"),
        )

    def update_after_pulse(
        self,
        state: GasAbsorptionState,
        H2_generato: "Q_" = None,
        tempo_inter_impulso: "Q_" = None
    ) -> GasAbsorptionState:
        """
        Aggiorna lo stato dopo un impulso.

        L'impulso:
        1. Genera H₂ per elettrolisi
        2. Il Pd assorbe parte dell'H₂
        3. L'H₂ rimanente forma bolle

        Parametri:
            state: Stato corrente
            H2_generato: Moli H₂ generate (default da config)
            tempo_inter_impulso: Tempo disponibile per assorbimento (s)

        Ritorna:
            Nuovo GasAbsorptionState
        """
        # Default H₂ generato
        if H2_generato is None:
            H2_generato = Q_(self.config.k_generazione_H2, "umol")

        if tempo_inter_impulso is None:
            tempo_inter_impulso = Q_(1, "ms")

        # Parametri correnti
        c_pd = state.concentrazione_pd.to("mg/L").magnitude
        H2_ass = state.H2_assorbito
        n_bolle = state.densita_bolle.to("1/mL").magnitude

        # Massa Pd per litro
        m_pd_g = c_pd / 1000  # g/L

        # 1. Calcola capacità residua di assorbimento
        # Max moli H per grammo Pd
        H_max = self.config.H2_max_per_Pd / self.M_Pd  # mol H per g Pd
        H2_max = H_max / 2  # mol H₂ per g Pd

        capacita_residua = max(0, H2_max - H2_ass)  # mol H₂/g Pd

        # 2. Calcola assorbimento in questo passo
        n_H2_generato = H2_generato.to("mol").magnitude
        dt = tempo_inter_impulso.to("s").magnitude

        # Cinetica primo ordine
        k = self.config.k_assorbimento
        H2_disponibile = n_H2_generato  # per L

        if m_pd_g > 0 and capacita_residua > 0:
            # Assorbimento: dH/dt = k * (H_max - H) * [H2]
            tasso_ass = k * capacita_residua * H2_disponibile
            H2_assorbito_step = tasso_ass * dt * m_pd_g

            # Non può assorbire più di quanto generato o capacità residua
            H2_assorbito_step = min(
                H2_assorbito_step,
                H2_disponibile,
                capacita_residua * m_pd_g
            )
        else:
            H2_assorbito_step = 0

        # 3. H₂ che forma bolle
        H2_bolle = max(0, n_H2_generato - H2_assorbito_step)

        # 4. Calcola nuove bolle
        # Volume gas a condizioni standard
        V_gas = H2_bolle * self.V_MOLARE_GAS * 1e6  # μL

        r_bolla = self.config.raggio_bolla_medio.to("um").magnitude
        V_bolla = (4/3) * np.pi * r_bolla**3  # μm³ = 10^-15 L = 10^-9 μL

        if V_bolla > 0:
            n_nuove_bolle = V_gas / (V_bolla * 1e-9)  # per L → per mL
            n_nuove_bolle /= 1000
        else:
            n_nuove_bolle = 0

        # 5. Aggiorna densità bolle (con dissoluzione naturale)
        k_dissoluzione = 0.001  # Tasso dissoluzione spontanea
        n_bolle_new = n_bolle * (1 - k_dissoluzione) + n_nuove_bolle

        # 6. Aggiorna stato assorbimento
        if m_pd_g > 0:
            H2_ass_new = H2_ass + H2_assorbito_step / m_pd_g
            H2_saturo_new = H2_ass_new / H2_max if H2_max > 0 else 0
        else:
            H2_ass_new = H2_ass
            H2_saturo_new = state.H2_saturo

        # Calcola volume bolle totale
        V_bolle_new = n_bolle_new * (4/3) * np.pi * (r_bolla * 1e-4)**3 * 1e6

        new_state = GasAbsorptionState(
            concentrazione_pd=state.concentrazione_pd,
            H2_assorbito=float(H2_ass_new),
            H2_saturo=float(np.clip(H2_saturo_new, 0, 1)),
            densita_bolle=Q_(n_bolle_new, "1/mL"),
            volume_bolle_totale=Q_(V_bolle_new, "uL/L"),
        )

        self._history.append(new_state)

        return new_state

    def calcola_frazione_bolle_assorbite(
        self,
        concentrazione_pd: "Q_",
        n_impulsi: int,
        saturazione_pd: float = 0.0
    ) -> float:
        """
        Calcola la frazione di bolle assorbite dal Pd.

        Parametri:
            concentrazione_pd: Concentrazione Pd (mg/L)
            n_impulsi: Numero di impulsi
            saturazione_pd: Livello saturazione Pd (0-1)

        Ritorna:
            Frazione bolle assorbite (0-1)
        """
        c_pd = concentrazione_pd.to("mg/L").magnitude

        if c_pd <= 0:
            return 0.0

        # Efficienza assorbimento decresce con saturazione
        eta_sat = 1 - saturazione_pd

        # Efficienza base (empirica)
        eta_base = 0.1  # 10% max

        # Dipendenza da concentrazione (saturazione cinetica)
        K_m = 10  # mg/L - costante mezza saturazione
        eta_conc = c_pd / (K_m + c_pd)

        # Frazione totale
        f_assorbite = eta_base * eta_sat * eta_conc

        return float(np.clip(f_assorbite, 0, 1))

    def effetto_bolle_su_cavitazione(
        self,
        densita_bolle: "Q_",
        densita_riferimento: "Q_" = None
    ) -> float:
        """
        Calcola l'effetto della densità di bolle sulla cavitazione.

        Più bolle di gas = più nuclei di cavitazione
        → soglia cavitazione più bassa
        → più cavitazione a parità di pressione

        Modello:
            η_cav = 1 + k · (n_bolle / n_ref - 1)

        dove η_cav > 1 significa più cavitazione.

        Parametri:
            densita_bolle: Densità attuale bolle (#/mL)
            densita_riferimento: Densità di riferimento (#/mL)

        Ritorna:
            Fattore moltiplicativo cavitazione (1.0 = nominale)
        """
        if densita_riferimento is None:
            densita_riferimento = self.config.densita_bolle_iniziale

        n = densita_bolle.to("1/mL").magnitude
        n_ref = densita_riferimento.to("1/mL").magnitude

        if n_ref <= 0:
            return 1.0

        # Coefficiente empirico
        k = 0.5

        ratio = n / n_ref
        eta_cav = 1 + k * (ratio - 1)

        # Limita a range ragionevole
        eta_cav = np.clip(eta_cav, 0.5, 2.0)

        return float(eta_cav)

    def soglia_cavitazione_modificata(
        self,
        soglia_base: "Q_",
        densita_bolle: "Q_"
    ) -> "Q_":
        """
        Calcola la soglia di cavitazione modificata dalle bolle.

        Più bolle → nucleazione facilitata → soglia meno negativa

        Parametri:
            soglia_base: Soglia cavitazione acqua pura (MPa, neg.)
            densita_bolle: Densità bolle (#/mL)

        Ritorna:
            Soglia cavitazione modificata (MPa)
        """
        P_base = soglia_base.to("MPa").magnitude  # Negativo
        n = densita_bolle.to("1/mL").magnitude

        n_ref = self.config.densita_bolle_iniziale.to("1/mL").magnitude

        if n_ref <= 0:
            return soglia_base

        # Più bolle → soglia meno negativa (più facile cavitare)
        # Modello: P_cav = P_base / (1 + k·log(n/n_ref))
        ratio = max(n / n_ref, 0.01)
        k = 0.2

        if ratio >= 1:
            P_cav = P_base / (1 + k * np.log(ratio))
        else:
            P_cav = P_base * (1 - k * np.log(1/ratio))

        return Q_(P_cav, "MPa")

    def stima_tempo_saturazione(
        self,
        concentrazione_pd: "Q_",
        H2_per_impulso: "Q_" = None,
        frequenza_impulsi: "Q_" = None
    ) -> int:
        """
        Stima il numero di impulsi per saturare il Pd.

        Parametri:
            concentrazione_pd: Concentrazione Pd (mg/L)
            H2_per_impulso: H₂ generato per impulso
            frequenza_impulsi: Frequenza impulsi (Hz)

        Ritorna:
            Numero stimato di impulsi
        """
        c_pd = concentrazione_pd.to("mg/L").magnitude

        if c_pd <= 0:
            return 0

        if H2_per_impulso is None:
            H2_per_impulso = Q_(self.config.k_generazione_H2, "umol")

        # Capacità totale Pd
        m_pd = c_pd / 1000  # g/L
        H2_max = self.config.H2_max_per_Pd / self.M_Pd / 2 * m_pd  # mol H₂/L

        # H₂ assorbibile per impulso (assumendo efficienza 10%)
        H2_impulso = H2_per_impulso.to("mol").magnitude
        H2_ass_per_impulso = H2_impulso * 0.1

        if H2_ass_per_impulso <= 0:
            return float('inf')

        n_impulsi = int(H2_max / H2_ass_per_impulso)

        return max(n_impulsi, 1)

    def get_statistics(self) -> dict:
        """Ritorna statistiche sullo storico."""
        if not self._history:
            return {"message": "Nessun dato disponibile"}

        saturazioni = [s.H2_saturo for s in self._history]
        bolle = [s.densita_bolle.to("1/mL").magnitude for s in self._history]

        return {
            "n_samples": len(self._history),
            "saturazione_pd": {
                "iniziale": saturazioni[0] if saturazioni else 0,
                "finale": saturazioni[-1] if saturazioni else 0,
                "max": max(saturazioni) if saturazioni else 0,
            },
            "densita_bolle": {
                "iniziale_per_mL": bolle[0] if bolle else 0,
                "finale_per_mL": bolle[-1] if bolle else 0,
                "min_per_mL": min(bolle) if bolle else 0,
                "max_per_mL": max(bolle) if bolle else 0,
            },
        }

    def reset(self):
        """Resetta lo storico."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"GasAbsorptionModel("
            f"H/Pd_max={self.config.H2_max_per_Pd})"
        )
