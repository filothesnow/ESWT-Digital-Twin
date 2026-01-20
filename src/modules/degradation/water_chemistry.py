# Modello chimica dell'acqua per ESWT
"""
Modello per la degradazione chimica dell'acqua nel sistema ESWT.

Questo modulo implementa:
    - Accumulo detriti metallici da erosione elettrodi
    - Variazione conducibilità elettrica
    - Effetti sulla formazione del plasma
    - Variazione pH nel tempo

Fenomeni modellati:
    1. Accumulo ossidi/detriti → aumento conducibilità
    2. Prodotti elettrolisi → variazione pH
    3. Conducibilità → energia necessaria per breakdown

Riferimenti:
    - Prompt.md sezioni 2.C e 3
    - Chen et al. (2010) - PAED water chemistry
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np

from ...core.units import ureg, Q_


@dataclass
class WaterChemistryState:
    """
    Stato chimico dell'acqua nel sistema ESWT.

    Attributi:
        concentrazione_detriti: Concentrazione detriti metallici (mg/L)
        concentrazione_ossidi: Concentrazione ossidi metallici (mg/L)
        conducibilita: Conducibilità elettrica (μS/cm)
        ph: pH dell'acqua
        temperatura: Temperatura dell'acqua (°C)
        volume_totale: Volume del bagno d'acqua (L)
        numero_impulsi: Impulsi erogati
    """

    concentrazione_detriti: "Q_"     # mg/L
    concentrazione_ossidi: "Q_"      # mg/L
    conducibilita: "Q_"              # μS/cm
    ph: float                         # adimensionale
    temperatura: "Q_"                 # °C
    volume_totale: "Q_"              # L
    numero_impulsi: int = 0

    def to_dict(self) -> dict:
        return {
            "concentrazione_detriti_mg_L": self.concentrazione_detriti.to("mg/L").magnitude,
            "concentrazione_ossidi_mg_L": self.concentrazione_ossidi.to("mg/L").magnitude,
            "conducibilita_uS_cm": self.conducibilita.to("uS/cm").magnitude,
            "ph": self.ph,
            "temperatura_C": self.temperatura.to("degC").magnitude,
            "volume_L": self.volume_totale.to("L").magnitude,
            "numero_impulsi": self.numero_impulsi,
        }


@dataclass
class WaterChemistryConfig:
    """
    Configurazione del modello chimica acqua.

    Attributi:
        conducibilita_base: Conducibilità acqua pura (μS/cm)
        k_conducibilita: Coefficiente conducibilità vs detriti (μS·L/(cm·mg))
        k_ossidazione: Frazione detriti che si ossida (0-1)
        k_ph_drift: Coefficiente deriva pH per impulso
        volume_acqua: Volume del bagno (L)
        temperatura_iniziale: Temperatura iniziale (°C)
        ph_iniziale: pH iniziale
    """

    conducibilita_base: "Q_" = None
    k_conducibilita: float = 0.5      # μS·L/(cm·mg) - empirico
    k_ossidazione: float = 0.3        # 30% dei detriti si ossida
    k_ph_drift: float = 1e-6          # Deriva pH per impulso
    volume_acqua: "Q_" = None
    temperatura_iniziale: "Q_" = None
    ph_iniziale: float = 7.0

    def __post_init__(self):
        if self.conducibilita_base is None:
            self.conducibilita_base = Q_(5, "uS/cm")  # Acqua deionizzata
        if self.volume_acqua is None:
            self.volume_acqua = Q_(0.5, "L")  # 500 mL tipico
        if self.temperatura_iniziale is None:
            self.temperatura_iniziale = Q_(25, "degC")


class WaterChemistryModel:
    """
    Modello per la chimica dell'acqua in sistemi ESWT.

    L'erosione degli elettrodi rilascia detriti metallici nell'acqua,
    che aumentano la conducibilità e modificano le condizioni di plasma.

    Relazioni chiave:
        σ = σ₀ + k_σ · c_detriti
        ΔpH = k_ph · n_impulsi

    Effetti sulla scarica:
        - Alta conducibilità → breakdown più facile
        - Ma anche → energia dissipata nel volume d'acqua

    Esempio:
        >>> model = WaterChemistryModel()
        >>> state = model.get_initial_state()
        >>> state = model.update(state, massa_erosa=Q_(0.1, "mg"))
    """

    def __init__(self, config: WaterChemistryConfig = None):
        """
        Inizializza il modello.

        Parametri:
            config: Configurazione del modello
        """
        self.config = config or WaterChemistryConfig()

        # Storico per analisi trend
        self._history: List[WaterChemistryState] = []

    def get_initial_state(self) -> WaterChemistryState:
        """
        Crea lo stato iniziale dell'acqua.

        Ritorna:
            WaterChemistryState iniziale
        """
        return WaterChemistryState(
            concentrazione_detriti=Q_(0, "mg/L"),
            concentrazione_ossidi=Q_(0, "mg/L"),
            conducibilita=self.config.conducibilita_base,
            ph=self.config.ph_iniziale,
            temperatura=self.config.temperatura_iniziale,
            volume_totale=self.config.volume_acqua,
            numero_impulsi=0,
        )

    def update(
        self,
        state: WaterChemistryState,
        massa_erosa: "Q_",
        energia_termica: "Q_" = None,
    ) -> WaterChemistryState:
        """
        Aggiorna lo stato chimico dopo un impulso.

        Parametri:
            state: Stato corrente
            massa_erosa: Massa erosa dagli elettrodi (mg)
            energia_termica: Energia dissipata come calore (J)

        Ritorna:
            Nuovo WaterChemistryState
        """
        # Parametri correnti
        volume_L = state.volume_totale.to("L").magnitude
        c_detriti = state.concentrazione_detriti.to("mg/L").magnitude
        c_ossidi = state.concentrazione_ossidi.to("mg/L").magnitude
        sigma = state.conducibilita.to("uS/cm").magnitude
        ph = state.ph
        T = state.temperatura.to("degC").magnitude

        # Massa erosa
        m_erosa = massa_erosa.to("mg").magnitude

        # 1. Aggiorna concentrazione detriti
        delta_c = m_erosa / volume_L
        c_detriti_new = c_detriti + delta_c * (1 - self.config.k_ossidazione)
        c_ossidi_new = c_ossidi + delta_c * self.config.k_ossidazione

        # 2. Aggiorna conducibilità
        #    σ = σ₀ + k_σ · (c_detriti + c_ossidi)
        c_totale = c_detriti_new + c_ossidi_new
        sigma_new = self.config.conducibilita_base.to("uS/cm").magnitude + \
                    self.config.k_conducibilita * c_totale

        # 3. Aggiorna pH (deriva lenta verso acidità per elettrolisi)
        ph_new = ph - self.config.k_ph_drift
        ph_new = np.clip(ph_new, 4.0, 9.0)  # Limiti fisici

        # 4. Aggiorna temperatura (se energia termica fornita)
        if energia_termica is not None:
            # ΔT = E / (m · c_p)
            # c_p acqua ≈ 4186 J/(kg·K)
            E_J = energia_termica.to("J").magnitude
            m_kg = volume_L  # 1 L acqua ≈ 1 kg
            c_p = 4186  # J/(kg·K)
            delta_T = E_J / (m_kg * c_p)
            T_new = T + delta_T
            # Dissipazione verso ambiente
            T_new = T_new * 0.999 + 25 * 0.001  # Lenta dissipazione
        else:
            T_new = T

        # Crea nuovo stato
        new_state = WaterChemistryState(
            concentrazione_detriti=Q_(c_detriti_new, "mg/L"),
            concentrazione_ossidi=Q_(c_ossidi_new, "mg/L"),
            conducibilita=Q_(sigma_new, "uS/cm"),
            ph=float(ph_new),
            temperatura=Q_(T_new, "degC"),
            volume_totale=state.volume_totale,
            numero_impulsi=state.numero_impulsi + 1,
        )

        self._history.append(new_state)

        return new_state

    def calcola_accumulo_detriti(
        self,
        n_impulsi: int,
        massa_erosa_per_impulso: "Q_",
        volume_acqua: "Q_"
    ) -> "Q_":
        """
        Calcola la concentrazione detriti dopo N impulsi.

        Parametri:
            n_impulsi: Numero di impulsi
            massa_erosa_per_impulso: Erosione media per impulso (mg)
            volume_acqua: Volume bagno acqua (L)

        Ritorna:
            Concentrazione detriti (mg/L)
        """
        m_totale = n_impulsi * massa_erosa_per_impulso.to("mg").magnitude
        V = volume_acqua.to("L").magnitude

        return Q_(m_totale / V, "mg/L")

    def conducibilita_da_concentrazione(
        self,
        concentrazione_detriti: "Q_",
        temperatura: "Q_" = None
    ) -> "Q_":
        """
        Calcola la conducibilità dall'concentrazione detriti.

        Include correzione per temperatura:
            σ(T) = σ(25°C) · [1 + α·(T - 25)]
            α ≈ 0.02 /°C per soluzioni acquose

        Parametri:
            concentrazione_detriti: Concentrazione totale detriti (mg/L)
            temperatura: Temperatura (default: 25°C)

        Ritorna:
            Conducibilità (μS/cm)
        """
        c = concentrazione_detriti.to("mg/L").magnitude
        sigma_base = self.config.conducibilita_base.to("uS/cm").magnitude

        sigma_25 = sigma_base + self.config.k_conducibilita * c

        # Correzione temperatura
        if temperatura is not None:
            T = temperatura.to("degC").magnitude
            alpha = 0.02  # Coefficiente temperatura
            sigma = sigma_25 * (1 + alpha * (T - 25))
        else:
            sigma = sigma_25

        return Q_(sigma, "uS/cm")

    def effetto_conducibilita_su_plasma(
        self,
        conducibilita: "Q_"
    ) -> float:
        """
        Calcola l'effetto della conducibilità sull'energia di plasma.

        Alta conducibilità → più facile formare plasma
        Ma anche → parte dell'energia si dissipa nel volume

        Modello semplificato:
            η_plasma = η₀ / (1 + k · σ)

        dove:
            - η₀ = efficienza nominale (acqua pura)
            - k = coefficiente di perdita
            - σ = conducibilità

        Parametri:
            conducibilita: Conducibilità dell'acqua (μS/cm)

        Ritorna:
            Fattore di efficienza (0-1)
        """
        sigma = conducibilita.to("uS/cm").magnitude
        sigma_ref = 5.0  # μS/cm - riferimento acqua pura

        # Coefficiente empirico
        k = 0.001  # 1/μS/cm

        # Efficienza normalizzata
        eta = 1.0 / (1 + k * (sigma - sigma_ref))
        eta = np.clip(eta, 0.5, 1.0)  # Limita degradazione massima

        return float(eta)

    def richiede_cambio_acqua(
        self,
        state: WaterChemistryState,
        soglia_conducibilita: "Q_" = None,
        soglia_detriti: "Q_" = None,
        soglia_ph_min: float = 5.5,
        soglia_ph_max: float = 8.5,
    ) -> Tuple[bool, str]:
        """
        Verifica se è necessario cambiare l'acqua.

        Parametri:
            state: Stato corrente dell'acqua
            soglia_conducibilita: Soglia conducibilità (default: 50 μS/cm)
            soglia_detriti: Soglia detriti (default: 100 mg/L)
            soglia_ph_min: pH minimo accettabile
            soglia_ph_max: pH massimo accettabile

        Ritorna:
            Tuple (necessario_cambio, motivo)
        """
        if soglia_conducibilita is None:
            soglia_conducibilita = Q_(50, "uS/cm")
        if soglia_detriti is None:
            soglia_detriti = Q_(100, "mg/L")

        sigma = state.conducibilita.to("uS/cm").magnitude
        c = state.concentrazione_detriti.to("mg/L").magnitude
        ph = state.ph

        motivi = []

        if sigma > soglia_conducibilita.magnitude:
            motivi.append(f"conducibilità alta ({sigma:.1f} > {soglia_conducibilita.magnitude} μS/cm)")

        if c > soglia_detriti.magnitude:
            motivi.append(f"detriti alti ({c:.1f} > {soglia_detriti.magnitude} mg/L)")

        if ph < soglia_ph_min:
            motivi.append(f"pH basso ({ph:.2f} < {soglia_ph_min})")

        if ph > soglia_ph_max:
            motivi.append(f"pH alto ({ph:.2f} > {soglia_ph_max})")

        if motivi:
            return True, "; ".join(motivi)
        else:
            return False, "Acqua in specifica"

    def simula_sessione(
        self,
        n_impulsi: int,
        massa_erosa_per_impulso: "Q_",
        energia_termica_per_impulso: "Q_" = None,
    ) -> Tuple[WaterChemistryState, List[WaterChemistryState]]:
        """
        Simula l'evoluzione chimica per una sessione completa.

        Parametri:
            n_impulsi: Numero totale di impulsi
            massa_erosa_per_impulso: Erosione media per impulso
            energia_termica_per_impulso: Energia termica per impulso

        Ritorna:
            Tuple (stato_finale, storico_campionato)
        """
        state = self.get_initial_state()
        storico = [state]

        # Campiona ogni 1% degli impulsi
        sample_interval = max(1, n_impulsi // 100)

        for i in range(n_impulsi):
            state = self.update(
                state,
                massa_erosa_per_impulso,
                energia_termica_per_impulso,
            )

            if (i + 1) % sample_interval == 0:
                storico.append(state)

        return state, storico

    def get_statistics(self) -> dict:
        """
        Ritorna statistiche sulla storia chimica.

        Ritorna:
            Dizionario con statistiche aggregate
        """
        if not self._history:
            return {"message": "Nessun dato disponibile"}

        conducibilita = [s.conducibilita.to("uS/cm").magnitude for s in self._history]
        ph_values = [s.ph for s in self._history]
        detriti = [s.concentrazione_detriti.to("mg/L").magnitude for s in self._history]

        return {
            "n_samples": len(self._history),
            "conducibilita": {
                "iniziale_uS_cm": conducibilita[0],
                "finale_uS_cm": conducibilita[-1],
                "max_uS_cm": max(conducibilita),
                "incremento_pct": (conducibilita[-1] / conducibilita[0] - 1) * 100
                    if conducibilita[0] > 0 else 0,
            },
            "ph": {
                "iniziale": ph_values[0],
                "finale": ph_values[-1],
                "min": min(ph_values),
                "max": max(ph_values),
            },
            "detriti": {
                "finale_mg_L": detriti[-1],
                "max_mg_L": max(detriti),
            },
        }

    def reset(self):
        """Resetta lo storico."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"WaterChemistryModel("
            f"σ_base={self.config.conducibilita_base:~.1fP}, "
            f"V={self.config.volume_acqua:~.2fP})"
        )
