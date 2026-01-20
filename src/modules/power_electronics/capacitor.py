# Modello del condensatore per elettronica di potenza ESWT
"""
Modello del condensatore per dispositivi ESWT.

Questo modulo implementa il modello di un condensatore ad alta tensione
usato per accumulare energia prima della scarica nel canale di plasma.

Equazioni principali (da Chen et al. 2010):
    E(t) = 0.5 * C * (U_M² - U(t)²)  # Energia rilasciata al tempo t
    E_max = 0.5 * C * U_M²            # Energia massima immagazzinata

Parametri tipici:
    - Capacità: 0.1-10 μF
    - Tensione: 10-30 kV
    - Energia: 20-3300 J
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ...core.units import ureg, Q_
from ...core.constants import ESWTParameters


@dataclass
class CapacitorState:
    """
    Stato istantaneo del condensatore.

    Attributi:
        tensione: Tensione ai capi (V)
        energia: Energia immagazzinata (J)
        carica: Carica accumulata (C)
    """

    tensione: "Q_"
    energia: "Q_"
    carica: "Q_"


class Capacitor:
    """
    Modello di condensatore ad alta tensione per ESWT.

    Parametri:
        capacita: Valore di capacità (F o μF)
        tensione_max: Tensione massima di carica (V o kV)
        tensione_iniziale: Tensione iniziale (opzionale, default=tensione_max)

    Esempio:
        >>> cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        >>> print(cap.energia_immagazzinata)
        200.0 J
    """

    def __init__(
        self,
        capacita: "Q_",
        tensione_max: "Q_",
        tensione_iniziale: Optional["Q_"] = None,
    ):
        """
        Inizializza il condensatore.

        Parametri:
            capacita: Capacità in Farad (accetta anche μF, nF)
            tensione_max: Tensione massima di carica in Volt (accetta anche kV)
            tensione_iniziale: Tensione iniziale (default = tensione_max)
        """
        # Converti a unità SI
        self._capacita = capacita.to("F")
        self._tensione_max = tensione_max.to("V")
        self._tensione = (
            tensione_iniziale.to("V") if tensione_iniziale else self._tensione_max
        )

        # Validazione parametri
        self._valida_parametri()

    def _valida_parametri(self):
        """Verifica che i parametri siano nei range tipici ESWT."""
        c_min = ESWTParameters.CAPACITA_MIN.to("F").magnitude
        c_max = ESWTParameters.CAPACITA_MAX.to("F").magnitude
        v_min = ESWTParameters.TENSIONE_MIN.to("V").magnitude
        v_max = ESWTParameters.TENSIONE_MAX.to("V").magnitude

        c_val = self._capacita.magnitude
        v_val = self._tensione_max.magnitude

        if not (c_min <= c_val <= c_max * 10):  # Margine 10x
            print(
                f"Attenzione: capacità {self._capacita:~.2fP} fuori range tipico "
                f"[{ESWTParameters.CAPACITA_MIN:~P} - {ESWTParameters.CAPACITA_MAX:~P}]"
            )

        if not (v_min <= v_val <= v_max * 1.5):  # Margine 1.5x
            print(
                f"Attenzione: tensione {self._tensione_max:~.2fP} fuori range tipico "
                f"[{ESWTParameters.TENSIONE_MIN:~P} - {ESWTParameters.TENSIONE_MAX:~P}]"
            )

    @property
    def capacita(self) -> "Q_":
        """Ritorna la capacità del condensatore."""
        return self._capacita

    @property
    def tensione(self) -> "Q_":
        """Ritorna la tensione attuale ai capi del condensatore."""
        return self._tensione

    @tensione.setter
    def tensione(self, valore: "Q_"):
        """Imposta la tensione ai capi del condensatore."""
        self._tensione = valore.to("V")

    @property
    def tensione_max(self) -> "Q_":
        """Ritorna la tensione massima di carica."""
        return self._tensione_max

    @property
    def energia_immagazzinata(self) -> "Q_":
        """
        Calcola l'energia attualmente immagazzinata nel condensatore.

        E = 0.5 * C * V²

        Ritorna:
            Energia in Joule
        """
        c = self._capacita.magnitude
        v = self._tensione.magnitude
        return Q_(0.5 * c * v**2, "J")

    @property
    def energia_max(self) -> "Q_":
        """
        Calcola l'energia massima immagazzinabile.

        E_max = 0.5 * C * V_max²

        Ritorna:
            Energia massima in Joule
        """
        c = self._capacita.magnitude
        v = self._tensione_max.magnitude
        return Q_(0.5 * c * v**2, "J")

    @property
    def carica(self) -> "Q_":
        """
        Calcola la carica attuale sul condensatore.

        Q = C * V

        Ritorna:
            Carica in Coulomb
        """
        c = self._capacita.magnitude
        v = self._tensione.magnitude
        return Q_(c * v, "C")

    def energia_rilasciata(self, tensione_finale: "Q_") -> "Q_":
        """
        Calcola l'energia rilasciata durante la scarica fino a una tensione finale.

        ΔE = 0.5 * C * (V_iniziale² - V_finale²)

        Da Chen et al. (2010), Eq. 1.

        Parametri:
            tensione_finale: Tensione finale dopo la scarica

        Ritorna:
            Energia rilasciata in Joule
        """
        c = self._capacita.magnitude
        v_i = self._tensione.magnitude
        v_f = tensione_finale.to("V").magnitude
        return Q_(0.5 * c * (v_i**2 - v_f**2), "J")

    def scarica(self, corrente: "Q_", dt: "Q_") -> "Q_":
        """
        Aggiorna lo stato del condensatore dopo un intervallo di tempo.

        dV/dt = -I/C

        Parametri:
            corrente: Corrente di scarica (A)
            dt: Intervallo di tempo (s)

        Ritorna:
            Nuova tensione del condensatore
        """
        c = self._capacita.magnitude
        i = corrente.to("A").magnitude
        dt_s = dt.to("s").magnitude

        v_new = self._tensione.magnitude - (i / c) * dt_s

        # La tensione non può diventare negativa (in valore assoluto)
        # In un circuito RLC può oscillare, quindi permettiamo valori negativi
        self._tensione = Q_(v_new, "V")
        return self._tensione

    def reset(self):
        """Ricarica il condensatore alla tensione massima."""
        self._tensione = self._tensione_max

    def get_state(self) -> CapacitorState:
        """Ritorna lo stato completo del condensatore."""
        return CapacitorState(
            tensione=self._tensione,
            energia=self.energia_immagazzinata,
            carica=self.carica,
        )

    def __repr__(self) -> str:
        return (
            f"Capacitor(C={self._capacita.to('uF'):~.2fP}, "
            f"V={self._tensione.to('kV'):~.2fP}, "
            f"E={self.energia_immagazzinata:~.1fP})"
        )


def calcola_capacita_da_energia(energia: "Q_", tensione: "Q_") -> "Q_":
    """
    Calcola la capacità necessaria per immagazzinare una certa energia.

    C = 2*E / V²

    Parametri:
        energia: Energia desiderata (J)
        tensione: Tensione di carica (V o kV)

    Ritorna:
        Capacità necessaria (F)

    Esempio:
        >>> C = calcola_capacita_da_energia(Q_(200, "J"), Q_(20, "kV"))
        >>> print(C.to("uF"))
        1.0 μF
    """
    e = energia.to("J").magnitude
    v = tensione.to("V").magnitude
    return Q_(2 * e / v**2, "F")
