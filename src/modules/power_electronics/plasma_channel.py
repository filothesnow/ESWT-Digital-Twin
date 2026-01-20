# Modello del canale di plasma per scariche ESWT
"""
Modello della resistenza dinamica del canale di plasma.

Questo modulo implementa modelli per la resistenza variabile nel tempo
del canale di plasma che si forma durante la scarica elettrica in acqua.

Modelli disponibili:
    1. Rompe-Weizel: R(t) = R_0 * (I_0/I(t))^α
    2. Empirico Chen: Semplificato per simulazioni PAED

Riferimenti:
    - Rompe & Weizel (1944) - Modello resistenza arco
    - Chen et al. (2010) - Approccio semplificato PAED
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import numpy as np

from ...core.units import ureg, Q_


class PlasmaModel(Enum):
    """Tipi di modelli per la resistenza del plasma."""

    ROMPE_WEIZEL = "rompe_weizel"
    COSTANTE = "costante"
    EMPIRICO = "empirico"


@dataclass
class PlasmaState:
    """
    Stato istantaneo del canale di plasma.

    Attributi:
        resistenza: Resistenza istantanea (Ohm)
        potenza: Potenza dissipata (W)
        temperatura: Temperatura stimata (K) - opzionale
    """

    resistenza: "Q_"
    potenza: "Q_"
    temperatura: Optional["Q_"] = None


class PlasmaChannel:
    """
    Modello del canale di plasma per scariche ESWT.

    Il canale di plasma si forma durante la scarica elettrica in acqua
    e presenta una resistenza che varia dinamicamente con la corrente.

    Parametri:
        modello: Tipo di modello per la resistenza (default: ROMPE_WEIZEL)
        resistenza_iniziale: Resistenza iniziale R_0 (Ohm)
        esponente: Esponente α per modello Rompe-Weizel (default: 0.5)
        corrente_riferimento: Corrente di riferimento I_0 (A)
        gap: Distanza inter-elettrodo (mm)

    Esempio:
        >>> plasma = PlasmaChannel(
        ...     resistenza_iniziale=Q_(1, "ohm"),
        ...     gap=Q_(5, "mm")
        ... )
        >>> R = plasma.calcola_resistenza(Q_(10, "kA"))
    """

    def __init__(
        self,
        modello: PlasmaModel = PlasmaModel.ROMPE_WEIZEL,
        resistenza_iniziale: Optional["Q_"] = None,
        esponente: float = 0.5,
        corrente_riferimento: Optional["Q_"] = None,
        gap: Optional["Q_"] = None,
    ):
        """
        Inizializza il modello del canale di plasma.

        Parametri:
            modello: Tipo di modello per la resistenza
            resistenza_iniziale: R_0 in Ohm (default: calcolata dal gap)
            esponente: α per Rompe-Weizel (tipico 0.5-0.7)
            corrente_riferimento: I_0 in Ampere (default: 1 kA)
            gap: Distanza inter-elettrodo in mm (default: 5 mm)
        """
        self.modello = modello
        self.esponente = esponente

        # Distanza inter-elettrodo
        self._gap = gap if gap else Q_(5, "mm")

        # Resistenza iniziale: se non specificata, stima dal gap
        if resistenza_iniziale:
            self._r0 = resistenza_iniziale.to("ohm")
        else:
            # Stima empirica: ~0.1-1 Ohm per gap 5-10mm
            gap_mm = self._gap.to("mm").magnitude
            self._r0 = Q_(0.2 * gap_mm / 5, "ohm")

        # Corrente di riferimento
        self._i0 = corrente_riferimento.to("A") if corrente_riferimento else Q_(1000, "A")

        # Stato corrente
        self._resistenza_corrente = self._r0
        self._corrente_corrente = Q_(0, "A")

    @property
    def resistenza_iniziale(self) -> "Q_":
        """Ritorna la resistenza iniziale R_0."""
        return self._r0

    @property
    def gap(self) -> "Q_":
        """Ritorna la distanza inter-elettrodo."""
        return self._gap

    @property
    def resistenza(self) -> "Q_":
        """Ritorna la resistenza corrente del canale."""
        return self._resistenza_corrente

    def calcola_resistenza(self, corrente: "Q_") -> "Q_":
        """
        Calcola la resistenza del plasma in funzione della corrente.

        Modello Rompe-Weizel:
            R(I) = R_0 * (I_0 / I)^α

        Per I → 0, la resistenza tende a infinito (breakdown non avvenuto).
        Per I grande, la resistenza diminuisce (plasma conduttivo).

        Parametri:
            corrente: Corrente attraverso il canale (A)

        Ritorna:
            Resistenza in Ohm
        """
        i = abs(corrente.to("A").magnitude)
        i0 = self._i0.magnitude
        r0 = self._r0.magnitude

        if self.modello == PlasmaModel.COSTANTE:
            resistenza = r0

        elif self.modello == PlasmaModel.ROMPE_WEIZEL:
            # Evita divisione per zero
            if i < 1e-6:  # Corrente trascurabile
                # Prima del breakdown, resistenza molto alta
                resistenza = r0 * 1000
            else:
                resistenza = r0 * (i0 / i) ** self.esponente

        elif self.modello == PlasmaModel.EMPIRICO:
            # Modello semplificato: decadimento esponenziale
            if i < 1e-6:
                resistenza = r0 * 1000
            else:
                tau = i0 / 10  # Costante di scala
                resistenza = r0 * np.exp(-i / tau) + r0 * 0.01

        else:
            resistenza = r0

        self._resistenza_corrente = Q_(resistenza, "ohm")
        self._corrente_corrente = corrente
        return self._resistenza_corrente

    def calcola_potenza_dissipata(self, corrente: "Q_") -> "Q_":
        """
        Calcola la potenza dissipata nel canale di plasma.

        P = I² * R

        Parametri:
            corrente: Corrente attraverso il canale (A)

        Ritorna:
            Potenza dissipata in Watt
        """
        i = corrente.to("A").magnitude
        r = self.calcola_resistenza(corrente).magnitude
        return Q_(i**2 * r, "W")

    def calcola_tensione_arco(self, corrente: "Q_") -> "Q_":
        """
        Calcola la caduta di tensione sul canale di plasma.

        V_arc = I * R

        Parametri:
            corrente: Corrente attraverso il canale (A)

        Ritorna:
            Tensione in Volt
        """
        i = corrente.to("A").magnitude
        r = self.calcola_resistenza(corrente).magnitude
        return Q_(i * r, "V")

    def stima_temperatura_plasma(self, potenza: "Q_", volume_plasma: "Q_" = None) -> "Q_":
        """
        Stima grossolana della temperatura del plasma.

        Questa è una stima molto approssimativa basata sulla potenza
        dissipata. Per simulazioni accurate, usare modelli termodinamici.

        Parametri:
            potenza: Potenza dissipata (W)
            volume_plasma: Volume del canale di plasma (opzionale)

        Ritorna:
            Temperatura stimata in Kelvin
        """
        # Stima molto semplificata
        # Per ESWT, le temperature del plasma sono tipicamente 10000-50000 K
        p = potenza.to("W").magnitude

        # Relazione empirica approssimativa
        # T ~ T_ambiente + k * P^0.25
        # dove k è un fattore di scala empirico
        t_amb = 300  # K
        k = 100  # Fattore empirico

        if p < 1:
            temperatura = t_amb
        else:
            temperatura = t_amb + k * (p**0.25)

        # Limita a valori realistici per plasma ESWT
        temperatura = min(temperatura, 50000)  # Max 50000 K

        return Q_(temperatura, "K")

    def get_state(self, corrente: "Q_") -> PlasmaState:
        """
        Ritorna lo stato completo del canale di plasma.

        Parametri:
            corrente: Corrente attraverso il canale

        Ritorna:
            PlasmaState con resistenza, potenza e temperatura
        """
        r = self.calcola_resistenza(corrente)
        p = self.calcola_potenza_dissipata(corrente)
        t = self.stima_temperatura_plasma(p)

        return PlasmaState(resistenza=r, potenza=p, temperatura=t)

    def reset(self):
        """Resetta il canale di plasma allo stato iniziale."""
        self._resistenza_corrente = self._r0
        self._corrente_corrente = Q_(0, "A")

    def __repr__(self) -> str:
        return (
            f"PlasmaChannel(modello={self.modello.value}, "
            f"R_0={self._r0:~.3fP}, α={self.esponente}, "
            f"gap={self._gap:~.1fP})"
        )


def crea_modello_rompe_weizel(
    gap: "Q_",
    esponente: float = 0.5,
    corrente_riferimento: "Q_" = None,
) -> PlasmaChannel:
    """
    Factory function per creare un modello Rompe-Weizel.

    Parametri:
        gap: Distanza inter-elettrodo
        esponente: Esponente α (default 0.5)
        corrente_riferimento: I_0 (default 1 kA)

    Ritorna:
        PlasmaChannel configurato con modello Rompe-Weizel
    """
    return PlasmaChannel(
        modello=PlasmaModel.ROMPE_WEIZEL,
        gap=gap,
        esponente=esponente,
        corrente_riferimento=corrente_riferimento or Q_(1, "kA"),
    )


def crea_modello_costante(resistenza: "Q_") -> PlasmaChannel:
    """
    Factory function per creare un modello a resistenza costante.

    Utile per test e confronti con soluzioni analitiche RLC.

    Parametri:
        resistenza: Valore costante di resistenza

    Ritorna:
        PlasmaChannel con resistenza costante
    """
    return PlasmaChannel(
        modello=PlasmaModel.COSTANTE,
        resistenza_iniziale=resistenza,
    )
