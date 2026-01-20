# Modello efficienza vs impulsi
"""
Modello per il calcolo dell'efficienza del dispositivo ESWT
in funzione del numero di impulsi erogati.

L'efficienza diminuisce nel tempo a causa di:
- Erosione degli elettrodi
- Aumento del gap inter-elettrodo
- Accumulo di detriti nell'acqua
- Degradazione del dielettrico

Modelli implementati:
    - Esponenziale: η(N) = η_0 * exp(-λ*N)
    - Lineare con saturazione: η(N) = η_0 * (1 - k*N) per N < N_max
    - Weibull: η(N) = η_0 * exp(-(N/N_c)^β)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import numpy as np

from ...core.units import ureg, Q_


class EfficiencyModelType(Enum):
    """Tipo di modello di efficienza."""

    EXPONENTIAL = "esponenziale"
    LINEAR = "lineare"
    WEIBULL = "weibull"


@dataclass
class EfficiencyModel:
    """
    Modello di efficienza vs numero impulsi.

    Attributi:
        tipo: Tipo di modello
        efficienza_iniziale: Efficienza a N=0 (0-1)
        parametro_decadimento: λ per esponenziale, k per lineare, N_c per Weibull
        parametro_forma: β per Weibull (ignorato per altri modelli)
        efficienza_minima: Efficienza minima accettabile (0-1)
    """

    tipo: EfficiencyModelType = EfficiencyModelType.EXPONENTIAL
    efficienza_iniziale: float = 1.0
    parametro_decadimento: float = 1e-5  # λ o k o N_c
    parametro_forma: float = 2.0  # β per Weibull
    efficienza_minima: float = 0.5

    def calcola(self, numero_impulsi: int) -> float:
        """
        Calcola l'efficienza per un dato numero di impulsi.

        Parametri:
            numero_impulsi: Numero totale di impulsi erogati

        Ritorna:
            Efficienza (0-1)
        """
        N = numero_impulsi
        eta_0 = self.efficienza_iniziale

        if self.tipo == EfficiencyModelType.EXPONENTIAL:
            # η(N) = η_0 * exp(-λ*N)
            lam = self.parametro_decadimento
            eta = eta_0 * np.exp(-lam * N)

        elif self.tipo == EfficiencyModelType.LINEAR:
            # η(N) = η_0 * (1 - k*N), saturato a η_min
            k = self.parametro_decadimento
            eta = eta_0 * (1 - k * N)

        elif self.tipo == EfficiencyModelType.WEIBULL:
            # η(N) = η_0 * exp(-(N/N_c)^β)
            N_c = self.parametro_decadimento
            beta = self.parametro_forma
            eta = eta_0 * np.exp(-((N / N_c) ** beta)) if N_c > 0 else eta_0

        else:
            eta = eta_0

        return max(eta, self.efficienza_minima)

    def impulsi_a_efficienza(self, efficienza_target: float) -> int:
        """
        Calcola il numero di impulsi per raggiungere un'efficienza target.

        Parametri:
            efficienza_target: Efficienza desiderata (0-1)

        Ritorna:
            Numero di impulsi
        """
        if efficienza_target >= self.efficienza_iniziale:
            return 0

        if efficienza_target <= self.efficienza_minima:
            efficienza_target = self.efficienza_minima * 1.01

        eta_0 = self.efficienza_iniziale
        eta = efficienza_target

        if self.tipo == EfficiencyModelType.EXPONENTIAL:
            # N = -ln(η/η_0) / λ
            lam = self.parametro_decadimento
            if lam > 0:
                N = -np.log(eta / eta_0) / lam
            else:
                N = float("inf")

        elif self.tipo == EfficiencyModelType.LINEAR:
            # N = (1 - η/η_0) / k
            k = self.parametro_decadimento
            if k > 0:
                N = (1 - eta / eta_0) / k
            else:
                N = float("inf")

        elif self.tipo == EfficiencyModelType.WEIBULL:
            # N = N_c * (-ln(η/η_0))^(1/β)
            N_c = self.parametro_decadimento
            beta = self.parametro_forma
            if N_c > 0 and beta > 0:
                log_ratio = -np.log(eta / eta_0)
                if log_ratio > 0:
                    N = N_c * (log_ratio ** (1 / beta))
                else:
                    N = 0
            else:
                N = float("inf")

        else:
            N = float("inf")

        return int(N)


def calcola_efficienza(
    numero_impulsi: int,
    modello: EfficiencyModel = None,
    gap_attuale: "Q_" = None,
    gap_nominale: "Q_" = None,
) -> float:
    """
    Calcola l'efficienza considerando sia il numero di impulsi che il gap.

    L'efficienza totale è il prodotto di:
    - Efficienza temporale (degradazione con N impulsi)
    - Efficienza geometrica (effetto del gap)

    Parametri:
        numero_impulsi: Numero di impulsi erogati
        modello: Modello di efficienza (default: esponenziale)
        gap_attuale: Gap inter-elettrodo attuale
        gap_nominale: Gap nominale di progetto

    Ritorna:
        Efficienza totale (0-1)
    """
    if modello is None:
        modello = EfficiencyModel()

    # Efficienza da modello temporale
    eta_tempo = modello.calcola(numero_impulsi)

    # Efficienza da gap (se fornito)
    if gap_attuale is not None and gap_nominale is not None:
        g_att = gap_attuale.to("mm").magnitude
        g_nom = gap_nominale.to("mm").magnitude

        # L'efficienza diminuisce se il gap devia dal nominale
        # Modello parabolico centrato sul gap nominale
        if g_nom > 0:
            deviazione_relativa = abs(g_att - g_nom) / g_nom
            eta_gap = max(1 - 0.5 * deviazione_relativa ** 2, 0.5)
        else:
            eta_gap = 1.0
    else:
        eta_gap = 1.0

    return eta_tempo * eta_gap


def stima_impulsi_rimanenti(
    efficienza_attuale: float,
    efficienza_limite: float = 0.7,
    modello: EfficiencyModel = None,
) -> int:
    """
    Stima il numero di impulsi rimanenti prima di raggiungere l'efficienza limite.

    Parametri:
        efficienza_attuale: Efficienza corrente
        efficienza_limite: Efficienza minima accettabile
        modello: Modello di efficienza

    Ritorna:
        Numero stimato di impulsi rimanenti
    """
    if modello is None:
        modello = EfficiencyModel()

    if efficienza_attuale <= efficienza_limite:
        return 0

    # Trova N attuale
    N_attuale = modello.impulsi_a_efficienza(efficienza_attuale)

    # Trova N al limite
    N_limite = modello.impulsi_a_efficienza(efficienza_limite)

    return max(int(N_limite - N_attuale), 0)


def genera_curva_efficienza(
    modello: EfficiencyModel,
    n_max: int = 100000,
    n_punti: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera la curva di efficienza vs numero impulsi.

    Parametri:
        modello: Modello di efficienza
        n_max: Numero massimo di impulsi
        n_punti: Numero di punti nella curva

    Ritorna:
        Tuple (array_impulsi, array_efficienza)
    """
    impulsi = np.linspace(0, n_max, n_punti, dtype=int)
    efficienza = np.array([modello.calcola(n) for n in impulsi])

    return impulsi, efficienza


def crea_modello_da_dati(
    impulsi: List[int],
    efficienze: List[float],
    tipo: EfficiencyModelType = EfficiencyModelType.EXPONENTIAL,
) -> EfficiencyModel:
    """
    Crea un modello di efficienza fittando dati sperimentali.

    Parametri:
        impulsi: Lista di numeri di impulsi
        efficienze: Lista di efficienze misurate
        tipo: Tipo di modello da fittare

    Ritorna:
        EfficiencyModel con parametri ottimizzati
    """
    from scipy.optimize import curve_fit

    N = np.array(impulsi)
    eta = np.array(efficienze)

    if tipo == EfficiencyModelType.EXPONENTIAL:
        def func(n, eta_0, lam):
            return eta_0 * np.exp(-lam * n)

        try:
            popt, _ = curve_fit(func, N, eta, p0=[1.0, 1e-5], bounds=([0, 0], [1.5, 1]))
            return EfficiencyModel(
                tipo=tipo,
                efficienza_iniziale=popt[0],
                parametro_decadimento=popt[1],
            )
        except:
            pass

    elif tipo == EfficiencyModelType.LINEAR:
        def func(n, eta_0, k):
            return eta_0 * (1 - k * n)

        try:
            popt, _ = curve_fit(func, N, eta, p0=[1.0, 1e-6], bounds=([0, 0], [1.5, 1e-3]))
            return EfficiencyModel(
                tipo=tipo,
                efficienza_iniziale=popt[0],
                parametro_decadimento=popt[1],
            )
        except:
            pass

    # Fallback: modello default
    return EfficiencyModel(tipo=tipo)


def calcola_energia_efficace(
    energia_nominale: "Q_",
    efficienza: float,
) -> "Q_":
    """
    Calcola l'energia efficace considerando l'efficienza.

    Parametri:
        energia_nominale: Energia nominale della scarica
        efficienza: Efficienza corrente (0-1)

    Ritorna:
        Energia efficace (= energia_nominale * efficienza)
    """
    e_nom = energia_nominale.to("J").magnitude
    return Q_(e_nom * efficienza, "J")


def report_stato_sistema(
    numero_impulsi: int,
    modello: EfficiencyModel,
    gap_attuale: "Q_" = None,
    gap_nominale: "Q_" = None,
) -> dict:
    """
    Genera un report sullo stato del sistema.

    Parametri:
        numero_impulsi: Impulsi erogati
        modello: Modello di efficienza
        gap_attuale: Gap attuale (opzionale)
        gap_nominale: Gap nominale (opzionale)

    Ritorna:
        Dizionario con il report
    """
    efficienza = calcola_efficienza(
        numero_impulsi, modello, gap_attuale, gap_nominale
    )

    impulsi_rimanenti_70 = stima_impulsi_rimanenti(efficienza, 0.7, modello)
    impulsi_rimanenti_50 = stima_impulsi_rimanenti(efficienza, 0.5, modello)

    report = {
        "numero_impulsi": numero_impulsi,
        "efficienza_attuale": efficienza,
        "efficienza_percentuale": efficienza * 100,
        "impulsi_rimanenti_a_70_percento": impulsi_rimanenti_70,
        "impulsi_rimanenti_a_50_percento": impulsi_rimanenti_50,
        "stato": "OK" if efficienza > 0.7 else "ATTENZIONE" if efficienza > 0.5 else "CRITICO",
    }

    if gap_attuale is not None:
        report["gap_attuale_mm"] = gap_attuale.to("mm").magnitude

    if gap_nominale is not None:
        report["gap_nominale_mm"] = gap_nominale.to("mm").magnitude

    return report
