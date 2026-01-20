# Simulatore di scarica capacitiva per ESWT
"""
Simulatore della scarica del condensatore attraverso il canale di plasma.

Questo modulo implementa il solver ODE per il circuito RLC con
resistenza di plasma variabile nel tempo.

Equazioni del circuito:
    dI/dt = (V_c - I*R_plasma(I) - V_arc) / L
    dV_c/dt = -I / C

Per il caso senza plasma (R costante), il sistema ammette soluzione analitica:
    - Sovrasmorzato: ζ > 1
    - Criticamente smorzato: ζ = 1
    - Sottosmorzato: ζ < 1 (oscillante)

dove ζ = R/(2*sqrt(L/C)) è il fattore di smorzamento.

Riferimenti:
    - Chen et al. (2010) - Eq. 1-3
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp

from ...core.units import ureg, Q_
from .capacitor import Capacitor
from .plasma_channel import PlasmaChannel, crea_modello_costante


@dataclass
class DischargeResult:
    """
    Risultato della simulazione di scarica.

    Attributi:
        tempo: Array dei tempi (s)
        corrente: Array delle correnti (A)
        tensione_condensatore: Array delle tensioni (V)
        energia_plasma: Energia totale dissipata nel plasma (J)
        corrente_picco: Corrente massima raggiunta (A)
        tempo_picco: Tempo al quale si raggiunge la corrente picco (s)
        energia_rilasciata: Energia rilasciata dal condensatore (J)
    """

    tempo: np.ndarray
    corrente: np.ndarray
    tensione_condensatore: np.ndarray
    resistenza_plasma: np.ndarray
    energia_plasma: "Q_"
    corrente_picco: "Q_"
    tempo_picco: "Q_"
    energia_rilasciata: "Q_"


class DischargeSimulator:
    """
    Simulatore di scarica RLC con canale di plasma.

    Parametri:
        condensatore: Oggetto Capacitor
        plasma: Oggetto PlasmaChannel
        induttanza: Induttanza del circuito (H o μH)
        tensione_arco: Caduta di tensione costante dell'arco (opzionale)

    Esempio:
        >>> cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        >>> plasma = PlasmaChannel(gap=Q_(5, "mm"))
        >>> sim = DischargeSimulator(cap, plasma, Q_(5, "uH"))
        >>> result = sim.simula(Q_(10, "us"))
    """

    def __init__(
        self,
        condensatore: Capacitor,
        plasma: PlasmaChannel,
        induttanza: "Q_",
        tensione_arco: Optional["Q_"] = None,
    ):
        """
        Inizializza il simulatore di scarica.

        Parametri:
            condensatore: Modello del condensatore
            plasma: Modello del canale di plasma
            induttanza: Induttanza del circuito
            tensione_arco: Caduta di tensione costante (default: 0)
        """
        self.condensatore = condensatore
        self.plasma = plasma
        self._induttanza = induttanza.to("H")
        self._v_arc = tensione_arco.to("V") if tensione_arco else Q_(0, "V")

    @property
    def induttanza(self) -> "Q_":
        """Ritorna l'induttanza del circuito."""
        return self._induttanza

    @property
    def frequenza_naturale(self) -> "Q_":
        """
        Calcola la frequenza naturale del circuito LC.

        ω_0 = 1 / sqrt(L*C)
        f_0 = ω_0 / (2π)

        Ritorna:
            Frequenza naturale in Hz
        """
        l = self._induttanza.magnitude
        c = self.condensatore.capacita.magnitude
        omega_0 = 1 / np.sqrt(l * c)
        return Q_(omega_0 / (2 * np.pi), "Hz")

    @property
    def periodo_naturale(self) -> "Q_":
        """
        Calcola il periodo naturale del circuito LC.

        T = 2π * sqrt(L*C)

        Ritorna:
            Periodo naturale in secondi
        """
        l = self._induttanza.magnitude
        c = self.condensatore.capacita.magnitude
        t = 2 * np.pi * np.sqrt(l * c)
        return Q_(t, "s")

    def fattore_smorzamento(self, resistenza: "Q_" = None) -> float:
        """
        Calcola il fattore di smorzamento del circuito.

        ζ = R / (2 * sqrt(L/C))

        Parametri:
            resistenza: Resistenza da usare (default: R_0 del plasma)

        Ritorna:
            Fattore di smorzamento (adimensionale)
        """
        r = (resistenza or self.plasma.resistenza_iniziale).to("ohm").magnitude
        l = self._induttanza.magnitude
        c = self.condensatore.capacita.magnitude
        return r / (2 * np.sqrt(l / c))

    def _sistema_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Sistema di ODE per la scarica RLC con plasma.

        y[0] = I (corrente)
        y[1] = V_c (tensione condensatore)

        dI/dt = (V_c - I*R_plasma(I) - V_arc) / L
        dV_c/dt = -I / C
        """
        i, v_c = y

        # Calcola resistenza plasma (dipende dalla corrente)
        r_plasma = self.plasma.calcola_resistenza(Q_(i, "A")).magnitude

        # Parametri circuito
        l = self._induttanza.magnitude
        c = self.condensatore.capacita.magnitude
        v_arc = self._v_arc.magnitude

        # Equazioni differenziali
        di_dt = (v_c - i * r_plasma - v_arc) / l
        dv_c_dt = -i / c

        return np.array([di_dt, dv_c_dt])

    def simula(
        self,
        durata: "Q_",
        dt: Optional["Q_"] = None,
        metodo: str = "RK45",
    ) -> DischargeResult:
        """
        Esegue la simulazione della scarica.

        Parametri:
            durata: Durata della simulazione (s o μs)
            dt: Passo temporale per output (opzionale)
            metodo: Metodo di integrazione scipy (default: RK45)

        Ritorna:
            DischargeResult con tutti i dati della simulazione
        """
        # Converti a secondi
        t_end = durata.to("s").magnitude

        # Determina passo temporale di output
        if dt:
            n_punti = int(t_end / dt.to("s").magnitude) + 1
        else:
            # Default: 1000 punti o 1 ns, quello che dà più punti
            n_punti = max(1000, int(t_end / 1e-9))

        t_eval = np.linspace(0, t_end, n_punti)

        # Condizioni iniziali: I=0, V_c=V_max
        y0 = np.array([0, self.condensatore.tensione.magnitude])

        # Risolvi sistema ODE
        sol = solve_ivp(
            self._sistema_ode,
            t_span=(0, t_end),
            y0=y0,
            method=metodo,
            t_eval=t_eval,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Simulazione fallita: {sol.message}")

        # Estrai risultati
        tempo = sol.t
        corrente = sol.y[0]
        tensione = sol.y[1]

        # Calcola resistenza plasma per ogni punto
        resistenza = np.array(
            [self.plasma.calcola_resistenza(Q_(i, "A")).magnitude for i in corrente]
        )

        # Calcola energia dissipata nel plasma
        # E = ∫ I² * R dt (integrazione numerica)
        potenza = corrente**2 * resistenza
        energia_plasma = np.trapezoid(potenza, tempo)

        # Trova corrente di picco
        idx_picco = np.argmax(np.abs(corrente))
        corrente_picco = corrente[idx_picco]
        tempo_picco = tempo[idx_picco]

        # Energia rilasciata dal condensatore
        v_iniziale = self.condensatore.tensione_max.magnitude
        v_finale = tensione[-1]
        c = self.condensatore.capacita.magnitude
        energia_rilasciata = 0.5 * c * (v_iniziale**2 - v_finale**2)

        return DischargeResult(
            tempo=tempo,
            corrente=corrente,
            tensione_condensatore=tensione,
            resistenza_plasma=resistenza,
            energia_plasma=Q_(energia_plasma, "J"),
            corrente_picco=Q_(corrente_picco, "A"),
            tempo_picco=Q_(tempo_picco, "s"),
            energia_rilasciata=Q_(energia_rilasciata, "J"),
        )

    def __repr__(self) -> str:
        return (
            f"DischargeSimulator("
            f"C={self.condensatore.capacita.to('uF'):~.2fP}, "
            f"L={self._induttanza.to('uH'):~.2fP}, "
            f"V={self.condensatore.tensione_max.to('kV'):~.1fP})"
        )


def simula_scarica_rlc_analitica(
    capacita: "Q_",
    induttanza: "Q_",
    resistenza: "Q_",
    tensione_iniziale: "Q_",
    durata: "Q_",
    n_punti: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soluzione analitica del circuito RLC (resistenza costante).

    Utile per validazione del solver numerico.

    Parametri:
        capacita: Capacità C (F)
        induttanza: Induttanza L (H)
        resistenza: Resistenza R (Ohm)
        tensione_iniziale: Tensione iniziale V_0 (V)
        durata: Durata simulazione (s)
        n_punti: Numero di punti temporali

    Ritorna:
        Tuple (tempo, corrente, tensione)
    """
    c = capacita.to("F").magnitude
    l = induttanza.to("H").magnitude
    r = resistenza.to("ohm").magnitude
    v0 = tensione_iniziale.to("V").magnitude
    t_end = durata.to("s").magnitude

    tempo = np.linspace(0, t_end, n_punti)

    # Parametri caratteristici
    omega_0 = 1 / np.sqrt(l * c)  # Frequenza naturale
    alpha = r / (2 * l)  # Coefficiente di smorzamento
    zeta = alpha / omega_0  # Fattore di smorzamento

    if zeta < 1:
        # Sottosmorzato (oscillante)
        omega_d = omega_0 * np.sqrt(1 - zeta**2)
        corrente = (v0 / (omega_d * l)) * np.exp(-alpha * tempo) * np.sin(omega_d * tempo)
        tensione = (
            v0 * np.exp(-alpha * tempo) * (np.cos(omega_d * tempo) + (alpha / omega_d) * np.sin(omega_d * tempo))
        )

    elif zeta == 1:
        # Criticamente smorzato
        corrente = (v0 / l) * tempo * np.exp(-alpha * tempo)
        tensione = v0 * (1 + alpha * tempo) * np.exp(-alpha * tempo)

    else:
        # Sovrasmorzato
        s1 = -alpha + np.sqrt(alpha**2 - omega_0**2)
        s2 = -alpha - np.sqrt(alpha**2 - omega_0**2)
        a1 = v0 * s2 / (l * (s2 - s1))
        a2 = -v0 * s1 / (l * (s2 - s1))
        corrente = a1 * np.exp(s1 * tempo) + a2 * np.exp(s2 * tempo)
        b1 = v0 * s2 / (s2 - s1)
        b2 = -v0 * s1 / (s2 - s1)
        tensione = b1 * np.exp(s1 * tempo) + b2 * np.exp(s2 * tempo)

    return tempo, corrente, tensione


def main():
    """
    Funzione principale per test e visualizzazione.

    Eseguire con: python -m src.modules.power_electronics.discharge
    """
    import argparse

    parser = argparse.ArgumentParser(description="Simulatore scarica ESWT")
    parser.add_argument("--voltage", type=float, default=20000, help="Tensione (V)")
    parser.add_argument("--capacitance", type=float, default=1e-6, help="Capacità (F)")
    parser.add_argument("--inductance", type=float, default=5e-6, help="Induttanza (H)")
    parser.add_argument("--duration", type=float, default=10e-6, help="Durata (s)")
    parser.add_argument("--plot", action="store_true", help="Mostra grafico")
    args = parser.parse_args()

    # Crea componenti
    cap = Capacitor(Q_(args.capacitance, "F"), Q_(args.voltage, "V"))
    plasma = PlasmaChannel(gap=Q_(5, "mm"))
    sim = DischargeSimulator(cap, plasma, Q_(args.inductance, "H"))

    print(f"Simulatore: {sim}")
    print(f"Frequenza naturale: {sim.frequenza_naturale.to('MHz'):~.3fP}")
    print(f"Periodo naturale: {sim.periodo_naturale.to('us'):~.3fP}")
    print(f"Energia iniziale: {cap.energia_max:~.2fP}")

    # Esegui simulazione
    result = sim.simula(Q_(args.duration, "s"))

    print(f"\nRisultati:")
    print(f"  Corrente picco: {result.corrente_picco.to('kA'):~.2fP}")
    print(f"  Tempo picco: {result.tempo_picco.to('us'):~.3fP}")
    print(f"  Energia plasma: {result.energia_plasma:~.2fP}")
    print(f"  Energia rilasciata: {result.energia_rilasciata:~.2fP}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Corrente
            axes[0].plot(result.tempo * 1e6, result.corrente / 1000, "b-")
            axes[0].set_ylabel("Corrente (kA)")
            axes[0].grid(True)
            axes[0].set_title("Simulazione Scarica ESWT")

            # Tensione
            axes[1].plot(result.tempo * 1e6, result.tensione_condensatore / 1000, "r-")
            axes[1].set_ylabel("Tensione (kV)")
            axes[1].grid(True)

            # Resistenza plasma
            axes[2].plot(result.tempo * 1e6, result.resistenza_plasma, "g-")
            axes[2].set_ylabel("R_plasma (Ω)")
            axes[2].set_xlabel("Tempo (μs)")
            axes[2].grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nMatplotlib non disponibile per grafico")


if __name__ == "__main__":
    main()
