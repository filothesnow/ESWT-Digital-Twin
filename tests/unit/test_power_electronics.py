# Test unitari per il modulo Power Electronics
"""
Test per i componenti del modulo power_electronics:
    - Capacitor
    - PlasmaChannel
    - DischargeSimulator
    - Funzioni di calcolo energia
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Import dal modulo da testare
from src.core.units import ureg, Q_
from src.modules.power_electronics import (
    Capacitor,
    PlasmaChannel,
    PlasmaModel,
    DischargeSimulator,
    calcola_energia_condensatore,
    calcola_energia_rilasciata,
    calcola_ripartizione_energia,
    simula_scarica_rlc_analitica,
    crea_modello_costante,
)


class TestCapacitor:
    """Test per la classe Capacitor."""

    def test_creazione_condensatore(self):
        """Verifica creazione condensatore con parametri validi."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))

        assert cap.capacita.to("uF").magnitude == pytest.approx(1.0)
        assert cap.tensione_max.to("kV").magnitude == pytest.approx(20.0)
        assert cap.tensione.to("kV").magnitude == pytest.approx(20.0)

    def test_energia_immagazzinata(self):
        """Verifica calcolo energia E = 0.5 * C * V²."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        energia = cap.energia_immagazzinata

        # E = 0.5 * 1e-6 * (20000)² = 200 J
        assert energia.to("J").magnitude == pytest.approx(200.0, rel=1e-6)

    def test_energia_max(self):
        """Verifica che energia_max == energia_immagazzinata a piena carica."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))

        assert cap.energia_max == cap.energia_immagazzinata

    def test_carica(self):
        """Verifica calcolo carica Q = C * V."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        carica = cap.carica

        # Q = 1e-6 * 20000 = 0.02 C
        assert carica.to("C").magnitude == pytest.approx(0.02, rel=1e-6)

    def test_energia_rilasciata(self):
        """Verifica calcolo energia rilasciata durante scarica parziale."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        energia = cap.energia_rilasciata(Q_(10, "kV"))

        # ΔE = 0.5 * 1e-6 * (20000² - 10000²) = 150 J
        assert energia.to("J").magnitude == pytest.approx(150.0, rel=1e-6)

    def test_scarica_condensatore(self):
        """Verifica che la scarica riduca la tensione."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        v_iniziale = cap.tensione.to("V").magnitude

        # Scarica con 1 kA per 1 μs
        cap.scarica(Q_(1, "kA"), Q_(1, "us"))
        v_finale = cap.tensione.to("V").magnitude

        # dV = -I*dt/C = -1000 * 1e-6 / 1e-6 = -1000 V
        assert v_finale < v_iniziale
        assert (v_iniziale - v_finale) == pytest.approx(1000, rel=1e-3)

    def test_reset_condensatore(self):
        """Verifica che reset riporti alla tensione massima."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        cap.scarica(Q_(1, "kA"), Q_(1, "us"))
        cap.reset()

        assert cap.tensione == cap.tensione_max


class TestPlasmaChannel:
    """Test per la classe PlasmaChannel."""

    def test_creazione_plasma_default(self):
        """Verifica creazione plasma con parametri di default."""
        plasma = PlasmaChannel()

        assert plasma.gap.to("mm").magnitude == pytest.approx(5.0)
        assert plasma.modello == PlasmaModel.ROMPE_WEIZEL

    def test_resistenza_costante(self):
        """Verifica modello a resistenza costante."""
        plasma = PlasmaChannel(
            modello=PlasmaModel.COSTANTE, resistenza_iniziale=Q_(0.5, "ohm")
        )

        r1 = plasma.calcola_resistenza(Q_(1, "kA"))
        r2 = plasma.calcola_resistenza(Q_(10, "kA"))

        assert r1.to("ohm").magnitude == pytest.approx(0.5)
        assert r2.to("ohm").magnitude == pytest.approx(0.5)

    def test_resistenza_rompe_weizel(self):
        """Verifica che R diminuisca con l'aumento della corrente."""
        plasma = PlasmaChannel(
            modello=PlasmaModel.ROMPE_WEIZEL,
            resistenza_iniziale=Q_(1, "ohm"),
            corrente_riferimento=Q_(1, "kA"),
            esponente=0.5,
        )

        r_bassa = plasma.calcola_resistenza(Q_(0.5, "kA")).magnitude
        r_alta = plasma.calcola_resistenza(Q_(2, "kA")).magnitude

        # R deve diminuire con l'aumento di I
        assert r_alta < r_bassa

    def test_potenza_dissipata(self):
        """Verifica calcolo potenza P = I² * R."""
        plasma = crea_modello_costante(Q_(0.5, "ohm"))

        potenza = plasma.calcola_potenza_dissipata(Q_(10, "kA"))

        # P = (10000)² * 0.5 = 50 MW
        assert potenza.to("MW").magnitude == pytest.approx(50, rel=1e-3)


class TestDischargeSimulator:
    """Test per la classe DischargeSimulator."""

    def test_creazione_simulatore(self):
        """Verifica creazione simulatore."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = PlasmaChannel()
        sim = DischargeSimulator(cap, plasma, Q_(5, "uH"))

        assert sim.induttanza.to("uH").magnitude == pytest.approx(5.0)

    def test_frequenza_naturale(self):
        """Verifica calcolo frequenza naturale f_0 = 1/(2π√(LC))."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = PlasmaChannel()
        sim = DischargeSimulator(cap, plasma, Q_(1, "uH"))

        # f_0 = 1/(2π√(1e-6 * 1e-6)) = 1/(2π * 1e-6) ≈ 159 kHz
        f = sim.frequenza_naturale
        assert f.to("kHz").magnitude == pytest.approx(159.15, rel=1e-2)

    def test_periodo_naturale(self):
        """Verifica calcolo periodo naturale T = 2π√(LC)."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = PlasmaChannel()
        sim = DischargeSimulator(cap, plasma, Q_(1, "uH"))

        # T = 2π√(1e-6 * 1e-6) = 2π * 1e-6 ≈ 6.28 μs
        t = sim.periodo_naturale
        assert t.to("us").magnitude == pytest.approx(6.28, rel=1e-2)

    def test_simulazione_esegue(self):
        """Verifica che la simulazione si esegua senza errori."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = crea_modello_costante(Q_(0.1, "ohm"))
        sim = DischargeSimulator(cap, plasma, Q_(5, "uH"))

        result = sim.simula(Q_(20, "us"))

        # Verifica che i risultati siano presenti
        assert len(result.tempo) > 0
        assert len(result.corrente) == len(result.tempo)
        assert len(result.tensione_condensatore) == len(result.tempo)
        assert result.corrente_picco.magnitude > 0

    def test_conservazione_energia(self):
        """
        Verifica conservazione energia nel circuito RLC.

        L'energia iniziale deve essere circa uguale alla somma di:
        - Energia residua nel condensatore
        - Energia dissipata nel plasma
        """
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = crea_modello_costante(Q_(0.5, "ohm"))
        sim = DischargeSimulator(cap, plasma, Q_(5, "uH"))

        result = sim.simula(Q_(50, "us"))

        e_iniziale = cap.energia_max.to("J").magnitude
        e_plasma = result.energia_plasma.to("J").magnitude

        # Energia residua nel condensatore
        v_finale = result.tensione_condensatore[-1]
        e_residua = 0.5 * cap.capacita.magnitude * v_finale**2

        # Energia nell'induttanza (idealmente 0 a fine simulazione)
        i_finale = result.corrente[-1]
        e_induttore = 0.5 * sim.induttanza.magnitude * i_finale**2

        e_totale = e_plasma + e_residua + e_induttore

        # Tolleranza dell'1%
        assert e_totale == pytest.approx(e_iniziale, rel=0.01)


class TestSoluzioneAnalitica:
    """Test per la soluzione analitica RLC."""

    def test_caso_sottosmorzato(self):
        """Verifica soluzione per circuito sottosmorzato (oscillante)."""
        t, i, v = simula_scarica_rlc_analitica(
            capacita=Q_(1, "uF"),
            induttanza=Q_(10, "uH"),
            resistenza=Q_(0.1, "ohm"),  # Bassa resistenza → sottosmorzato
            tensione_iniziale=Q_(1, "kV"),
            durata=Q_(100, "us"),
        )

        # La corrente deve oscillare (cambiare segno)
        assert np.any(i > 0) and np.any(i < 0)

    def test_confronto_numerico_analitico(self):
        """Confronta soluzione numerica con analitica (R costante)."""
        cap = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
        plasma = crea_modello_costante(Q_(0.5, "ohm"))
        sim = DischargeSimulator(cap, plasma, Q_(5, "uH"))

        result_num = sim.simula(Q_(30, "us"))

        t_an, i_an, v_an = simula_scarica_rlc_analitica(
            capacita=Q_(1, "uF"),
            induttanza=Q_(5, "uH"),
            resistenza=Q_(0.5, "ohm"),
            tensione_iniziale=Q_(20, "kV"),
            durata=Q_(30, "us"),
            n_punti=len(result_num.tempo),
        )

        # Interpola soluzione analitica sui tempi numerici
        i_an_interp = np.interp(result_num.tempo, t_an, i_an)

        # Confronta con tolleranza del 5%
        # (la discrepanza può derivare da errori numerici)
        correlazione = np.corrcoef(result_num.corrente, i_an_interp)[0, 1]
        assert correlazione > 0.99


class TestEnergia:
    """Test per le funzioni di calcolo energia."""

    def test_calcola_energia_condensatore(self):
        """Verifica formula E = 0.5 * C * V²."""
        energia = calcola_energia_condensatore(Q_(1, "uF"), Q_(20, "kV"))

        assert energia.to("J").magnitude == pytest.approx(200.0, rel=1e-6)

    def test_calcola_energia_rilasciata(self):
        """Verifica formula ΔE = 0.5 * C * (V_i² - V_f²)."""
        energia = calcola_energia_rilasciata(
            Q_(1, "uF"), Q_(20, "kV"), Q_(10, "kV")
        )

        assert energia.to("J").magnitude == pytest.approx(150.0, rel=1e-6)

    def test_ripartizione_energia_chen(self):
        """Verifica ripartizione 10% riscaldamento, 90% shockwave."""
        breakdown = calcola_ripartizione_energia(Q_(100, "J"))

        assert breakdown.energia_riscaldamento.to("J").magnitude == pytest.approx(10.0)
        assert breakdown.energia_shockwave.to("J").magnitude == pytest.approx(90.0)
        assert breakdown.efficienza == pytest.approx(0.9)


# Test di validazione con dati Chen 2010
class TestValidazioneChen:
    """Test di validazione con dati sperimentali Chen et al. (2010)."""

    @pytest.mark.parametrize(
        "energia_J,distanza_cm,pressione_attesa_MPa",
        [
            (3300, 17.5, 8.0),
            (600, 17.5, 4.4),
            (31, 9, 2.8),
            (20, 9, 2.0),
        ],
    )
    def test_ordine_grandezza_pressione(
        self, energia_J, distanza_cm, pressione_attesa_MPa
    ):
        """
        Verifica che le pressioni stimate siano nell'ordine di grandezza corretto.

        Nota: questo non è un test di precisione, ma verifica che il modello
        dia risultati ragionevoli rispetto ai dati sperimentali.
        """
        from src.modules.power_electronics.energy import calcola_pressione_picco_chen

        pressione = calcola_pressione_picco_chen(
            Q_(energia_J, "J"), Q_(distanza_cm, "cm")
        )

        p_calc = pressione.to("MPa").magnitude

        # Verifica ordine di grandezza (fattore 3)
        assert p_calc > pressione_attesa_MPa / 3
        assert p_calc < pressione_attesa_MPa * 3
