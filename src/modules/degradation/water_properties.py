# Proprietà fisiche dell'acqua per ESWT
"""
Modello delle proprietà fisiche dell'acqua rilevanti per ESWT.

Questo modulo implementa:
    - Tensione superficiale vs soluti e temperatura
    - Potenziale di rottura dielettrica (breakdown)
    - Costante dielettrica vs temperatura
    - Viscosità vs temperatura

Le proprietà dell'acqua influenzano:
    - Formazione del plasma (breakdown dielettrico)
    - Dinamica delle bolle (tensione superficiale)
    - Propagazione onde d'urto (densità, velocità suono)
    - Cavitazione (tensione superficiale, pressione vapore)

Riferimenti:
    - Prompt.md sezione 3
    - CRC Handbook of Chemistry and Physics
    - Ogden et al. (2001) - Parametri cavitazione
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ...core.units import ureg, Q_


@dataclass
class WaterPropertiesState:
    """
    Proprietà fisiche istantanee dell'acqua.

    Attributi:
        temperatura: Temperatura (°C)
        tensione_superficiale: Tensione superficiale (mN/m)
        costante_dielettrica: Costante dielettrica relativa (adim.)
        viscosita: Viscosità dinamica (mPa·s)
        densita: Densità (kg/m³)
        velocita_suono: Velocità del suono (m/s)
        pressione_vapore: Pressione di vapore saturo (kPa)
        V_breakdown: Potenziale di breakdown (kV)
    """

    temperatura: "Q_"
    tensione_superficiale: "Q_"
    costante_dielettrica: float
    viscosita: "Q_"
    densita: "Q_"
    velocita_suono: "Q_"
    pressione_vapore: "Q_"
    V_breakdown: "Q_"

    def to_dict(self) -> dict:
        return {
            "temperatura_C": self.temperatura.to("degC").magnitude,
            "tensione_superficiale_mN_m": self.tensione_superficiale.to("mN/m").magnitude,
            "costante_dielettrica": self.costante_dielettrica,
            "viscosita_mPa_s": self.viscosita.to("mPa*s").magnitude,
            "densita_kg_m3": self.densita.to("kg/m^3").magnitude,
            "velocita_suono_m_s": self.velocita_suono.to("m/s").magnitude,
            "pressione_vapore_kPa": self.pressione_vapore.to("kPa").magnitude,
            "V_breakdown_kV": self.V_breakdown.to("kV").magnitude,
        }


class WaterPropertiesModel:
    """
    Modello delle proprietà fisiche dell'acqua per ESWT.

    Calcola le proprietà dell'acqua in funzione di:
    - Temperatura
    - Concentrazione di soluti (detriti, ioni)
    - Gap inter-elettrodo

    Le equazioni sono basate su correlazioni empiriche
    validate per il range di interesse ESWT (20-40°C).

    Esempio:
        >>> model = WaterPropertiesModel()
        >>> props = model.calcola_proprieta(T=Q_(25, "degC"))
        >>> print(f"Tensione superficiale: {props.tensione_superficiale}")
    """

    # Costanti per acqua pura a 25°C
    GAMMA_0 = 71.97       # mN/m - tensione superficiale
    EPSILON_0 = 78.4      # Costante dielettrica relativa
    ETA_0 = 0.890         # mPa·s - viscosità
    RHO_0 = 997.05        # kg/m³ - densità
    C_0 = 1497            # m/s - velocità suono
    P_VAP_0 = 3.17        # kPa - pressione vapore

    def __init__(self):
        """Inizializza il modello."""
        pass

    def tensione_superficiale(
        self,
        temperatura: "Q_",
        concentrazione_soluti: "Q_" = None
    ) -> "Q_":
        """
        Calcola la tensione superficiale dell'acqua.

        Equazione di Eötvös modificata:
            γ(T) = γ₀ · (1 - T/T_c)^n

        Con correzione per soluti:
            γ = γ(T) - k_st · c

        Parametri:
            temperatura: Temperatura (°C)
            concentrazione_soluti: Concentrazione soluti (mg/L)

        Ritorna:
            Tensione superficiale (mN/m)
        """
        T = temperatura.to("degC").magnitude
        T_c = 374.0  # °C - temperatura critica acqua
        n = 1.256    # Esponente empirico

        # Tensione superficiale vs temperatura
        gamma_T = self.GAMMA_0 * ((T_c - T) / (T_c - 25)) ** n

        # Correzione per soluti
        if concentrazione_soluti is not None:
            c = concentrazione_soluti.to("mg/L").magnitude
            # k_st negativo per la maggior parte dei sali (aumentano γ)
            # ma detriti metallici tendono a ridurre leggermente γ
            k_st = 0.001  # mN·L/(m·mg)
            gamma = gamma_T - k_st * c
        else:
            gamma = gamma_T

        # Limiti fisici
        gamma = np.clip(gamma, 20, 80)

        return Q_(gamma, "mN/m")

    def costante_dielettrica(
        self,
        temperatura: "Q_",
        conducibilita: "Q_" = None
    ) -> float:
        """
        Calcola la costante dielettrica relativa dell'acqua.

        Correlazione:
            ε(T) = ε₀ · [1 - α_ε · (T - 25)]

        dove α_ε ≈ 0.004 /°C

        Parametri:
            temperatura: Temperatura (°C)
            conducibilita: Conducibilità elettrica (μS/cm) - per correzione

        Ritorna:
            Costante dielettrica relativa (adimensionale)
        """
        T = temperatura.to("degC").magnitude

        # Coefficiente temperatura
        alpha_eps = 0.004  # /°C

        epsilon = self.EPSILON_0 * (1 - alpha_eps * (T - 25))

        # Correzione per conducibilità (effetto ioni)
        if conducibilita is not None:
            sigma = conducibilita.to("uS/cm").magnitude
            # Alta conducibilità → leggera riduzione ε
            epsilon *= (1 - 0.0001 * sigma)

        # Limiti fisici
        epsilon = np.clip(epsilon, 50, 90)

        return float(epsilon)

    def potenziale_rottura_dielettrica(
        self,
        gap: "Q_",
        conducibilita: "Q_",
        temperatura: "Q_"
    ) -> "Q_":
        """
        Calcola il potenziale di breakdown dielettrico dell'acqua.

        Il breakdown in acqua segue un modello tipo Paschen modificato:
            V_bd = A · d^n / ln(1 + B·d/σ)

        dove:
            - d = gap inter-elettrodo
            - σ = conducibilità
            - A, B, n = costanti empiriche

        Alta conducibilità → breakdown più facile (V_bd più basso)

        Parametri:
            gap: Gap inter-elettrodo (mm)
            conducibilita: Conducibilità (μS/cm)
            temperatura: Temperatura (°C)

        Ritorna:
            Potenziale di breakdown (kV)
        """
        d = gap.to("mm").magnitude
        sigma = conducibilita.to("uS/cm").magnitude
        T = temperatura.to("degC").magnitude

        # Costanti empiriche per acqua
        # Basate su letteratura scariche in acqua
        A = 15.0   # kV·mm^(-n)
        B = 0.1    # mm·cm/μS
        n = 0.5    # Esponente gap (~ radice quadrata)

        # Evita log(0)
        sigma = max(sigma, 0.1)

        # Modello base
        denominator = np.log(1 + B * d / sigma)
        denominator = max(denominator, 0.1)

        V_bd = A * (d ** n) / denominator

        # Correzione temperatura
        # Acqua più calda → breakdown più facile
        alpha_T = 0.01  # /°C
        V_bd *= (1 - alpha_T * (T - 25))

        # Limiti fisici
        V_bd = np.clip(V_bd, 1, 50)

        return Q_(V_bd, "kV")

    def viscosita(self, temperatura: "Q_") -> "Q_":
        """
        Calcola la viscosità dinamica dell'acqua.

        Equazione di Vogel:
            η(T) = A · exp(B / (T + C))

        Parametri:
            temperatura: Temperatura (°C)

        Ritorna:
            Viscosità dinamica (mPa·s)
        """
        T = temperatura.to("degC").magnitude

        # Costanti Vogel per acqua
        A = 0.02414  # mPa·s
        B = 247.8    # °C
        C = 133.15   # °C

        eta = A * np.exp(B / (T + C))

        return Q_(eta, "mPa*s")

    def densita(self, temperatura: "Q_") -> "Q_":
        """
        Calcola la densità dell'acqua.

        Polinomio di Kell:
            ρ(T) = a₀ + a₁·T + a₂·T² + ...

        Parametri:
            temperatura: Temperatura (°C)

        Ritorna:
            Densità (kg/m³)
        """
        T = temperatura.to("degC").magnitude

        # Coefficienti polinomio (semplificato)
        # Valido per 0-40°C con buona precisione
        a0 = 999.83
        a1 = 0.068
        a2 = -0.0085

        rho = a0 + a1 * T + a2 * T**2

        return Q_(rho, "kg/m^3")

    def velocita_suono(self, temperatura: "Q_") -> "Q_":
        """
        Calcola la velocità del suono in acqua.

        Equazione di Marczak:
            c(T) = c₀ + α·(T-25) + β·(T-25)²

        Parametri:
            temperatura: Temperatura (°C)

        Ritorna:
            Velocità del suono (m/s)
        """
        T = temperatura.to("degC").magnitude

        # Coefficienti per acqua pura
        c0 = 1497   # m/s a 25°C
        alpha = 4.6  # m/s/°C
        beta = -0.05 # m/s/°C²

        dT = T - 25
        c = c0 + alpha * dT + beta * dT**2

        return Q_(c, "m/s")

    def pressione_vapore(self, temperatura: "Q_") -> "Q_":
        """
        Calcola la pressione di vapore saturo dell'acqua.

        Equazione di Antoine:
            log₁₀(P) = A - B / (C + T)

        Parametri:
            temperatura: Temperatura (°C)

        Ritorna:
            Pressione di vapore (kPa)
        """
        T = temperatura.to("degC").magnitude

        # Costanti Antoine per acqua (P in mmHg, T in °C)
        A = 8.07131
        B = 1730.63
        C = 233.426

        log_P_mmHg = A - B / (C + T)
        P_mmHg = 10 ** log_P_mmHg
        P_kPa = P_mmHg * 0.133322  # Conversione mmHg -> kPa

        return Q_(P_kPa, "kPa")

    def calcola_proprieta(
        self,
        temperatura: "Q_",
        conducibilita: "Q_" = None,
        concentrazione_soluti: "Q_" = None,
        gap: "Q_" = None
    ) -> WaterPropertiesState:
        """
        Calcola tutte le proprietà dell'acqua.

        Parametri:
            temperatura: Temperatura (°C)
            conducibilita: Conducibilità (μS/cm)
            concentrazione_soluti: Concentrazione soluti (mg/L)
            gap: Gap inter-elettrodo per V_breakdown (mm)

        Ritorna:
            WaterPropertiesState con tutte le proprietà
        """
        # Default values
        if conducibilita is None:
            conducibilita = Q_(5, "uS/cm")
        if gap is None:
            gap = Q_(5, "mm")

        # Calcola proprietà
        gamma = self.tensione_superficiale(temperatura, concentrazione_soluti)
        epsilon = self.costante_dielettrica(temperatura, conducibilita)
        eta = self.viscosita(temperatura)
        rho = self.densita(temperatura)
        c = self.velocita_suono(temperatura)
        p_vap = self.pressione_vapore(temperatura)
        V_bd = self.potenziale_rottura_dielettrica(gap, conducibilita, temperatura)

        return WaterPropertiesState(
            temperatura=temperatura,
            tensione_superficiale=gamma,
            costante_dielettrica=epsilon,
            viscosita=eta,
            densita=rho,
            velocita_suono=c,
            pressione_vapore=p_vap,
            V_breakdown=V_bd,
        )

    def impatto_soluti_su_plasma_energy(
        self,
        energia_nominale: "Q_",
        V_bd_attuale: "Q_",
        V_bd_nominale: "Q_"
    ) -> "Q_":
        """
        Calcola l'effetto dei soluti sull'energia del plasma.

        Se V_bd diminuisce, il breakdown avviene prima e
        l'energia trasferita al plasma potrebbe variare.

        Modello semplificato:
            E_plasma = E_nom · (V_bd_att / V_bd_nom)^α

        dove α < 1 indica che l'energia non scala linearmente.

        Parametri:
            energia_nominale: Energia plasma nominale (J)
            V_bd_attuale: Potenziale breakdown attuale (kV)
            V_bd_nominale: Potenziale breakdown nominale (kV)

        Ritorna:
            Energia plasma effettiva (J)
        """
        E_nom = energia_nominale.to("J").magnitude
        V_att = V_bd_attuale.to("kV").magnitude
        V_nom = V_bd_nominale.to("kV").magnitude

        if V_nom <= 0:
            return energia_nominale

        # Esponente empirico
        alpha = 0.7

        ratio = V_att / V_nom
        ratio = np.clip(ratio, 0.5, 1.5)

        E_eff = E_nom * (ratio ** alpha)

        return Q_(E_eff, "J")

    def soglia_cavitazione(
        self,
        tensione_superficiale: "Q_",
        temperatura: "Q_"
    ) -> "Q_":
        """
        Calcola la soglia di pressione negativa per cavitazione.

        La soglia di cavitazione dipende da:
        - Tensione superficiale (nucleazione)
        - Temperatura (pressione vapore)
        - Nuclei di gas presenti

        Modello semplificato:
            P_cav = -2γ/R_min + P_vap

        dove R_min è il raggio minimo dei nuclei di gas.

        Parametri:
            tensione_superficiale: Tensione superficiale (mN/m)
            temperatura: Temperatura (°C)

        Ritorna:
            Pressione soglia cavitazione (MPa, negativa)
        """
        gamma = tensione_superficiale.to("N/m").magnitude
        p_vap = self.pressione_vapore(temperatura).to("Pa").magnitude

        # Raggio minimo nuclei (stima)
        R_min = 1e-6  # 1 μm

        # Pressione di Laplace
        P_laplace = 2 * gamma / R_min

        # Soglia cavitazione
        P_cav = p_vap - P_laplace

        # Converti in MPa (negativo = tensione)
        P_cav_MPa = P_cav / 1e6

        # Per acqua pulita, tipicamente -10 a -20 MPa
        # Per acqua con nuclei, può essere molto più bassa
        P_cav_MPa = np.clip(P_cav_MPa, -20, -0.1)

        return Q_(P_cav_MPa, "MPa")

    def __repr__(self) -> str:
        return "WaterPropertiesModel()"
