# Modelli di riflettore per ESWT
"""
Modelli geometrici di riflettori per dispositivi ESWT.

Implementa due tipologie principali:
    - Riflettore ellissoidale: due fuochi (sorgente + target)
    - Riflettore parabolico: fuoco singolo + onde piane

Geometria ellisse:
    - Equazione: x²/a² + y²/b² = 1
    - Proprietà: a² = b² + c² dove c = distanza centro-fuoco
    - F1 = fuoco primario (spark gap)
    - F2 = fuoco secondario (target terapeutico)

Geometria parabola:
    - Equazione: y² = 4px (p = parametro focale)
    - Onde riflesse sono parallele all'asse

Riferimenti:
    - Ogden et al. (2001) - Table 1 dimensioni focali OssaTron
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from ...core.units import ureg, Q_


class ReflectorType(Enum):
    """Tipo di riflettore."""

    ELLIPTICAL = "ellittico"
    PARABOLIC = "parabolico"


@dataclass
class ReflectorGeometry:
    """
    Geometria base di un riflettore.

    Attributi:
        tipo: Tipo di riflettore (ellittico/parabolico)
        apertura: Diametro di apertura (mm)
        profondita: Profondità del riflettore (mm)
        materiale: Materiale del riflettore (default: acciaio)
    """

    tipo: ReflectorType
    apertura: "Q_"
    profondita: "Q_"
    materiale: str = "acciaio_inox"

    @property
    def area_apertura(self) -> "Q_":
        """Area dell'apertura circolare."""
        r = self.apertura.to("mm").magnitude / 2
        return Q_(np.pi * r**2, "mm^2")


class BaseReflector(ABC):
    """Classe base astratta per riflettori."""

    def __init__(self, geometria: ReflectorGeometry):
        self.geometria = geometria

    @abstractmethod
    def posizione_fuoco_primario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """Ritorna la posizione (x, y, z) del fuoco primario."""
        pass

    @abstractmethod
    def posizione_fuoco_secondario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """Ritorna la posizione (x, y, z) del fuoco secondario."""
        pass

    @abstractmethod
    def distanza_focale(self) -> "Q_":
        """Ritorna la distanza tra i due fuochi."""
        pass

    @abstractmethod
    def guadagno_geometrico(self) -> float:
        """Ritorna il guadagno geometrico del riflettore."""
        pass

    @abstractmethod
    def calcola_percorso_raggio(
        self, angolo_emissione: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcola il percorso di un raggio dal fuoco primario."""
        pass


class EllipticalReflector(BaseReflector):
    """
    Riflettore ellissoidale per focalizzazione ESWT.

    Il riflettore ellissoidale ha due fuochi:
    - F1: fuoco primario dove si genera la scarica (spark gap)
    - F2: fuoco secondario dove si concentra l'energia (target)

    La proprietà fondamentale è che tutti i raggi emessi da F1
    convergono in F2 dopo la riflessione.

    Parametri:
        semiasse_maggiore: Semiasse a dell'ellisse (mm)
        semiasse_minore: Semiasse b dell'ellisse (mm)
        o in alternativa:
        apertura: Diametro di apertura (mm)
        distanza_fuochi: Distanza tra F1 e F2 (mm)
    """

    def __init__(
        self,
        semiasse_maggiore: "Q_" = None,
        semiasse_minore: "Q_" = None,
        apertura: "Q_" = None,
        distanza_fuochi: "Q_" = None,
        profondita: "Q_" = None,
    ):
        # Calcola parametri ellisse
        if semiasse_maggiore is not None and semiasse_minore is not None:
            self.a = semiasse_maggiore.to("mm").magnitude
            self.b = semiasse_minore.to("mm").magnitude
            self.c = np.sqrt(self.a**2 - self.b**2)
        elif apertura is not None and distanza_fuochi is not None:
            # Calcola da apertura e distanza fuochi
            d = apertura.to("mm").magnitude / 2  # Raggio apertura
            f = distanza_fuochi.to("mm").magnitude / 2  # Semidistanza focale
            self.c = f
            # Per un'ellisse con apertura d all'altezza del vertice
            # b = d (approssimazione per ellissi poco eccentriche)
            self.b = d
            self.a = np.sqrt(self.b**2 + self.c**2)
        else:
            # Valori default tipici OssaTron
            self.a = 100.0  # mm
            self.b = 60.0  # mm
            self.c = np.sqrt(self.a**2 - self.b**2)

        # Profondità
        if profondita is not None:
            prof = profondita.to("mm").magnitude
        else:
            prof = self.a - self.c  # Profondità dal vertice al fuoco F1

        # Crea geometria
        geometria = ReflectorGeometry(
            tipo=ReflectorType.ELLIPTICAL,
            apertura=Q_(2 * self.b, "mm"),
            profondita=Q_(prof, "mm"),
        )
        super().__init__(geometria)

    @property
    def eccentricita(self) -> float:
        """Eccentricità dell'ellisse (e = c/a)."""
        return self.c / self.a

    @property
    def semiasse_maggiore(self) -> "Q_":
        """Semiasse maggiore a."""
        return Q_(self.a, "mm")

    @property
    def semiasse_minore(self) -> "Q_":
        """Semiasse minore b."""
        return Q_(self.b, "mm")

    def posizione_fuoco_primario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """
        Posizione del fuoco primario F1 (sorgente).

        Convenzione: l'asse dell'ellisse è lungo z,
        F1 è il fuoco più vicino all'apertura.
        """
        return (Q_(0, "mm"), Q_(0, "mm"), Q_(-self.c, "mm"))

    def posizione_fuoco_secondario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """
        Posizione del fuoco secondario F2 (target).

        F2 è il fuoco più lontano dall'apertura (verso il paziente).
        """
        return (Q_(0, "mm"), Q_(0, "mm"), Q_(self.c, "mm"))

    def distanza_focale(self) -> "Q_":
        """Distanza tra i due fuochi (2c)."""
        return Q_(2 * self.c, "mm")

    def guadagno_geometrico(self) -> float:
        """
        Guadagno geometrico del riflettore.

        Il guadagno è il rapporto tra l'area dell'apertura
        e l'area della zona focale teorica.
        Per ellissi, dipende dall'angolo solido sotteso.
        """
        # Approssimazione: guadagno ∝ (apertura/zona_focale)²
        # Per un riflettore ellissoidale tipico: G ≈ 100-1000
        angolo_solido = 2 * np.pi * (1 - self.c / self.a)
        guadagno = 4 * np.pi / angolo_solido
        return guadagno

    def calcola_percorso_raggio(
        self, angolo_emissione: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola il percorso di un raggio dal fuoco F1.

        Parametri:
            angolo_emissione: Angolo di emissione rispetto all'asse (radianti)

        Ritorna:
            Tuple (array_z, array_r) con le coordinate del percorso
        """
        # Punto di partenza: F1
        z0, r0 = -self.c, 0

        # Direzione iniziale
        dz = np.cos(angolo_emissione)
        dr = np.sin(angolo_emissione)

        # Trova intersezione con ellisse
        # Parametrizzazione: z = z0 + t*dz, r = r0 + t*dr
        # Sostituisci in z²/a² + r²/b² = 1
        A = (dz / self.a) ** 2 + (dr / self.b) ** 2
        B = 2 * (z0 * dz / self.a**2 + r0 * dr / self.b**2)
        C = (z0 / self.a) ** 2 + (r0 / self.b) ** 2 - 1

        discriminante = B**2 - 4 * A * C
        if discriminante < 0:
            return np.array([z0]), np.array([r0])

        t = (-B + np.sqrt(discriminante)) / (2 * A)

        # Punto di riflessione
        z1 = z0 + t * dz
        r1 = r0 + t * dr

        # Dopo riflessione, il raggio va verso F2
        z2, r2 = self.c, 0

        # Costruisci percorso
        z_path = np.array([z0, z1, z2])
        r_path = np.array([r0, r1, r2])

        return z_path, r_path

    def calcola_zona_focale_6dB(self) -> Tuple["Q_", "Q_"]:
        """
        Stima le dimensioni della zona focale a -6dB.

        Basato su dati empirici da Ogden et al. (2001).

        Ritorna:
            Tuple (diametro_laterale, lunghezza_assiale) in mm
        """
        # Modello empirico basato su OssaTron
        # La zona focale dipende dall'eccentricità e dall'apertura
        e = self.eccentricita
        apertura = self.geometria.apertura.to("mm").magnitude

        # Diametro laterale: più stretto per ellissi meno eccentriche
        d_laterale = 0.1 * apertura * (1 + e)  # Tipico: 6-9 mm

        # Lunghezza assiale: più lunga per ellissi più eccentriche
        l_assiale = 0.5 * apertura * (1 + 2 * e)  # Tipico: 44-68 mm

        return (Q_(d_laterale, "mm"), Q_(l_assiale, "mm"))


class ParabolicReflector(BaseReflector):
    """
    Riflettore parabolico per ESWT.

    Il riflettore parabolico ha un solo fuoco geometrico.
    I raggi emessi dal fuoco vengono riflessi parallelamente all'asse.

    Per ottenere focalizzazione, si usa tipicamente una lente acustica
    o si sfrutta la diffrazione naturale.

    Parametri:
        parametro_focale: Distanza fuoco-vertice p (mm)
        apertura: Diametro di apertura (mm)
    """

    def __init__(
        self,
        parametro_focale: "Q_" = None,
        apertura: "Q_" = None,
    ):
        if parametro_focale is not None:
            self.p = parametro_focale.to("mm").magnitude
        else:
            self.p = 30.0  # mm default

        if apertura is not None:
            self.apertura_mm = apertura.to("mm").magnitude
        else:
            self.apertura_mm = 100.0  # mm default

        # Calcola profondità: y² = 4px → x = y²/(4p)
        r = self.apertura_mm / 2
        prof = r**2 / (4 * self.p)

        geometria = ReflectorGeometry(
            tipo=ReflectorType.PARABOLIC,
            apertura=Q_(self.apertura_mm, "mm"),
            profondita=Q_(prof, "mm"),
        )
        super().__init__(geometria)

        # Distanza focale effettiva (per lente acustica)
        self._distanza_focale_effettiva = Q_(150, "mm")  # Default

    @property
    def parametro_focale(self) -> "Q_":
        """Parametro focale p."""
        return Q_(self.p, "mm")

    def set_distanza_focale_effettiva(self, distanza: "Q_"):
        """Imposta la distanza focale effettiva (con lente)."""
        self._distanza_focale_effettiva = distanza

    def posizione_fuoco_primario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """Posizione del fuoco geometrico (sorgente)."""
        return (Q_(0, "mm"), Q_(0, "mm"), Q_(self.p, "mm"))

    def posizione_fuoco_secondario(self) -> Tuple["Q_", "Q_", "Q_"]:
        """
        Posizione del fuoco secondario (target).

        Per parabola, il fuoco secondario è determinato dalla lente
        acustica o dalla geometria di focalizzazione aggiuntiva.
        """
        d = self._distanza_focale_effettiva.to("mm").magnitude
        return (Q_(0, "mm"), Q_(0, "mm"), Q_(d, "mm"))

    def distanza_focale(self) -> "Q_":
        """Distanza focale effettiva."""
        return self._distanza_focale_effettiva

    def guadagno_geometrico(self) -> float:
        """
        Guadagno geometrico del riflettore parabolico.

        Per parabole, il guadagno dipende dal rapporto f/D
        (parametro focale / diametro).
        """
        f_D = self.p / self.apertura_mm
        # Guadagno approssimato
        guadagno = (np.pi * self.apertura_mm / (4 * f_D)) ** 2 / (4 * np.pi)
        return max(guadagno, 1.0)

    def calcola_percorso_raggio(
        self, angolo_emissione: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola il percorso di un raggio dal fuoco.

        Per parabola, i raggi riflessi sono paralleli all'asse.
        """
        # Punto di partenza: fuoco
        z0, r0 = self.p, 0

        # Direzione iniziale
        dz = np.cos(angolo_emissione)
        dr = np.sin(angolo_emissione)

        # Trova intersezione con parabola: r² = 4*p*z
        # r = r0 + t*dr, z = z0 + t*dz
        # (r0 + t*dr)² = 4*p*(z0 + t*dz)
        A = dr**2
        B = 2 * r0 * dr - 4 * self.p * dz
        C = r0**2 - 4 * self.p * z0

        if abs(A) < 1e-10:
            if abs(B) < 1e-10:
                return np.array([z0]), np.array([r0])
            t = -C / B
        else:
            discriminante = B**2 - 4 * A * C
            if discriminante < 0:
                return np.array([z0]), np.array([r0])
            t = (-B + np.sqrt(discriminante)) / (2 * A)

        # Punto di riflessione
        z1 = z0 + t * dz
        r1 = abs(r0 + t * dr)

        # Dopo riflessione, raggio parallelo all'asse
        z2 = z1 + 200  # mm, estendi il raggio
        r2 = r1  # Parallelo

        z_path = np.array([z0, z1, z2])
        r_path = np.array([r0, r1, r2])

        return z_path, r_path

    def calcola_zona_focale_6dB(self) -> Tuple["Q_", "Q_"]:
        """
        Stima le dimensioni della zona focale a -6dB.

        Per parabole con lente, la zona focale è generalmente
        più ampia di quella ellissoidale.
        """
        # Modello empirico
        f = self._distanza_focale_effettiva.to("mm").magnitude
        apertura = self.apertura_mm

        # Zona focale più diffusa rispetto all'ellisse
        lambda_eff = 1.5  # mm, lunghezza d'onda effettiva in acqua
        d_laterale = 1.22 * lambda_eff * f / apertura * 2
        l_assiale = 2 * lambda_eff * (f / apertura) ** 2 * 8

        return (Q_(max(d_laterale, 5), "mm"), Q_(max(l_assiale, 30), "mm"))


def crea_riflettore(
    tipo: ReflectorType,
    apertura: "Q_" = None,
    distanza_fuochi: "Q_" = None,
    parametro_focale: "Q_" = None,
    **kwargs,
) -> BaseReflector:
    """
    Factory function per creare riflettori.

    Parametri:
        tipo: ReflectorType.ELLIPTICAL o ReflectorType.PARABOLIC
        apertura: Diametro di apertura (mm)
        distanza_fuochi: Per ellisse, distanza tra i fuochi (mm)
        parametro_focale: Per parabola, distanza fuoco-vertice (mm)

    Ritorna:
        Istanza di EllipticalReflector o ParabolicReflector
    """
    if tipo == ReflectorType.ELLIPTICAL:
        return EllipticalReflector(
            apertura=apertura, distanza_fuochi=distanza_fuochi, **kwargs
        )
    elif tipo == ReflectorType.PARABOLIC:
        return ParabolicReflector(
            apertura=apertura, parametro_focale=parametro_focale, **kwargs
        )
    else:
        raise ValueError(f"Tipo riflettore non supportato: {tipo}")
