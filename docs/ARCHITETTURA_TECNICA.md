# ESWT Digital Twin - Documentazione Tecnica

## Panoramica del Sistema

Il Digital Twin ESWT (Extracorporeal Shock Wave Therapy) simula un dispositivo medico per terapia con onde d'urto focalizzate a generazione elettroidraulica. Il sistema modella l'intera catena fisica dalla scarica elettrica fino all'onda d'urto focalizzata sul tessuto target.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESWT Digital Twin                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │    Power     │    │   Physics    │    │ Degradation  │    │  Control  │ │
│  │ Electronics  │───▶│   Engine     │───▶│   Module     │◀──▶│  Module   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                  │        │
│         └───────────────────┴───────────────────┴──────────────────┘        │
│                                    │                                         │
│                            ┌───────▼───────┐                                │
│                            │   Dashboard   │                                │
│                            │    (Dash)     │                                │
│                            └───────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Power Electronics Module

**Percorso:** `src/modules/power_electronics/`

### 1.1 Condensatore (`capacitor.py`)

Il condensatore immagazzina l'energia elettrica che verrà rilasciata durante la scarica.

**Equazione fondamentale:**
```
E = ½ C V²
```

dove:
- `E` = energia immagazzinata (J)
- `C` = capacità (F)
- `V` = tensione di carica (V)

**Parametri tipici OssaTron:**
| Tensione | Capacità | Energia |
|----------|----------|---------|
| 14 kV    | 1 µF     | 98 J    |
| 20 kV    | 1 µF     | 200 J   |
| 28 kV    | 1 µF     | 392 J   |

### 1.2 Canale di Plasma (`plasma_channel.py`)

Quando la tensione supera la soglia di breakdown dielettrico dell'acqua, si forma un canale di plasma tra gli elettrodi.

**Modello Rompe-Weizel per la resistenza:**
```
R(I) = R₀ × (I₀/I)^α
```

dove:
- `R₀` = resistenza iniziale (~0.2 Ω per gap 5mm)
- `I₀` = corrente di riferimento (1 kA)
- `α` = esponente (tipicamente 0.5)

La resistenza diminuisce all'aumentare della corrente perché il plasma diventa più conduttivo.

### 1.3 Simulazione Scarica (`discharge.py`)

Il circuito RLC sottosmorzato è descritto dall'equazione differenziale:

```
L(dI/dt) + R(I)×I + (1/C)∫I dt = 0
```

**Soluzione analitica (R costante, caso sottosmorzato):**
```
I(t) = (V₀/ωL) × exp(-αt) × sin(ωt)

dove:
  α = R/(2L)           (coefficiente di smorzamento)
  ω = √(1/LC - α²)     (frequenza angolare)
```

**Output della simulazione:**
- `tempo`: array temporale (s)
- `corrente`: forma d'onda corrente (A)
- `tensione_condensatore`: scarica tensione (V)
- `resistenza_plasma`: evoluzione R(t) (Ω)
- `corrente_picco`: massimo raggiunto (~10-15 kA)
- `energia_plasma`: energia dissipata nel plasma (J)

### 1.4 Ripartizione Energia (Chen et al. 2010)

L'energia elettrica si ripartisce in:

```
E_totale = E_shockwave + E_termica + E_perdite

Tipicamente:
  - 90% → onda d'urto (E_shockwave)
  - 8%  → riscaldamento acqua
  - 2%  → perdite ohmiche
```

---

## 2. Physics Engine Module

**Percorso:** `src/modules/physics_engine/`

### 2.1 Dinamica del Plasma (`plasma_dynamics.py`)

#### Espansione della Bolla (Rayleigh-Plesset semplificato)

Quando il plasma si espande rapidamente, crea una bolla che genera l'onda d'urto.

```
R×R'' + (3/2)×R'² = (P_plasma - P_ambiente) / ρ_acqua
```

dove:
- `R` = raggio bolla (m)
- `R'`, `R''` = derivate temporali
- `P_plasma` = pressione interna (~GPa iniziale)
- `P_ambiente` = 101 kPa
- `ρ_acqua` = 998 kg/m³

**Tempo di Rayleigh:**
```
τ_R = 0.915 × R_max × √(ρ/P_ambiente)
```

Questo definisce il tempo caratteristico di espansione/collasso della bolla.

### 2.2 Propagazione Onda d'Urto (`shockwave.py`)

#### Modello Lineare (attenuazione geometrica + assorbimento)

```
P(r) = P₀ × (r₀/r)^n × exp(-α×r)
```

dove:
- `n` = 1 per onda sferica
- `α` = coefficiente di assorbimento (~0.01 /mm in acqua)

#### Modello Non-Lineare (Burgers)

Per pressioni elevate, gli effetti non lineari diventano significativi:

```
∂p/∂x + (1/c₀)×∂p/∂t = (β/2ρc₀³)×p×∂p/∂t + δ×∂²p/∂t²
```

dove:
- `β` = 3.5 (parametro di non-linearità per acqua)
- `δ` = diffusività (viscosità + conduzione termica)

**Soluzione approssimata (weak shock):**
```
P(r) = P₀×(r₀/r)×exp(-α×(r-r₀)) × [1 - β×P₀×(r-r₀)/(2ρc₀³)]⁻¹
```

#### Rise Time Calibrato (Ogden et al. 2001)

Il rise time dell'impulso è calibrato sui dati sperimentali:

```
t_rise = k / √P_peak

Dati OssaTron:
  - 14 kV: ~100 ns
  - 20 kV: ~80 ns
  - 28 kV: ~50 ns
```

### 2.3 Geometria Riflettore (`reflector.py`)

#### Riflettore Ellissoidale

L'ellissoide ha due fuochi: F1 (sorgente) e F2 (target terapeutico).

```
Equazione ellisse:  (z/a)² + (r/b)² = 1

Relazioni geometriche:
  c = √(a² - b²)     (distanza fuochi dal centro)
  e = c/a            (eccentricità)

Proprietà ottica:
  Tutti i raggi da F1 convergono in F2
```

**Guadagno geometrico:**
```
G = (Ω_riflettore / 4π) × η_riflessione

dove Ω = angolo solido sotteso dal riflettore
```

#### Ray Tracing

Per ogni raggio emesso dalla sorgente:
1. Calcola intersezione con ellissoide
2. Calcola normale alla superficie
3. Applica legge di riflessione: `r_riflesso = r_incidente - 2(r·n)n`
4. Verifica convergenza al fuoco F2

### 2.4 Focalizzazione (`focusing.py`)

#### Pressione Focale

```
P_focale = P_sorgente × G × η_propagazione

dove:
  G = guadagno geometrico riflettore
  η_propagazione = exp(-α×d_focale)
```

#### Energy Flux Density (EFD)

L'EFD è il parametro clinico fondamentale:

```
EFD = (1/ρc) × ∫ p²(t) dt    [mJ/mm²]

Classificazione clinica:
  - Bassa energia:   < 0.08 mJ/mm²
  - Media energia:   0.08 - 0.28 mJ/mm²
  - Alta energia:    > 0.28 mJ/mm²
```

#### Zona Focale (-6dB)

La zona focale è definita dove la pressione è ≥ 50% del picco:

```
Diametro laterale (-6dB):  d_lat ≈ λ × F/D
Lunghezza assiale (-6dB):  L_ax ≈ 7 × λ × (F/D)²

dove:
  λ = lunghezza d'onda
  F = distanza focale
  D = apertura riflettore
```

**Dati OssaTron (Ogden Table 1):**
| kV | P_max (MPa) | EFD (mJ/mm²) | d_lat (mm) | L_ax (mm) |
|----|-------------|--------------|------------|-----------|
| 14 | 40.6        | 0.105        | 6.8        | 44.1      |
| 20 | 45.6        | 0.255        | 6.4        | 59.0      |
| 28 | 71.9        | 0.370        | 8.7        | 67.6      |

### 2.5 Impedenza Acustica (`impedance.py`)

L'impedenza acustica determina riflessione/trasmissione alle interfacce:

```
Z = ρ × c    [kg/(m²×s) = Rayl]

Coefficiente di riflessione:
  R = (Z₂ - Z₁)/(Z₂ + Z₁)

Coefficiente di trasmissione:
  T = 2×Z₂/(Z₂ + Z₁)
```

**Impedenze tipiche:**
| Materiale      | Z (MRayl) |
|----------------|-----------|
| Acqua          | 1.48      |
| Tessuto molle  | 1.63      |
| Osso corticale | 7.8       |
| Aria           | 0.0004    |

### 2.6 Cavitazione (`cavitation.py`)

La fase negativa dell'onda può causare cavitazione:

```
Condizione di attivazione:
  |P_negativa| > σ_tensile_dinamica  (~10 MPa per acqua)

Raggio massimo bolla:
  R_max = R₀ × (P₀/P_ambiente)^(1/3)

Energia rilasciata al collasso:
  E_collapse ∝ P_ambiente × R_max³
```

---

## 3. Degradation Module

**Percorso:** `src/modules/degradation/`

### 3.1 Erosione Elettrodi (`electrode.py`)

Gli elettrodi si erodono ad ogni scarica, aumentando il gap.

**Modello di erosione:**
```
Δm = k × Q^α

dove:
  Δm = massa erosa per impulso (mg)
  Q = carica trasferita (C)
  k = coefficiente materiale
  α = esponente (1.0-1.5)
```

**Coefficienti per materiale:**
| Materiale    | k (mg/C^α) |
|--------------|------------|
| Tungsteno    | 0.05       |
| Rame         | 0.15       |
| Acciaio Inox | 0.08       |

**Variazione gap:**
```
Δgap = Δm / (ρ × A_punta)

dove A_punta = area della punta elettrodo
```

### 3.2 Efficienza (`efficiency.py`)

L'efficienza decresce esponenzialmente col numero di impulsi:

```
η(N) = η₀ × exp(-λ×N)

dove:
  η₀ = efficienza iniziale (100%)
  λ = parametro di decadimento (~5×10⁻⁶)
  N = numero impulsi
```

**Soglie operative:**
- η > 70%: Funzionamento normale
- 50% < η < 70%: Attenzione
- η < 50%: Manutenzione richiesta

### 3.3 Chimica dell'Acqua (`water_chemistry.py`)

L'erosione rilascia detriti metallici che modificano le proprietà dell'acqua.

**Accumulo detriti:**
```
c_detriti(N) = Σ(Δm_i) / V_acqua    [mg/L]
```

**Conducibilità:**
```
σ = σ₀ + k_σ × c_detriti

dove:
  σ₀ = 5 µS/cm (acqua deionizzata)
  k_σ = 0.5 µS×L/(cm×mg)
```

**Effetto sulla scarica:**
Alta conducibilità → breakdown più facile, ma anche più energia dissipata nel volume.

### 3.4 Proprietà Acqua (`water_properties.py`)

#### Tensione Superficiale vs Temperatura

```
γ(T) = γ₀ × [(T_c - T)/(T_c - 25)]^n

dove:
  γ₀ = 71.97 mN/m (a 25°C)
  T_c = 374°C (temperatura critica)
  n = 1.256
```

#### Potenziale di Breakdown Dielettrico

```
V_bd = A × d^n / ln(1 + B×d/σ)

dove:
  d = gap (mm)
  σ = conducibilità (µS/cm)
  A, B, n = costanti empiriche
```

### 3.5 Assorbimento Gas (`gas_absorption.py`)

Il palladio assorbe idrogeno prodotto dall'elettrolisi:

```
Cinetica: dC_H/dt = k × (C_max - C_H) × P_H2

Capacità: H/Pd ratio max ≈ 0.7
```

L'assorbimento riduce la densità di bolle di gas, influenzando la cavitazione.

---

## 4. Control Module

**Percorso:** `src/modules/control/`

### 4.1 Strutture Dati (`data_structures.py`)

```python
@dataclass
class ControlState:
    gap_target: Q_      # Gap obiettivo (mm)
    gap_current: Q_     # Gap attuale (mm)
    I_target: Q_        # Corrente target (kA)
    I_feedback: Q_      # Corrente misurata (kA)
    status: ControlStatus  # NOMINAL, WARNING, CRITICAL

@dataclass
class MotorAction:
    impulse_num: int
    gap_before: Q_
    gap_after: Q_
    delta_gap: Q_
    pid_output: Tuple[float, float, float]  # (P, I, D)
```

### 4.2 Controllore PID (`pid_controller.py`)

Il PID regola il gap basandosi sulla corrente di feedback:

```
Logica di controllo:
  - I_picco < I_target → gap troppo grande → avvicina punte
  - I_picco > I_target → gap troppo piccolo → allontana punte

Equazione PID:
  u(t) = Kp×e(t) + Ki×∫e(τ)dτ + Kd×de/dt

dove e(t) = I_target - I_feedback
```

**Parametri tuning ESWT:**
| Parametro | Valore | Unità |
|-----------|--------|-------|
| Kp        | 0.5    | mm/kA |
| Ki        | 0.01   | mm/(kA×s) |
| Kd        | 0.1    | mm×s/kA |
| I_target  | 10     | kA |

**Anti-windup:**
L'integrale è limitato per evitare sovraelongazioni:
```
|integral| ≤ anti_windup_limit
```

### 4.3 Monitor Feedback (`feedback_monitor.py`)

Estrae informazioni dalla scarica per il controllo:

**Stima gap da resistenza:**
```
gap_stimato = gap_ref × (R_misurata / R_ref)

Relazione empirica: R_plasma ∝ gap
```

**Calcolo resistenza media:**
```
R_avg = ∫R(t)×I(t)dt / ∫I(t)dt
```

### 4.4 Stimatore Gap Kalman (`gap_estimator.py`)

Filtro di Kalman a 3 stati per stima robusta:

```
Stato: x = [gap, gap_dot, erosion_rate]ᵀ

Dinamica:
  gap[k+1] = gap[k] + gap_dot[k]×dt + erosion_rate[k] + u[k]
  gap_dot[k+1] = gap_dot[k]
  erosion_rate[k+1] = erosion_rate[k]

Misure:
  z₁ = gap (da resistenza plasma, rumorosa)
  z₂ = gap (da encoder motore, precisa)
```

**Equazioni Kalman:**
```
Predizione:
  x̂⁻ = F×x̂ + B×u
  P⁻ = F×P×Fᵀ + Q

Aggiornamento:
  K = P⁻×Hᵀ×(H×P⁻×Hᵀ + R)⁻¹
  x̂ = x̂⁻ + K×(z - H×x̂⁻)
  P = (I - K×H)×P⁻
```

### 4.5 Logger Motore (`motor_log.py`)

Registra ogni intervento del motore di avvicinamento:

```python
class MotorInterventionLogger:
    def log_intervention(action: MotorAction)
    def export_csv(filename) -> Path
    def export_json(filename) -> Path
    def summary_report() -> dict
```

**Formato CSV:**
```
sequence,impulse,timestamp,gap_before_mm,gap_after_mm,delta_gap_mm,
I_feedback_kA,error_A,pid_p,pid_i,pid_d,status
```

---

## 5. Dashboard

**Percorso:** `src/dashboard/app.py`

### 5.1 Architettura

La dashboard usa Dash (Plotly) con Bootstrap per l'interfaccia:

```
┌─────────────────────────────────────────────────────────────┐
│                    ESWT Digital Twin                         │
├───────────────┬─────────────────────────────────────────────┤
│               │  ┌─────────────┐  ┌─────────────┐           │
│  CONTROLLI    │  │ Geometria   │  │  Corrente   │           │
│  - Riflettore │  │ Riflettore  │  │  Scarica    │           │
│  - Tensione   │  └─────────────┘  └─────────────┘           │
│  - Capacità   │  ┌─────────────┐  ┌─────────────┐           │
│  - Materiale  │  │  Profilo    │  │ Efficienza  │           │
│  - Gap        │  │  Pressione  │  │ vs Impulsi  │           │
│               │  └─────────────┘  └─────────────┘           │
│  [SIMULA]     │  ┌───────────────────────────────┐          │
│               │  │   Log Interventi Motore       │          │
│               │  └───────────────────────────────┘          │
│               │  ┌───────────────────────────────┐          │
│               │  │      Riepilogo Simulazione    │          │
│               │  └───────────────────────────────┘          │
└───────────────┴─────────────────────────────────────────────┘
```

### 5.2 Flusso di Simulazione

```python
def esegui_simulazione():
    # 1. Crea riflettore (ellittico o parabolico)
    riflettore = EllipticalReflector(apertura, distanza_fuochi)

    # 2. Calcola energia
    condensatore = Capacitor(capacita, tensione)
    energia = condensatore.energia_immagazzinata

    # 3. Simula scarica RLC
    simulatore = DischargeSimulator(condensatore, plasma, induttanza)
    risultato_scarica = simulatore.simula(durata=50µs)

    # 4. Calcola focalizzazione
    risultato_focale = calcola_pressione_focale_completa(
        energia, riflettore, gap
    )

    # 5. Genera curva efficienza
    modello_eff = EfficiencyModel(parametro_decadimento=5e-6)
    impulsi, efficienza = genera_curva_efficienza(modello_eff)

    # 6. Simula controllo PID
    log_impulsi, log_gaps, log_interventi = simula_controllo_pid(
        gap_iniziale, n_impulsi=1000, I_target_kA=10.0
    )

    # 7. Genera grafici
    return (fig_riflettore, fig_scarica, fig_pressione,
            fig_efficienza, fig_motore, riepilogo)
```

### 5.3 Grafici Generati

1. **Geometria Riflettore**: Visualizza ellisse/parabola con fuochi e ray tracing
2. **Corrente di Scarica**: Forma d'onda I(t) e V(t)
3. **Profilo Pressione**: Distribuzione assiale e laterale
4. **Efficienza vs Impulsi**: Decadimento esponenziale con soglie
5. **Log Motore**: Evoluzione gap e interventi cumulativi

---

## 6. Validazione

### 6.1 Dati di Riferimento (Ogden et al. 2001)

Il modello è validato contro i dati sperimentali del dispositivo OssaTron:

```python
OGDEN_DATA = {
    14: {"p_max": 40.6, "efd": 0.105, "d_lat": 6.8, "l_ax": 44.1},
    20: {"p_max": 45.6, "efd": 0.255, "d_lat": 6.4, "l_ax": 59.0},
    28: {"p_max": 71.9, "efd": 0.370, "d_lat": 8.7, "l_ax": 67.6},
}
```

### 6.2 Criteri di Accettazione

- Errore pressione picco: < 20%
- Errore EFD: < 25%
- Errore zona focale: < 30%

---

## 7. Unità di Misura

Il progetto usa la libreria `pint` per gestire le unità fisiche:

```python
from src.core.units import ureg, Q_

# Esempi
tensione = Q_(20, "kV")
capacita = Q_(1, "uF")
pressione = Q_(50, "MPa")
energia = Q_(200, "J")

# Conversioni automatiche
energia_mJ = energia.to("mJ")  # 200000 mJ
```

---

## 8. Esecuzione

### Avvio Dashboard

```bash
cd "ESWT Digital Twin"
.venv/bin/python -m src.dashboard.app
# Apri http://127.0.0.1:8050
```

### Test

```bash
.venv/bin/python -m pytest tests/ -v
```

### Uso Programmatico

```python
from src.core.units import Q_
from src.modules.power_electronics import Capacitor, DischargeSimulator
from src.modules.physics_engine import EllipticalReflector, calcola_pressione_focale_completa
from src.modules.control import PIDGapController, MotorInterventionLogger

# Configura sistema
condensatore = Capacitor(Q_(1, "uF"), Q_(20, "kV"))
riflettore = EllipticalReflector(
    apertura=Q_(120, "mm"),
    distanza_fuochi=Q_(150, "mm")
)

# Simula
energia = condensatore.energia_immagazzinata
risultato = calcola_pressione_focale_completa(energia, riflettore, Q_(5, "mm"))

print(f"Pressione picco: {risultato.pressione_picco}")
print(f"EFD: {risultato.energy_flux_density}")
```

---

## Riferimenti

1. **Chen W. et al. (2010)** - "A Novel Pulsed Accelerator for PAED" - Modello ripartizione energia
2. **Ogden J.A. et al. (2001)** - "Principles of Shock Wave Therapy" - Dati validazione OssaTron
3. **Rompe & Weizel (1944)** - Modello resistenza arco elettrico
4. **Rayleigh (1917)** - Dinamica bolle cavitazione
