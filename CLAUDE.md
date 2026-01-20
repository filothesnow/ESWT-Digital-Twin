# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Twin for electrohydraulic ESWT (Extracorporeal Shock Wave Therapy) medical devices. Simulates the complete chain from capacitor discharge to focused shockwave on tissue target.

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run single test file
pytest tests/unit/test_power_electronics.py -v

# Run dashboard
python -m src.dashboard.app
# or
eswt-dashboard

# Lint
ruff check src/
black --check src/

# Type check
mypy src/
```

## Architecture

```
Power Electronics → Physics Engine → Degradation ↔ Control
        ↓                 ↓               ↓            ↓
    [discharge]     [shockwave]      [erosion]      [PID]
                          ↓
                    Dashboard (Dash/Plotly)
```

### Module Dependencies

1. **power_electronics**: RLC circuit discharge, plasma channel resistance (Rompe-Weizel model)
   - Input: voltage (kV), capacitance (µF)
   - Output: `DischargeResult` with current waveform, peak current, plasma energy

2. **physics_engine**: Shockwave propagation and focusing
   - `plasma_dynamics.py`: Rayleigh-Plesset bubble expansion
   - `shockwave.py`: Linear + nonlinear (Burgers) propagation
   - `reflector.py`: Ellipsoidal geometry, F1 (spark) → F2 (target)
   - `focusing.py`: Focal pressure, EFD (Energy Flux Density)
   - `cavitation.py`: Negative pressure threshold, bubble collapse

3. **degradation**: Long-term component wear
   - `electrode.py`: Erosion rate Δm = k·Q^α, gap evolution
   - `water_chemistry.py`: Debris accumulation, conductivity changes
   - `water_properties.py`: Surface tension, dielectric breakdown vs solutes

4. **control**: Gap maintenance
   - `pid_controller.py`: Adjusts gap based on current feedback (I < I_target → gap too large)
   - `gap_estimator.py`: Kalman filter (3-state: gap, gap_dot, erosion_rate)

### Units System

All physical quantities use `pint` for dimensional analysis:

```python
from src.core.units import ureg, Q_

tensione = Q_(20, "kV")
energia = Q_(200, "J")
pressione = Q_(50, "MPa")
```

### Validation Reference

Model validated against Ogden et al. (2001) OssaTron data:

| kV | P_max (MPa) | EFD (mJ/mm²) |
|----|-------------|--------------|
| 14 | 40.6        | 0.105        |
| 20 | 45.6        | 0.255        |
| 28 | 71.9        | 0.370        |

## Key Physical Equations

- **Discharge**: `L(dI/dt) + R(I)·I + (1/C)∫I dt = 0` (underdamped RLC)
- **Plasma resistance**: `R(I) = R₀·(I₀/I)^α`
- **Bubble expansion**: `R·R'' + (3/2)·R'² = (P_plasma - P_amb)/ρ`
- **Shockwave attenuation**: `P(r) = P₀·(r₀/r)·exp(-α·r)`
- **EFD**: `EFD = (1/ρc)·∫p²(t)dt`

## Documentation

- [docs/ARCHITETTURA_TECNICA.md](docs/ARCHITETTURA_TECNICA.md) - Complete technical documentation in Italian
