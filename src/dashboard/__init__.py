# Modulo Dashboard
"""
Dashboard interattiva per ESWT Digital Twin.

Interfaccia grafica basata su Plotly Dash per visualizzare
i risultati delle simulazioni in tempo reale.

Funzionalità:
    - Selezione forma riflettore (ellisse/parabola)
    - Modifica dimensioni riflettore
    - Selezione tensione di scarica
    - Selezione materiale elettrodo
    - Visualizzazione risultati simulazione

Esecuzione:
    python -m src.dashboard.app

Moduli:
    - app: Applicazione Dash principale
    - layouts: Layout componenti UI
    - callbacks: Callback interattività
"""

from .app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
