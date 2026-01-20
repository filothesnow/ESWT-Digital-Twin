# ESWT Digital Twin
# Digital Twin per dispositivo medico ESWT a generazione elettroidraulica
"""
Modulo principale del simulatore ESWT Digital Twin.

Questo pacchetto fornisce strumenti per simulare il comportamento di un
dispositivo medico a onde d'urto focalizzate (ESWT) con generazione elettroidraulica.

Moduli:
    - core: Costanti fisiche, unità di misura, proprietà materiali
    - modules.power_electronics: Simulazione scarica condensatore e plasma
    - modules.physics_engine: Propagazione onde d'urto e focalizzazione
    - modules.degradation: Erosione elettrodi e chimica dell'acqua
    - modules.control: Logica di controllo PID/Fuzzy
    - simulation: Motore di simulazione integrato
    - dashboard: Interfaccia grafica Dash
"""

__version__ = "0.1.0"
__author__ = "ESWT Team"
