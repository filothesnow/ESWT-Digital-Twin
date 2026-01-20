# Dashboard ESWT Digital Twin
"""
Applicazione Dash principale per il simulatore ESWT.

Permette di:
    - Selezionare forma del riflettore (ellisse/parabola)
    - Modificare dimensioni del riflettore
    - Selezionare tensione di scarica
    - Selezionare materiale dell'elettrodo
    - Lanciare simulazioni e visualizzare risultati
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Import moduli ESWT
from ..core.units import Q_
from ..core.constants import PhysicalConstants
from ..modules.power_electronics import (
    Capacitor,
    DischargeSimulator,
    crea_modello_costante,
    calcola_pressione_picco_chen,
    calcola_ripartizione_energia,
)
from ..modules.physics_engine import (
    ReflectorType,
    EllipticalReflector,
    ParabolicReflector,
    crea_riflettore,
    calcola_zona_focale,
    calcola_pressione_focale_completa,
    calcola_profilo_pressione_assiale,
    calcola_profilo_pressione_laterale,
)
from ..modules.degradation import (
    crea_stato_elettrodo,
    ElectrodeErosionModel,
    EfficiencyModel,
    calcola_efficienza,
    genera_curva_efficienza,
    simula_degradazione,
)
from ..modules.control import (
    PIDGapController,
    PIDParameters,
    MotorInterventionLogger,
)


# Stili CSS
CARD_STYLE = {
    "margin": "10px",
    "padding": "15px",
    "borderRadius": "10px",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
}

HEADER_STYLE = {
    "backgroundColor": "#2c3e50",
    "color": "white",
    "padding": "20px",
    "marginBottom": "20px",
    "borderRadius": "0 0 10px 10px",
}


def create_app():
    """Crea e configura l'applicazione Dash."""

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="ESWT Digital Twin",
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()
    register_callbacks(app)

    return app


def create_layout():
    """Crea il layout della dashboard."""

    return dbc.Container([
        # Header
        html.Div([
            html.H1("ESWT Digital Twin", className="text-center"),
            html.P(
                "Simulatore onde d'urto focalizzate a generazione elettroidraulica",
                className="text-center lead"
            ),
        ], style=HEADER_STYLE),

        # Riga principale: Controlli + Grafici
        dbc.Row([
            # Colonna sinistra: Controlli
            dbc.Col([
                # Card Riflettore
                dbc.Card([
                    dbc.CardHeader(html.H5("Geometria Riflettore")),
                    dbc.CardBody([
                        html.Label("Tipo di Riflettore"),
                        dcc.Dropdown(
                            id="dropdown-tipo-riflettore",
                            options=[
                                {"label": "Ellissoidale", "value": "ellittico"},
                                {"label": "Parabolico", "value": "parabolico"},
                            ],
                            value="ellittico",
                            clearable=False,
                        ),
                        html.Br(),

                        html.Label("Apertura (mm)"),
                        dcc.Slider(
                            id="slider-apertura",
                            min=50,
                            max=200,
                            step=10,
                            value=120,
                            marks={50: "50", 100: "100", 150: "150", 200: "200"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Distanza Fuochi / Parametro Focale (mm)"),
                        dcc.Slider(
                            id="slider-distanza-fuochi",
                            min=50,
                            max=300,
                            step=10,
                            value=150,
                            marks={50: "50", 150: "150", 300: "300"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ], style=CARD_STYLE),

                # Card Parametri Elettrici
                dbc.Card([
                    dbc.CardHeader(html.H5("Parametri Elettrici")),
                    dbc.CardBody([
                        html.Label("Tensione (kV)"),
                        dcc.Slider(
                            id="slider-tensione",
                            min=10,
                            max=30,
                            step=1,
                            value=20,
                            marks={10: "10", 15: "15", 20: "20", 25: "25", 30: "30"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Capacita (uF)"),
                        dcc.Slider(
                            id="slider-capacita",
                            min=0.1,
                            max=10,
                            step=0.1,
                            value=1,
                            marks={0.1: "0.1", 1: "1", 5: "5", 10: "10"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ], style=CARD_STYLE),

                # Card Elettrodo
                dbc.Card([
                    dbc.CardHeader(html.H5("Materiale Elettrodo")),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="dropdown-materiale",
                            options=[
                                {"label": "Tungsteno", "value": "tungsteno"},
                                {"label": "Rame", "value": "rame"},
                                {"label": "Acciaio Inox", "value": "acciaio_inox"},
                            ],
                            value="tungsteno",
                            clearable=False,
                        ),
                        html.Br(),

                        html.Label("Gap Elettrodi (mm)"),
                        dcc.Slider(
                            id="slider-gap",
                            min=2,
                            max=10,
                            step=0.5,
                            value=5,
                            marks={2: "2", 5: "5", 10: "10"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ], style=CARD_STYLE),

                # Pulsante Simulazione
                dbc.Card([
                    dbc.CardBody([
                        dbc.Button(
                            "Esegui Simulazione",
                            id="btn-simula",
                            color="primary",
                            size="lg",
                            className="w-100",
                        ),
                        html.Br(),
                        html.Br(),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[html.Div(id="output-status")]
                        ),
                    ]),
                ], style=CARD_STYLE),

            ], width=3),

            # Colonna destra: Grafici
            dbc.Col([
                dbc.Row([
                    # Grafico Geometria Riflettore
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Geometria Riflettore"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-riflettore", style={"height": "350px"}),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=6),

                    # Grafico Scarica
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Corrente di Scarica"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-scarica", style={"height": "350px"}),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=6),
                ]),

                dbc.Row([
                    # Grafico Pressione Focale
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Profilo Pressione"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-pressione", style={"height": "350px"}),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=6),

                    # Grafico Efficienza
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Efficienza vs Impulsi"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-efficienza", style={"height": "350px"}),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=6),
                ]),

                # Grafico Log Motore
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Log Interventi Motore e Gap"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-motore-log", style={"height": "350px"}),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=12),
                ]),

                # Riepilogo Risultati
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Riepilogo Simulazione"),
                            dbc.CardBody([
                                html.Div(id="output-riepilogo"),
                            ]),
                        ], style=CARD_STYLE),
                    ], width=12),
                ]),
            ], width=9),
        ]),

        # Footer
        html.Footer([
            html.Hr(),
            html.P(
                "ESWT Digital Twin - Simulatore basato su Chen et al. (2010) e Ogden et al. (2001)",
                className="text-center text-muted"
            ),
        ]),

        # Store per dati simulazione
        dcc.Store(id="store-risultati"),

    ], fluid=True)


def register_callbacks(app):
    """Registra i callback della dashboard."""

    @app.callback(
        [
            Output("graph-riflettore", "figure"),
            Output("graph-scarica", "figure"),
            Output("graph-pressione", "figure"),
            Output("graph-efficienza", "figure"),
            Output("graph-motore-log", "figure"),
            Output("output-riepilogo", "children"),
            Output("output-status", "children"),
            Output("store-risultati", "data"),
        ],
        [Input("btn-simula", "n_clicks")],
        [
            State("dropdown-tipo-riflettore", "value"),
            State("slider-apertura", "value"),
            State("slider-distanza-fuochi", "value"),
            State("slider-tensione", "value"),
            State("slider-capacita", "value"),
            State("dropdown-materiale", "value"),
            State("slider-gap", "value"),
        ],
        prevent_initial_call=False,
    )
    def esegui_simulazione(
        n_clicks,
        tipo_riflettore,
        apertura,
        distanza_fuochi,
        tensione_kV,
        capacita_uF,
        materiale,
        gap_mm,
    ):
        """Esegue la simulazione completa e aggiorna i grafici."""

        # Parametri
        tensione = Q_(tensione_kV, "kV")
        capacita = Q_(capacita_uF, "uF")
        gap = Q_(gap_mm, "mm")
        apertura_q = Q_(apertura, "mm")
        distanza_q = Q_(distanza_fuochi, "mm")

        # 1. Crea riflettore
        if tipo_riflettore == "ellittico":
            riflettore = EllipticalReflector(
                apertura=apertura_q,
                distanza_fuochi=distanza_q,
            )
        else:
            riflettore = ParabolicReflector(
                apertura=apertura_q,
                parametro_focale=Q_(distanza_fuochi / 2, "mm"),
            )
            riflettore.set_distanza_focale_effettiva(distanza_q)

        # 2. Calcola energia
        condensatore = Capacitor(capacita, tensione)
        energia = condensatore.energia_immagazzinata

        # 3. Simula scarica
        plasma = crea_modello_costante(Q_(0.5, "ohm"))
        simulatore = DischargeSimulator(condensatore, plasma, Q_(5, "uH"))
        risultato_scarica = simulatore.simula(durata=Q_(50, "us"))

        # 4. Calcola focalizzazione
        risultato_focale = calcola_pressione_focale_completa(
            energia, riflettore, gap
        )

        # 5. Calcola degradazione
        modello_eff = EfficiencyModel(parametro_decadimento=5e-6)
        impulsi_array, efficienza_array = genera_curva_efficienza(
            modello_eff, n_max=200000, n_punti=100
        )

        # 6. Simula controllo PID e log motore
        log_impulsi, log_gaps, log_interventi = simula_controllo_pid(
            gap_iniziale=gap_mm,
            n_impulsi=1000,
            I_target_kA=10.0,
        )

        # --- GRAFICI ---

        # Grafico 1: Geometria riflettore
        fig_riflettore = crea_grafico_riflettore(riflettore, tipo_riflettore)

        # Grafico 2: Corrente scarica
        fig_scarica = crea_grafico_scarica(risultato_scarica)

        # Grafico 3: Profilo pressione
        fig_pressione = crea_grafico_pressione(riflettore, risultato_focale)

        # Grafico 4: Efficienza
        fig_efficienza = crea_grafico_efficienza(impulsi_array, efficienza_array, materiale)

        # Grafico 5: Log Motore
        fig_motore = crea_grafico_motore_log(log_impulsi, log_gaps, log_interventi)

        # Riepilogo
        riepilogo = crea_riepilogo(
            energia, risultato_focale, riflettore, materiale, tensione_kV
        )

        status = dbc.Alert("Simulazione completata", color="success", duration=3000)

        # Dati per store
        dati = {
            "energia_J": energia.to("J").magnitude,
            "pressione_MPa": risultato_focale.pressione_picco.to("MPa").magnitude,
            "efd_mJ_mm2": risultato_focale.energy_flux_density.to("mJ/mm^2").magnitude,
        }

        return (
            fig_riflettore,
            fig_scarica,
            fig_pressione,
            fig_efficienza,
            fig_motore,
            riepilogo,
            status,
            dati,
        )


def crea_grafico_riflettore(riflettore, tipo):
    """Crea il grafico della geometria del riflettore."""

    fig = go.Figure()

    if tipo == "ellittico":
        # Disegna ellisse
        a = riflettore.a
        b = riflettore.b
        c = riflettore.c

        theta = np.linspace(0, 2 * np.pi, 100)
        z = a * np.cos(theta)
        r = b * np.sin(theta)

        # Mostra solo meta superiore (riflettore)
        mask = r >= 0
        fig.add_trace(go.Scatter(
            x=z[mask], y=r[mask],
            mode="lines",
            name="Riflettore",
            line=dict(color="blue", width=3),
        ))

        # Fuochi
        fig.add_trace(go.Scatter(
            x=[-c, c], y=[0, 0],
            mode="markers",
            name="Fuochi",
            marker=dict(size=12, color=["red", "green"], symbol="x"),
        ))

        # Raggi esempio
        for angle in [np.pi/6, np.pi/4, np.pi/3]:
            z_path, r_path = riflettore.calcola_percorso_raggio(angle)
            fig.add_trace(go.Scatter(
                x=z_path, y=r_path,
                mode="lines",
                line=dict(color="orange", width=1, dash="dash"),
                showlegend=False,
            ))

    else:  # Parabolico
        p = riflettore.p
        r_max = riflettore.apertura_mm / 2

        r = np.linspace(-r_max, r_max, 100)
        z = r**2 / (4 * p)

        fig.add_trace(go.Scatter(
            x=z, y=r,
            mode="lines",
            name="Riflettore",
            line=dict(color="blue", width=3),
        ))

        # Fuoco
        fig.add_trace(go.Scatter(
            x=[p], y=[0],
            mode="markers",
            name="Fuoco",
            marker=dict(size=12, color="red", symbol="x"),
        ))

    fig.update_layout(
        title=f"Riflettore {tipo.capitalize()}",
        xaxis_title="Z (mm)",
        yaxis_title="R (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def crea_grafico_scarica(risultato):
    """Crea il grafico della corrente di scarica."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Corrente", "Tensione"),
        vertical_spacing=0.15,
    )

    t_us = risultato.tempo * 1e6  # Converti a microsecondi

    # Corrente
    fig.add_trace(
        go.Scatter(
            x=t_us,
            y=risultato.corrente / 1000,  # kA
            mode="lines",
            name="Corrente",
            line=dict(color="blue"),
        ),
        row=1, col=1,
    )

    # Tensione
    fig.add_trace(
        go.Scatter(
            x=t_us,
            y=risultato.tensione_condensatore / 1000,  # kV
            mode="lines",
            name="Tensione",
            line=dict(color="red"),
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Tempo (us)", row=2, col=1)
    fig.update_yaxes(title_text="Corrente (kA)", row=1, col=1)
    fig.update_yaxes(title_text="Tensione (kV)", row=2, col=1)

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def crea_grafico_pressione(riflettore, risultato_focale):
    """Crea il grafico del profilo di pressione."""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Profilo Assiale", "Profilo Laterale"),
    )

    # Profilo assiale
    z, p_ax = calcola_profilo_pressione_assiale(
        riflettore, risultato_focale.pressione_picco
    )

    fig.add_trace(
        go.Scatter(
            x=z, y=p_ax,
            mode="lines",
            name="Assiale",
            line=dict(color="blue"),
        ),
        row=1, col=1,
    )

    # Profilo laterale
    r, p_lat = calcola_profilo_pressione_laterale(
        riflettore, risultato_focale.pressione_picco
    )

    fig.add_trace(
        go.Scatter(
            x=r, y=p_lat,
            mode="lines",
            name="Laterale",
            line=dict(color="green"),
        ),
        row=1, col=2,
    )

    # Linea -6dB
    p_max = risultato_focale.pressione_picco.to("MPa").magnitude
    p_6dB = p_max / 2

    fig.add_hline(y=p_6dB, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=p_6dB, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_xaxes(title_text="Z (mm)", row=1, col=1)
    fig.update_xaxes(title_text="R (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Pressione (MPa)", row=1, col=1)
    fig.update_yaxes(title_text="Pressione (MPa)", row=1, col=2)

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def crea_grafico_efficienza(impulsi, efficienza, materiale):
    """Crea il grafico efficienza vs impulsi."""

    # Colori per materiale
    colori = {
        "tungsteno": "blue",
        "rame": "orange",
        "acciaio_inox": "gray",
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=impulsi / 1000,  # kilo-impulsi
        y=efficienza * 100,  # percentuale
        mode="lines",
        name=materiale.replace("_", " ").title(),
        line=dict(color=colori.get(materiale, "blue"), width=2),
        fill="tozeroy",
        fillcolor=f"rgba(0, 100, 200, 0.1)",
    ))

    # Soglie
    fig.add_hline(y=70, line_dash="dash", line_color="orange",
                  annotation_text="Soglia attenzione (70%)")
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Soglia critica (50%)")

    fig.update_layout(
        xaxis_title="Numero Impulsi (x1000)",
        yaxis_title="Efficienza (%)",
        yaxis=dict(range=[0, 105]),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def crea_riepilogo(energia, risultato_focale, riflettore, materiale, tensione_kV):
    """Crea il riepilogo testuale dei risultati."""

    zona = risultato_focale.zona_focale

    return dbc.Row([
        dbc.Col([
            html.H6("Energia"),
            html.P(f"{energia.to('J').magnitude:.1f} J", className="h4 text-primary"),
        ], width=2),

        dbc.Col([
            html.H6("Pressione Picco"),
            html.P(f"{risultato_focale.pressione_picco.to('MPa').magnitude:.1f} MPa",
                   className="h4 text-success"),
        ], width=2),

        dbc.Col([
            html.H6("EFD"),
            html.P(f"{risultato_focale.energy_flux_density.to('mJ/mm^2').magnitude:.3f} mJ/mm²",
                   className="h4 text-info"),
        ], width=2),

        dbc.Col([
            html.H6("Zona Focale Laterale"),
            html.P(f"{zona.diametro_laterale_6dB.to('mm').magnitude:.1f} mm",
                   className="h4"),
        ], width=2),

        dbc.Col([
            html.H6("Zona Focale Assiale"),
            html.P(f"{zona.lunghezza_assiale_6dB.to('mm').magnitude:.1f} mm",
                   className="h4"),
        ], width=2),

        dbc.Col([
            html.H6("Materiale"),
            html.P(materiale.replace("_", " ").title(), className="h4"),
        ], width=2),
    ])


def simula_controllo_pid(gap_iniziale, n_impulsi, I_target_kA):
    """
    Simula il controllo PID del gap per generare dati per il grafico.

    Parametri:
        gap_iniziale: Gap iniziale in mm
        n_impulsi: Numero di impulsi da simulare
        I_target_kA: Corrente target in kA

    Ritorna:
        Tuple (impulsi, gaps, interventi_cumulativi)
    """
    # Parametri PID
    params = PIDParameters(
        Kp=0.5,
        Ki=0.01,
        Kd=0.1,
        I_target=Q_(I_target_kA, "kA"),
        gap_min=Q_(3, "mm"),
        gap_max=Q_(15, "mm"),
    )

    controller = PIDGapController(params)
    logger = MotorInterventionLogger()

    # Simula
    gap_current = Q_(gap_iniziale, "mm")
    erosione_per_impulso = 0.0001  # mm per impulso

    impulsi = []
    gaps = []
    interventi_cumulativi = []
    n_interventi = 0

    for i in range(n_impulsi):
        # Simula feedback con rumore
        # Corrente varia inversamente col gap
        I_nominal = I_target_kA  # kA
        I_variation = (gap_iniziale - gap_current.to("mm").magnitude) * 0.5
        I_noise = np.random.normal(0, 0.2)
        I_feedback = Q_(I_nominal + I_variation + I_noise, "kA")

        # Step PID
        new_gap, action = controller.step(I_feedback, gap_current, i + 1)

        # Verifica se c'e stato un intervento significativo
        if abs(action.delta_gap.to("mm").magnitude) > 0.001:
            logger.log_intervention(action)
            n_interventi += 1

        # Applica erosione
        gap_current = Q_(new_gap.to("mm").magnitude + erosione_per_impulso, "mm")

        # Campiona ogni 10 impulsi
        if (i + 1) % 10 == 0:
            impulsi.append(i + 1)
            gaps.append(gap_current.to("mm").magnitude)
            interventi_cumulativi.append(n_interventi)

    return np.array(impulsi), np.array(gaps), np.array(interventi_cumulativi)


def crea_grafico_motore_log(impulsi, gaps, interventi):
    """Crea il grafico del log motore con gap e interventi."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Evoluzione Gap", "Interventi Motore Cumulativi"),
        vertical_spacing=0.15,
    )

    # Grafico Gap
    fig.add_trace(
        go.Scatter(
            x=impulsi,
            y=gaps,
            mode="lines",
            name="Gap",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 100, 200, 0.1)",
        ),
        row=1, col=1,
    )

    # Linee soglia gap
    fig.add_hline(y=3, line_dash="dash", line_color="red",
                  annotation_text="Gap Min", row=1, col=1)
    fig.add_hline(y=15, line_dash="dash", line_color="red",
                  annotation_text="Gap Max", row=1, col=1)

    # Grafico Interventi
    fig.add_trace(
        go.Scatter(
            x=impulsi,
            y=interventi,
            mode="lines",
            name="Interventi",
            line=dict(color="orange", width=2),
        ),
        row=2, col=1,
    )

    # Aggiungi marcatori per ogni 100 interventi
    milestone_mask = np.mod(interventi, 100) == 0
    milestones_x = impulsi[milestone_mask]
    milestones_y = interventi[milestone_mask]

    if len(milestones_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=milestones_x,
                y=milestones_y,
                mode="markers",
                name="Milestone",
                marker=dict(size=8, color="red", symbol="diamond"),
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Numero Impulsi", row=2, col=1)
    fig.update_yaxes(title_text="Gap (mm)", row=1, col=1)
    fig.update_yaxes(title_text="N. Interventi", row=2, col=1)

    fig.update_layout(
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def run_dashboard(host="127.0.0.1", port=8050, debug=True):
    """Avvia la dashboard."""
    app = create_app()
    print(f"\n{'='*50}")
    print("ESWT Digital Twin Dashboard")
    print(f"{'='*50}")
    print(f"Apri il browser a: http://{host}:{port}")
    print(f"{'='*50}\n")
    app.run(host=host, port=port, debug=debug)


# Entry point per esecuzione diretta
if __name__ == "__main__":
    run_dashboard()
