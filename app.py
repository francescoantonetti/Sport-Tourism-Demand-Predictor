from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(
    page_title="Sport Tourism Demand Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#17374E"
NAVY_2 = "#20384F"
ORANGE = "#F7931E"
BG = "#F5F7FB"
CARD = "#FFFFFF"
BORDER = "#D7E0EA"
PALE_BLUE = "#DDE9FF"
TEXT = "#20384F"
MUTED = "#6B7C8F"
GREEN = "#2E8B57"
RED = "#C94C4C"
GRID = "#E7ECF2"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: light !important; }}
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        color-scheme: light !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {NAVY} !important;
        border-right: 3px solid {ORANGE};
    }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.20) !important; }}
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label p {{ color: #FFFFFF !important; }}

    .block-container {{
        padding-top: 4rem !important;
        padding-bottom: 2rem;
        max-width: 1450px;
    }}

    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 4rem !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT} !important; letter-spacing: -0.02em; }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{ color: {TEXT}; }}

    /* KPI cards: force labels and values to remain visible */
    div[data-testid="stMetric"] {{
        background: {CARD} !important;
        border: 1px solid {BORDER};
        border-top: 3px solid {ORANGE};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(23,55,78,.05);
    }}
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {{
        color: {MUTED} !important;
        -webkit-text-fill-color: {MUTED} !important;
        font-weight: 650 !important;
        opacity: 1 !important;
    }}
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }}

    .eyebrow {{
    color: #355D7A !important;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .18em;
    margin-top: 0.5rem !important;
    margin-bottom: 0.3rem !important;
    }}

    .info-box {{
        background: {PALE_BLUE} !important;
        border-radius: 10px;
        padding: 14px 16px;
        color: {TEXT} !important;
        border: 1px solid #CCDDF8;
        margin: 10px 0 18px 0;
    }}
    .info-box, .info-box * {{ color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important; }}

    .section-card {{
        background: {CARD} !important;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(23,55,78,.04);
        color: {TEXT} !important;
    }}
    .section-card, .section-card * {{ color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important; }}
    .section-card .accent {{ color: {ORANGE} !important; -webkit-text-fill-color: {ORANGE} !important; font-weight: 800; }}
    .section-card .small-note {{ color: {MUTED} !important; -webkit-text-fill-color: {MUTED} !important; font-size: .90rem; }}
    .accent {{ color: {ORANGE} !important; font-weight: 800; }}

    /* Tabs: BaseWeb puts the visible text in nested p/span elements */
    .stTabs [data-baseweb="tab-list"] {{ gap: 20px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ color: {TEXT} !important; font-weight: 650; opacity: 1 !important; }}
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        opacity: 1 !important;
    }}
    .stTabs [aria-selected="true"] {{ border-bottom: 3px solid {ORANGE} !important; }}
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {{
        color: #FF4B4B !important;
        -webkit-text-fill-color: #FF4B4B !important;
    }}

    /* Form controls */
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label p,
    [data-testid="stSegmentedControl"] label,
    [data-testid="stSegmentedControl"] label p {{ color: {TEXT} !important; }}

    /* Dataframes / buttons / download controls */
    [data-testid="stDownloadButton"] button {{
        border-color: {ORANGE} !important;
        color: {TEXT} !important;
        background: {CARD} !important;
    }}
    [data-testid="stDownloadButton"] button p {{ color: {TEXT} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
LOGO_PATH = Path(__file__).parent / "logo_sbl.png"
HERO_LOGO_PATH = Path(__file__).parent / "logo_sbl_hero.png"

@st.cache_data
def load_data():
    perf = pd.read_csv(DATA_DIR / "final_model_performance_v5.csv")
    intervals = pd.read_csv(DATA_DIR / "final_interval_summary_v5.csv")
    kpis = pd.read_csv(DATA_DIR / "final_project_kpis_v5.csv")
    events = pd.read_csv(DATA_DIR / "final_supported_individual_events_v5.csv")
    scenario = pd.read_csv(DATA_DIR / "final_supported_scenario_summary_v5.csv")
    geo = pd.read_csv(DATA_DIR / "kepler_sport_tourism_results_v5.csv")
    return perf, intervals, kpis, events, scenario, geo

perf, intervals, kpis, events, scenario, geo = load_data()

TEXTS = {
    "EN": {
        "language": "Language / Lingua",
        "side_desc": "Predictive tourism analytics for Torino, Milano, Roma, Napoli and Bologna.",
        "scope": "Project scope",
        "cities": "Cities",
        "targets": "Targets",
        "arrivals": "Arrivals",
        "presences": "Presences",
        "final_models": "Final models",
        "forecast_test": "Forecast test",
        "validation": "Key validation",
        "mean_mape": "Mean final MAPE",
        "pi_coverage": "95% PI coverage",
        "improved_models": "Improved models",
        "side_note": "Event effects are predictive associations, not causal estimates. Davis Cup evidence is supported only for Bologna presences.",
        "hero_sub": "Forecast accuracy, city comparison and supported sports-event impact across five Italian destinations.",
        "tabs": ["Overview", "Forecast Performance", "City Explorer", "Geographic Map", "Event Impact", "Model & Data Quality"],
        "overview_title": "Project Overview",
        "overview_info": "The dashboard compares the final out-of-sample forecasting performance of 10 city-target models and separates validated forecasting gains from event-effect evidence.",
        "performance_city": "Forecasting performance by city",
        "final_mape_city": "Final MAPE by city and target",
        "improvement_base": "Relative improvement vs base model",
        "improvement": "Improvement (%)",
        "key_insights": "Key insights",
        "best_gain": "Best relative forecasting gain",
        "gain_text": "improvement vs base model.",
        "lowest_city": "Lowest city-average final MAPE",
        "across_targets": "across arrivals and presences.",
        "event_evidence": "Supported event-effect evidence",
        "forecast_perf": "Forecast Performance",
        "target": "Target",
        "all": "All",
        "no_change": "No change",
        "base_vs_final": "Base vs final MAPE",
        "base_mape": "Base MAPE (%)",
        "final_mape": "Final MAPE (%)",
        "model_table": "Model comparison table",
        "city_explorer": "City Explorer",
        "select_city": "Select city",
        "arrivals_mape": "Arrivals final MAPE",
        "presences_mape": "Presences final MAPE",
        "mean_improvement": "Mean improvement",
        "model_spec": "Final model specification",
        "no_event": "No statistically supported event-specific effect in the current clean sample.",
        "map_title": "Geographic Performance Map",
        "map_info": "Point size represents city-average final MAPE. Orange points indicate at least one city-target model improved versus the base specification. Bologna is additionally the only city with a statistically supported event-specific effect.",
        "download": "Download Kepler-ready CSV",
        "event_title": "Supported Event Impact",
        "event_warning": "Only Davis Cup presences in Bologna are statistically supported in the current clean sample. Other event-specific outputs should not be presented as validated impacts.",
        "supported_city": "Supported city",
        "canonical_uplift": "Canonical uplift",
        "mean_event_pres": "Mean supported event presences",
        "max_monthly_impact": "Max supported monthly impact",
        "historical_chart": "Supported Davis Cup historical attribution — Bologna",
        "event_year": "Event year",
        "additional_pres": "Additional presences",
        "historical_events": "Historical supported events",
        "scenario_engine": "Supported scenario engine",
        "best_month": "Best supported month",
        "additional_int_pres": "Additional international presences",
        "direct_impact": "Direct international tourism impact",
        "scenario_note": "Scenario-engine outputs are hypothetical supported monthly placements in Bologna and are not interchangeable with historical event attribution.",
        "quality_title": "Model & Data Quality",
        "forecast_rows": "Forecast rows with 95% PI",
        "models_pi": "Models with 95% PI",
        "reproduction": "Reproduction MATCH",
        "close_review": "CLOSE REVIEW",
        "interval_status": "Interval reproduction status",
        "models": "Models",
        "pi_validation": "Prediction-interval validation",
        "limits": "Interpretation limits",
        "limit_items": [
            "Sports uplift is a predictive association, not causal identification.",
            "The event-effect clean sample covers 2022–2024 and only five cities.",
            "Davis Cup support is based on three treated city-months, all in Bologna.",
            "Davis Cup effects must not be transported to Torino, Milano, Roma or Napoli.",
            "Direct economic impact includes international tourism spending only; indirect and induced effects are not included.",
        ],
        "footer": "Sport Tourism Demand Predictor · Final V5 dashboard · Python / Streamlit",
    },
    "IT": {
        "language": "Lingua / Language",
        "side_desc": "Analisi predittiva dei flussi turistici per Torino, Milano, Roma, Napoli e Bologna.",
        "scope": "Perimetro del progetto",
        "cities": "Città",
        "targets": "Target",
        "arrivals": "Arrivi",
        "presences": "Presenze",
        "final_models": "Modelli finali",
        "forecast_test": "Test di previsione",
        "validation": "Validazione principale",
        "mean_mape": "MAPE finale medio",
        "pi_coverage": "Copertura PI al 95%",
        "improved_models": "Modelli migliorati",
        "side_note": "Gli effetti degli eventi sono associazioni predittive, non stime causali. L'evidenza sulla Davis Cup è supportata solo per le presenze a Bologna.",
        "hero_sub": "Accuratezza delle previsioni, confronto tra città e impatto supportato degli eventi sportivi in cinque destinazioni italiane.",
        "tabs": ["Panoramica", "Performance previsiva", "Analisi per città", "Mappa geografica", "Impatto eventi", "Qualità modello e dati"],
        "overview_title": "Panoramica del progetto",
        "overview_info": "La dashboard confronta la performance previsiva out-of-sample dei 10 modelli città-target e distingue i miglioramenti previsivi validati dall'evidenza relativa agli effetti degli eventi.",
        "performance_city": "Performance previsiva per città",
        "final_mape_city": "MAPE finale per città e target",
        "improvement_base": "Miglioramento relativo rispetto al modello base",
        "improvement": "Miglioramento (%)",
        "key_insights": "Risultati principali",
        "best_gain": "Miglior miglioramento relativo della previsione",
        "gain_text": "di miglioramento rispetto al modello base.",
        "lowest_city": "MAPE finale medio più basso per città",
        "across_targets": "considerando arrivi e presenze.",
        "event_evidence": "Evidenza supportata dell'effetto evento",
        "forecast_perf": "Performance previsiva",
        "target": "Target",
        "all": "Tutti",
        "no_change": "Nessuna variazione",
        "base_vs_final": "MAPE base vs finale",
        "base_mape": "MAPE base (%)",
        "final_mape": "MAPE finale (%)",
        "model_table": "Tabella di confronto dei modelli",
        "city_explorer": "Analisi per città",
        "select_city": "Seleziona città",
        "arrivals_mape": "MAPE finale arrivi",
        "presences_mape": "MAPE finale presenze",
        "mean_improvement": "Miglioramento medio",
        "model_spec": "Specifiche del modello finale",
        "no_event": "Nessun effetto specifico degli eventi risulta statisticamente supportato nel campione pulito attuale.",
        "map_title": "Mappa geografica delle performance",
        "map_info": "La dimensione dei punti rappresenta il MAPE finale medio della città. I punti arancioni indicano che almeno un modello città-target è migliorato rispetto alla specifica base. Bologna è inoltre l'unica città con un effetto specifico degli eventi statisticamente supportato.",
        "download": "Scarica CSV pronto per Kepler",
        "event_title": "Impatto supportato degli eventi",
        "event_warning": "Solo le presenze associate alla Davis Cup a Bologna sono statisticamente supportate nel campione pulito attuale. Gli altri output specifici per evento non devono essere presentati come impatti validati.",
        "supported_city": "Città supportata",
        "canonical_uplift": "Incremento canonico",
        "mean_event_pres": "Presenze medie supportate per evento",
        "max_monthly_impact": "Impatto mensile massimo supportato",
        "historical_chart": "Attribuzione storica supportata della Davis Cup — Bologna",
        "event_year": "Anno dell'evento",
        "additional_pres": "Presenze aggiuntive",
        "historical_events": "Eventi storici supportati",
        "scenario_engine": "Motore degli scenari supportati",
        "best_month": "Mese migliore supportato",
        "additional_int_pres": "Presenze internazionali aggiuntive",
        "direct_impact": "Impatto turistico internazionale diretto",
        "scenario_note": "Gli output del motore di scenario rappresentano collocazioni mensili ipotetiche supportate a Bologna e non sono intercambiabili con l'attribuzione storica degli eventi.",
        "quality_title": "Qualità del modello e dei dati",
        "forecast_rows": "Righe previsive con PI al 95%",
        "models_pi": "Modelli con PI al 95%",
        "reproduction": "Riproduzione MATCH",
        "close_review": "CLOSE REVIEW",
        "interval_status": "Stato di riproduzione degli intervalli",
        "models": "Modelli",
        "pi_validation": "Validazione degli intervalli di previsione",
        "limits": "Limiti interpretativi",
        "limit_items": [
            "L'incremento associato allo sport è un'associazione predittiva, non un'identificazione causale.",
            "Il campione pulito per l'effetto degli eventi copre il periodo 2022–2024 e solo cinque città.",
            "Il supporto relativo alla Davis Cup si basa su tre city-month trattati, tutti a Bologna.",
            "Gli effetti della Davis Cup non devono essere trasferiti a Torino, Milano, Roma o Napoli.",
            "L'impatto economico diretto include solo la spesa turistica internazionale; gli effetti indiretti e indotti non sono inclusi.",
        ],
        "footer": "Sport Tourism Demand Predictor · Dashboard finale V5 · Python / Streamlit",
    },
}

def kpi_value(name, default=np.nan):
    row = kpis.loc[kpis["kpi"] == name, "value"]
    return float(row.iloc[0]) if len(row) else default


def style_plot(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Arial", size=13),
        title=dict(font=dict(color=TEXT, family="Arial", size=17)),
        margin=dict(l=20, r=20, t=60, b=30),
        legend=(
            dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(color=TEXT, family="Arial", size=12),
                bgcolor="rgba(255,255,255,0.0)",
            ) if legend else None
        ),
        hoverlabel=dict(bgcolor="white", font_color=TEXT),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
    )
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
    )
    return fig


def translated_target(v, lang):
    if v == "arrivals":
        return TEXTS[lang]["arrivals"]
    if v == "presences":
        return TEXTS[lang]["presences"]
    return v

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=190)

    language_label = "Language / Lingua"
    language_choice = st.radio(
        language_label,
        ["English", "Italiano"],
        horizontal=True,
        label_visibility="visible",
    )
    lang = "IT" if language_choice == "Italiano" else "EN"
    T = TEXTS[lang]

    st.markdown("## Sport Tourism")
    st.markdown("### Demand Predictor")
    st.caption("PYTHON · FORECASTING · EVENT ANALYTICS")
    st.markdown(T["side_desc"])
    st.markdown("---")
    st.markdown(f"### {T['scope']}")
    st.markdown(f"**{T['cities']}:** 5")
    st.markdown(f"**{T['targets']}:** {T['arrivals']} + {T['presences']}")
    st.markdown(f"**{T['final_models']}:** 10")
    st.markdown(f"**{T['forecast_test']}:** 2024")
    st.markdown("---")
    st.markdown(f"### {T['validation']}")
    st.markdown(f"**{T['mean_mape']}:** {kpi_value('mean_final_MAPE'):.2f}%")
    st.markdown(f"**{T['pi_coverage']}:** {kpi_value('mean_empirical_2024_PI_coverage'):.0f}%")
    st.markdown(f"**{T['improved_models']}:** {int(kpi_value('models_improved_vs_base'))}/10")
    st.markdown("---")
    st.caption(T["side_note"])

col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
with col_logo:
    if HERO_LOGO_PATH.exists():
        st.image(str(HERO_LOGO_PATH), width=118)
with col_title:
    st.markdown(
        f"""
        <div class="eyebrow">SBL CONSULTANCY / SPORT TOURISM ANALYTICS</div>
        <div class="hero-title">Sport Tourism Demand Predictor</div>
        <div class="hero-sub">{T['hero_sub']}</div>
        """,
        unsafe_allow_html=True,
    )

tab_overview, tab_perf, tab_city, tab_map, tab_event, tab_quality = st.tabs(T["tabs"])

with tab_overview:
    st.markdown(f"## {T['overview_title']}")
    st.markdown(f'<div class="info-box">{T["overview_info"]}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["cities"], f"{int(kpi_value('cities'))}")
    c2.metric(T["final_models"], f"{int(kpi_value('forecast_models'))}")
    c3.metric(T["mean_mape"], f"{kpi_value('mean_final_MAPE'):.2f}%")
    c4.metric("2024 " + T["pi_coverage"], f"{kpi_value('mean_empirical_2024_PI_coverage'):.0f}%")

    st.markdown(f"### {T['performance_city']}")
    city_perf = perf.pivot(index="city", columns="target", values="final_MAPE").reset_index()
    order = geo.sort_values("mean_final_MAPE")["city"].tolist()
    city_perf["city"] = pd.Categorical(city_perf["city"], categories=order, ordered=True)
    city_perf = city_perf.sort_values("city")

    fig = go.Figure()
    fig.add_bar(name=T["arrivals"], x=city_perf["city"], y=city_perf["arrivals"], marker_color=NAVY_2)
    fig.add_bar(name=T["presences"], x=city_perf["city"], y=city_perf["presences"], marker_color=ORANGE)
    fig.update_layout(barmode="group", title=T["final_mape_city"])
    fig.update_yaxes(title="MAPE (%)")
    st.plotly_chart(style_plot(fig), width="stretch")

    left, right = st.columns([1.2, 1])
    with left:
        improvement = perf.copy()
        improvement["series"] = improvement.apply(
            lambda r: f"{r['city']} · {translated_target(r['target'], lang)}", axis=1
        )
        improvement = improvement.sort_values("improvement_pct", ascending=True)
        fig2 = px.bar(
            improvement,
            x="improvement_pct",
            y="series",
            orientation="h",
            title=T["improvement_base"],
            labels={"improvement_pct": T["improvement"], "series": ""},
        )
        fig2.update_traces(marker_color=ORANGE)
        st.plotly_chart(style_plot(fig2, height=430, legend=False), width="stretch")

    with right:
        best = perf.loc[perf["improvement_pct"].idxmax()]
        napoli = geo.loc[geo["city"] == "Napoli"].iloc[0]
        st.markdown(f"### {T['key_insights']}")
        st.markdown(
            f"""
            <div class="section-card">
            <b>{T['best_gain']}</b><br>
            <span class="accent">{best['city']} · {translated_target(best['target'], lang)}</span><br>
            {best['improvement_pct']:.2f}% {T['gain_text']}
            <br><br>
            <b>{T['lowest_city']}</b><br>
            <span class="accent">Napoli</span><br>
            {napoli['mean_final_MAPE']:.2f}% {T['across_targets']}
            <br><br>
            <b>{T['event_evidence']}</b><br>
            Bologna · Davis Cup {T['presences'].lower()}: <span class="accent">+12.26%</span>.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_perf:
    st.markdown(f"## {T['forecast_perf']}")
    option_labels = [T["all"], T["arrivals"], T["presences"]]
    selected_label = st.segmented_control(T["target"], options=option_labels, default=T["all"])
    reverse_target = {T["arrivals"]: "arrivals", T["presences"]: "presences"}
    pf = perf.copy() if selected_label == T["all"] else perf[perf["target"] == reverse_target[selected_label]].copy()

    fig = go.Figure()
    for target, color in [("arrivals", NAVY_2), ("presences", ORANGE)]:
        d = pf[pf["target"] == target]
        if not d.empty:
            fig.add_trace(go.Scatter(
                x=d["base_MAPE"], y=d["final_MAPE"], mode="markers+text",
                text=d["city"], textposition="top center", name=translated_target(target, lang),
                textfont=dict(color=TEXT),
                marker=dict(size=14, color=color, line=dict(width=1, color="white")),
                customdata=np.stack([d["improvement_pct"]], axis=-1),
                hovertemplate=(
                    "%{text}<br>Base MAPE: %{x:.2f}%<br>Final MAPE: %{y:.2f}%<br>"
                    + T["improvement"] + ": %{customdata[0]:.2f}%<extra></extra>"
                )
            ))
    maxv = max(perf["base_MAPE"].max(), perf["final_MAPE"].max()) + .4
    fig.add_trace(go.Scatter(
        x=[0, maxv], y=[0, maxv], mode="lines",
        line=dict(color="#9AA8B5", dash="dash"), name=T["no_change"]
    ))
    fig.update_layout(title=T["base_vs_final"], xaxis_title=T["base_mape"], yaxis_title=T["final_mape"])
    st.plotly_chart(style_plot(fig, 500), width="stretch")

    st.markdown(f"### {T['model_table']}")
    table = pf[["city", "target", "sports_exog_retained", "base_MAPE", "final_MAPE", "improvement_pp", "improvement_pct", "improved"]].copy()
    table["target"] = table["target"].map(lambda x: translated_target(x, lang))
    if lang == "IT":
        table.columns = ["Città", "Target", "Esogene sportive mantenute", "MAPE base", "MAPE finale", "Miglioramento (pp)", "Miglioramento (%)", "Migliorato"]
        fmt = {"MAPE base": "{:.3f}%", "MAPE finale": "{:.3f}%", "Miglioramento (pp)": "{:.3f}", "Miglioramento (%)": "{:.2f}%"}
    else:
        table.columns = ["City", "Target", "Sports exog retained", "Base MAPE", "Final MAPE", "Improvement (pp)", "Improvement (%)", "Improved"]
        fmt = {"Base MAPE": "{:.3f}%", "Final MAPE": "{:.3f}%", "Improvement (pp)": "{:.3f}", "Improvement (%)": "{:.2f}%"}
    st.dataframe(table.style.format(fmt), width="stretch", hide_index=True)

with tab_city:
    st.markdown(f"## {T['city_explorer']}")
    city = st.selectbox(T["select_city"], geo["city"].tolist())
    g = geo.loc[geo["city"] == city].iloc[0]
    p = perf[perf["city"] == city]
    ints = intervals[intervals["city"] == city]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["mean_mape"], f"{g['mean_final_MAPE']:.2f}%")
    c2.metric(T["arrivals_mape"], f"{g['arrivals_final_MAPE']:.2f}%")
    c3.metric(T["presences_mape"], f"{g['presences_final_MAPE']:.2f}%")
    c4.metric(T["mean_improvement"], f"{g['mean_improvement_pct']:.2f}%")

    left, right = st.columns([1.15, 1])
    with left:
        comp = p.melt(id_vars=["target"], value_vars=["base_MAPE", "final_MAPE"], var_name="model", value_name="MAPE")
        comp["model"] = comp["model"].map({"base_MAPE": "Base", "final_MAPE": "Finale" if lang == "IT" else "Final"})
        comp["target_display"] = comp["target"].map(lambda x: translated_target(x, lang))
        final_label = "Finale" if lang == "IT" else "Final"
        fig = px.bar(
            comp, x="target_display", y="MAPE", color="model", barmode="group",
            title=f"{city}: {T['base_vs_final'].lower()}",
            labels={"target_display": T["target"], "model": "Modello" if lang == "IT" else "Model"},
            color_discrete_map={"Base": "#9FB0C0", final_label: ORANGE},
        )
        fig.update_yaxes(title="MAPE (%)")
        st.plotly_chart(style_plot(fig, 390), width="stretch")

    with right:
        st.markdown(f"### {T['model_spec']}")
        display_cols = ["target", "model_family", "order", "seasonal_order", "exogenous_variables", "reproduction_status", "empirical_2024_coverage_pct"]
        model_table = ints[display_cols].copy()
        model_table["target"] = model_table["target"].map(lambda x: translated_target(x, lang))
        if lang == "IT":
            model_table.columns = ["Target", "Famiglia modello", "Ordine", "Ordine stagionale", "Variabili esogene", "Stato riproduzione", "Copertura 2024 (%)"]
        st.dataframe(model_table, width="stretch", hide_index=True)
        if bool(g["supported_event_effect"]):
            supported_note = (
                "Bologna: effetto statisticamente supportato della Davis Cup sulle presenze (+12,26%)."
                if lang == "IT" else g["supported_event_note"]
            )
            st.success(supported_note)
        else:
            st.info(T["no_event"])

with tab_map:
    st.markdown(f"## {T['map_title']}")
    st.markdown(f'<div class="info-box">{T["map_info"]}</div>', unsafe_allow_html=True)

    map_df = geo.copy()
    map_df["radius"] = 22000 + map_df["mean_final_MAPE"] * 6500
    map_df["color"] = map_df["any_model_improved"].map({True: [247,147,30,190], False: [32,56,79,185]})

    layer = pdk.Layer(
        "ScatterplotLayer", data=map_df,
        get_position="[longitude, latitude]",
        get_radius="radius", get_fill_color="color",
        pickable=True, stroked=True,
        get_line_color=[255,255,255], line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=42.8, longitude=11.2, zoom=4.7, pitch=0)
    if lang == "IT":
        tooltip_html = (
            "<b>{city}</b><br/>MAPE finale medio: {mean_final_MAPE}%<br/>"
            "Arrivi: {arrivals_final_MAPE}%<br/>Presenze: {presences_final_MAPE}%<br/>"
            "Miglioramento medio: {mean_improvement_pct}%<br/>{supported_event_note}"
        )
    else:
        tooltip_html = (
            "<b>{city}</b><br/>Mean final MAPE: {mean_final_MAPE}%<br/>"
            "Arrivals: {arrivals_final_MAPE}%<br/>Presences: {presences_final_MAPE}%<br/>"
            "Mean improvement: {mean_improvement_pct}%<br/>{supported_event_note}"
        )
    tooltip = {"html": tooltip_html, "style": {"backgroundColor": NAVY, "color": "white"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light"), width="stretch")

    st.download_button(
        T["download"], data=geo.to_csv(index=False).encode("utf-8"),
        file_name="kepler_sport_tourism_results_v5.csv", mime="text/csv",
    )

with tab_event:
    st.markdown(f"## {T['event_title']}")
    st.warning(T["event_warning"])

    sc = scenario.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["supported_city"], sc["supported_city"])
    c2.metric(T["canonical_uplift"], "+12.26%")
    c3.metric(T["mean_event_pres"], f"{events['presences_supported_additional'].mean():,.0f}")
    c4.metric(T["max_monthly_impact"], f"€{sc['max_direct_international_tourism_impact_eur']/1e6:.2f}M")

    ev = events.copy()
    ev["year"] = ev["year"].astype(str)
    fig = px.bar(
        ev, x="year", y="presences_supported_additional", title=T["historical_chart"],
        labels={"year": T["event_year"], "presences_supported_additional": T["additional_pres"]},
    )
    fig.update_traces(marker_color=ORANGE, texttemplate="%{y:,.0f}", textposition="outside", textfont=dict(color=TEXT))
    st.plotly_chart(style_plot(fig, 420, legend=False), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown(f"### {T['historical_events']}")
        ev_table = ev[["start_date", "event_name", "venue", "presences_supported_additional"]].copy()
        if lang == "IT":
            ev_table.columns = ["Data inizio", "Evento", "Sede", "Presenze aggiuntive supportate"]
            ev_fmt = {"Presenze aggiuntive supportate": "{:,.0f}"}
        else:
            ev_table.columns = ["Start date", "Event", "Venue", "Supported additional presences"]
            ev_fmt = {"Supported additional presences": "{:,.0f}"}
        st.dataframe(ev_table.style.format(ev_fmt), width="stretch", hide_index=True)

    with right:
        st.markdown(f"### {T['scenario_engine']}")
        st.markdown(
            f"""
            <div class="section-card">
            <b>{T['best_month']}:</b> <span class="accent">{sc['best_month']}</span><br><br>
            <b>{T['additional_pres']}:</b> {sc['best_month_additional_presences']:,.0f}<br>
            <b>{T['additional_int_pres']}:</b> {sc['best_month_additional_international_presences']:,.0f}<br>
            <b>{T['direct_impact']}:</b> €{sc['best_month_direct_impact_eur']:,.0f}<br><br>
            <span class="small-note">{T['scenario_note']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_quality:
    st.markdown(f"## {T['quality_title']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["forecast_rows"], f"{int(kpi_value('forecast_rows_with_95PI'))}")
    c2.metric(T["models_pi"], f"{int(kpi_value('models_with_95PI'))}/10")
    c3.metric(T["reproduction"], f"{int(kpi_value('interval_reproduction_MATCH'))}/10")
    c4.metric(T["close_review"], f"{int(kpi_value('interval_reproduction_CLOSE_REVIEW'))}/10")

    stat = intervals["reproduction_status"].value_counts().rename_axis("status").reset_index(name="count")
    fig = px.bar(stat, x="status", y="count", title=T["interval_status"], labels={"status": "Status", "count": T["models"]})
    fig.update_traces(
        marker_color=[ORANGE if s == "CLOSE_REVIEW" else NAVY_2 for s in stat["status"]],
        text="count", textposition="outside", textfont=dict(color=TEXT),
    )
    st.plotly_chart(style_plot(fig, 360, legend=False), width="stretch")

    st.markdown(f"### {T['pi_validation']}")
    quality_cols = ["city", "target", "model_family", "expected_final_MAPE", "reproduced_MAPE", "MAPE_difference_pp", "reproduction_status", "empirical_2024_coverage_pct"]
    quality_table = intervals[quality_cols].copy()
    quality_table["target"] = quality_table["target"].map(lambda x: translated_target(x, lang))
    if lang == "IT":
        quality_table.columns = ["Città", "Target", "Famiglia modello", "MAPE finale atteso", "MAPE riprodotto", "Differenza MAPE (pp)", "Stato riproduzione", "Copertura 2024 (%)"]
        qfmt = {"MAPE finale atteso": "{:.3f}%", "MAPE riprodotto": "{:.3f}%", "Differenza MAPE (pp)": "{:.3f}", "Copertura 2024 (%)": "{:.0f}%"}
    else:
        quality_table.columns = ["City", "Target", "Model family", "Expected final MAPE", "Reproduced MAPE", "MAPE difference (pp)", "Reproduction status", "2024 coverage (%)"]
        qfmt = {"Expected final MAPE": "{:.3f}%", "Reproduced MAPE": "{:.3f}%", "MAPE difference (pp)": "{:.3f}", "2024 coverage (%)": "{:.0f}%"}
    st.dataframe(quality_table.style.format(qfmt), width="stretch", hide_index=True)

    st.markdown(f"### {T['limits']}")
    st.markdown("\n".join([f"- {item}" for item in T["limit_items"]]))

st.markdown("---")
st.caption(T["footer"])
