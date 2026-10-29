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
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: 3px solid {ORANGE};
    }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF; }}
    [data-testid="stSidebar"] .stRadio label {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.20); }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px; }}
    h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.02em; }}
    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-top: 3px solid {ORANGE};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(23,55,78,.05);
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 650; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 800; }}
    .hero {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 24px 28px; margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(23,55,78,.05);
    }}
    .eyebrow {{ color: #355D7A; font-size: .78rem; font-weight: 800; letter-spacing: .18em; }}
    .hero-title {{ color: {TEXT}; font-size: 2.25rem; font-weight: 850; margin-top: .35rem; }}
    .hero-sub {{ color: #496279; font-size: 1.05rem; margin-top: .45rem; }}
    .info-box {{
        background: {PALE_BLUE}; border-radius: 10px; padding: 14px 16px;
        color: {TEXT}; border: 1px solid #CCDDF8; margin: 10px 0 18px 0;
    }}
    .section-card {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(23,55,78,.04);
    }}
    .small-note {{ color: {MUTED}; font-size: .90rem; }}
    .accent {{ color: {ORANGE}; font-weight: 800; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 20px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ color: {TEXT}; font-weight: 650; }}
    .stTabs [aria-selected="true"] {{ border-bottom: 3px solid {ORANGE}; }}
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

def kpi_value(name, default=np.nan):
    row = kpis.loc[kpis["kpi"] == name, "value"]
    return float(row.iloc[0]) if len(row) else default


def style_plot(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
        hoverlabel=dict(bgcolor="white", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def header():
    col_logo, col_title = st.columns([1, 6], vertical_alignment="center")

    with col_logo:
        st.image(str(HERO_LOGO_PATH), width=105)

    with col_title:
        st.markdown(
            """
            <div class="eyebrow">
                SBL CONSULTANCY / SPORT TOURISM ANALYTICS
            </div>
            <div class="hero-title">
                Sport Tourism Demand Predictor
            </div>
            <div class="hero-sub">
                Forecast accuracy, city comparison and supported sports-event
                impact across five Italian destinations.
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.image(str(LOGO_PATH), width=190)

    st.markdown("## Sport Tourism")
    st.markdown("### Demand Predictor")
    st.caption("PYTHON · FORECASTING · EVENT ANALYTICS")
    st.markdown("Predictive tourism analytics for Torino, Milano, Roma, Napoli and Bologna.")
    st.markdown("---")
    st.markdown("### Project scope")
    st.markdown("**Cities:** 5")
    st.markdown("**Targets:** Arrivals + Presences")
    st.markdown("**Final models:** 10")
    st.markdown("**Forecast test:** 2024")
    st.markdown("---")
    st.markdown("### Key validation")
    st.markdown(f"**Mean final MAPE:** {kpi_value('mean_final_MAPE'):.2f}%")
    st.markdown(f"**95% PI coverage:** {kpi_value('mean_empirical_2024_PI_coverage'):.0f}%")
    st.markdown(f"**Improved models:** {int(kpi_value('models_improved_vs_base'))}/10")
    st.markdown("---")
    st.caption("Event effects are predictive associations, not causal estimates. Davis Cup evidence is supported only for Bologna presences.")

header()

tab_overview, tab_perf, tab_city, tab_map, tab_event, tab_quality = st.tabs([
    "Overview",
    "Forecast Performance",
    "City Explorer",
    "Geographic Map",
    "Event Impact",
    "Model & Data Quality",
])

with tab_overview:
    st.markdown("## Project Overview")
    st.markdown(
        '<div class="info-box">The dashboard compares the final out-of-sample forecasting performance of 10 city-target models and separates validated forecasting gains from event-effect evidence.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", f"{int(kpi_value('cities'))}")
    c2.metric("Final models", f"{int(kpi_value('forecast_models'))}")
    c3.metric("Mean final MAPE", f"{kpi_value('mean_final_MAPE'):.2f}%")
    c4.metric("2024 PI coverage", f"{kpi_value('mean_empirical_2024_PI_coverage'):.0f}%")

    st.markdown("### Forecasting performance by city")
    city_perf = perf.pivot(index="city", columns="target", values="final_MAPE").reset_index()
    order = geo.sort_values("mean_final_MAPE")["city"].tolist()
    city_perf["city"] = pd.Categorical(city_perf["city"], categories=order, ordered=True)
    city_perf = city_perf.sort_values("city")

    fig = go.Figure()
    fig.add_bar(name="Arrivals", x=city_perf["city"], y=city_perf["arrivals"], marker_color=NAVY_2)
    fig.add_bar(name="Presences", x=city_perf["city"], y=city_perf["presences"], marker_color=ORANGE)
    fig.update_layout(barmode="group", title="Final MAPE by city and target")
    fig.update_yaxes(title="MAPE (%)")
    st.plotly_chart(style_plot(fig), use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        improvement = perf.copy()
        improvement["series"] = improvement["city"] + " · " + improvement["target"].str.title()
        improvement = improvement.sort_values("improvement_pct", ascending=True)
        fig2 = px.bar(
            improvement,
            x="improvement_pct",
            y="series",
            orientation="h",
            title="Relative improvement vs base model",
            labels={"improvement_pct": "Improvement (%)", "series": ""},
        )
        fig2.update_traces(marker_color=ORANGE)
        st.plotly_chart(style_plot(fig2, height=430, legend=False), use_container_width=True)
    with right:
        best = perf.loc[perf["improvement_pct"].idxmax()]
        napoli = geo.loc[geo["city"] == "Napoli"].iloc[0]
        st.markdown("### Key insights")
        st.markdown(
            f"""
            <div class="section-card">
            <b>Best relative forecasting gain</b><br>
            <span class="accent">{best['city']} · {best['target'].title()}</span><br>
            {best['improvement_pct']:.2f}% improvement vs base model.
            <br><br>
            <b>Lowest city-average final MAPE</b><br>
            <span class="accent">Napoli</span><br>
            {napoli['mean_final_MAPE']:.2f}% across arrivals and presences.
            <br><br>
            <b>Supported event-effect evidence</b><br>
            Bologna · Davis Cup presences: <span class="accent">+12.26%</span> canonical uplift.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_perf:
    st.markdown("## Forecast Performance")
    target_filter = st.segmented_control("Target", options=["All", "arrivals", "presences"], default="All")
    pf = perf.copy() if target_filter == "All" else perf[perf["target"] == target_filter].copy()

    fig = go.Figure()
    for target, color in [("arrivals", NAVY_2), ("presences", ORANGE)]:
        d = pf[pf["target"] == target]
        if not d.empty:
            fig.add_trace(go.Scatter(
                x=d["base_MAPE"], y=d["final_MAPE"], mode="markers+text",
                text=d["city"], textposition="top center", name=target.title(),
                marker=dict(size=14, color=color, line=dict(width=1, color="white")),
                customdata=np.stack([d["improvement_pct"]], axis=-1),
                hovertemplate="%{text}<br>Base MAPE: %{x:.2f}%<br>Final MAPE: %{y:.2f}%<br>Improvement: %{customdata[0]:.2f}%<extra></extra>"
            ))
    maxv = max(perf["base_MAPE"].max(), perf["final_MAPE"].max()) + .4
    fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", line=dict(color="#9AA8B5", dash="dash"), name="No change"))
    fig.update_layout(title="Base vs final MAPE", xaxis_title="Base MAPE (%)", yaxis_title="Final MAPE (%)")
    st.plotly_chart(style_plot(fig, 500), use_container_width=True)

    st.markdown("### Model comparison table")
    table = pf[["city", "target", "sports_exog_retained", "base_MAPE", "final_MAPE", "improvement_pp", "improvement_pct", "improved"]].copy()
    table.columns = ["City", "Target", "Sports exog retained", "Base MAPE", "Final MAPE", "Improvement (pp)", "Improvement (%)", "Improved"]
    st.dataframe(table.style.format({"Base MAPE": "{:.3f}%", "Final MAPE": "{:.3f}%", "Improvement (pp)": "{:.3f}", "Improvement (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)

with tab_city:
    st.markdown("## City Explorer")
    city = st.selectbox("Select city", geo["city"].tolist())
    g = geo.loc[geo["city"] == city].iloc[0]
    p = perf[perf["city"] == city]
    ints = intervals[intervals["city"] == city]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean final MAPE", f"{g['mean_final_MAPE']:.2f}%")
    c2.metric("Arrivals final MAPE", f"{g['arrivals_final_MAPE']:.2f}%")
    c3.metric("Presences final MAPE", f"{g['presences_final_MAPE']:.2f}%")
    c4.metric("Mean improvement", f"{g['mean_improvement_pct']:.2f}%")

    left, right = st.columns([1.15, 1])
    with left:
        comp = p.melt(id_vars=["target"], value_vars=["base_MAPE", "final_MAPE"], var_name="model", value_name="MAPE")
        comp["model"] = comp["model"].map({"base_MAPE": "Base", "final_MAPE": "Final"})
        fig = px.bar(comp, x="target", y="MAPE", color="model", barmode="group", title=f"{city}: base vs final MAPE",
                     color_discrete_map={"Base": "#9FB0C0", "Final": ORANGE})
        fig.update_yaxes(title="MAPE (%)")
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)
    with right:
        st.markdown("### Final model specification")
        display_cols = ["target", "model_family", "order", "seasonal_order", "exogenous_variables", "reproduction_status", "empirical_2024_coverage_pct"]
        st.dataframe(ints[display_cols], use_container_width=True, hide_index=True)
        if bool(g["supported_event_effect"]):
            st.success(g["supported_event_note"])
        else:
            st.info("No statistically supported event-specific effect in the current clean sample.")

with tab_map:
    st.markdown("## Geographic Performance Map")
    st.markdown('<div class="info-box">Point size represents city-average final MAPE. Orange points indicate at least one city-target model improved versus the base specification. Bologna is additionally the only city with a statistically supported event-specific effect.</div>', unsafe_allow_html=True)

    map_df = geo.copy()
    map_df["radius"] = 22000 + map_df["mean_final_MAPE"] * 6500
    map_df["color"] = map_df["any_model_improved"].map({True: [247,147,30,190], False: [32,56,79,185]})

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        get_line_color=[255,255,255],
        line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=42.8, longitude=11.2, zoom=4.7, pitch=0)
    tooltip = {
        "html": "<b>{city}</b><br/>Mean final MAPE: {mean_final_MAPE}%<br/>Arrivals: {arrivals_final_MAPE}%<br/>Presences: {presences_final_MAPE}%<br/>Mean improvement: {mean_improvement_pct}%<br/>{supported_event_note}",
        "style": {"backgroundColor": NAVY, "color": "white"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light"), use_container_width=True)

    st.download_button(
        "Download Kepler-ready CSV",
        data=geo.to_csv(index=False).encode("utf-8"),
        file_name="kepler_sport_tourism_results_v5.csv",
        mime="text/csv",
    )

with tab_event:
    st.markdown("## Supported Event Impact")
    st.warning("Only Davis Cup presences in Bologna are statistically supported in the current clean sample. Other event-specific outputs should not be presented as validated impacts.")

    sc = scenario.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported city", sc["supported_city"])
    c2.metric("Canonical uplift", "+12.26%")
    c3.metric("Mean supported event presences", f"{events['presences_supported_additional'].mean():,.0f}")
    c4.metric("Max supported monthly impact", f"€{sc['max_direct_international_tourism_impact_eur']/1e6:.2f}M")

    ev = events.copy()
    ev["year"] = ev["year"].astype(str)
    fig = px.bar(ev, x="year", y="presences_supported_additional", title="Supported Davis Cup historical attribution — Bologna",
                 labels={"year": "Event year", "presences_supported_additional": "Additional presences"})
    fig.update_traces(marker_color=ORANGE, texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(style_plot(fig, 420, legend=False), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Historical supported events")
        st.dataframe(ev[["start_date", "event_name", "venue", "presences_supported_additional"]].style.format({"presences_supported_additional": "{:,.0f}"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Supported scenario engine")
        st.markdown(
            f"""
            <div class="section-card">
            <b>Best supported month:</b> <span class="accent">{sc['best_month']}</span><br><br>
            <b>Additional presences:</b> {sc['best_month_additional_presences']:,.0f}<br>
            <b>Additional international presences:</b> {sc['best_month_additional_international_presences']:,.0f}<br>
            <b>Direct international tourism impact:</b> €{sc['best_month_direct_impact_eur']:,.0f}<br><br>
            <span class="small-note">Scenario-engine outputs are hypothetical supported monthly placements in Bologna and are not interchangeable with historical event attribution.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_quality:
    st.markdown("## Model & Data Quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast rows with 95% PI", f"{int(kpi_value('forecast_rows_with_95PI'))}")
    c2.metric("Models with 95% PI", f"{int(kpi_value('models_with_95PI'))}/10")
    c3.metric("Reproduction MATCH", f"{int(kpi_value('interval_reproduction_MATCH'))}/10")
    c4.metric("CLOSE REVIEW", f"{int(kpi_value('interval_reproduction_CLOSE_REVIEW'))}/10")

    stat = intervals["reproduction_status"].value_counts().rename_axis("status").reset_index(name="count")
    fig = px.bar(stat, x="status", y="count", title="Interval reproduction status", labels={"status":"Status", "count":"Models"})
    fig.update_traces(marker_color=[ORANGE if s == "CLOSE_REVIEW" else NAVY_2 for s in stat["status"]], text="count", textposition="outside")
    st.plotly_chart(style_plot(fig, 360, legend=False), use_container_width=True)

    st.markdown("### Prediction-interval validation")
    quality_cols = ["city", "target", "model_family", "expected_final_MAPE", "reproduced_MAPE", "MAPE_difference_pp", "reproduction_status", "empirical_2024_coverage_pct"]
    st.dataframe(intervals[quality_cols].style.format({
        "expected_final_MAPE": "{:.3f}%", "reproduced_MAPE": "{:.3f}%", "MAPE_difference_pp": "{:.3f}", "empirical_2024_coverage_pct": "{:.0f}%"
    }), use_container_width=True, hide_index=True)

    st.markdown("### Interpretation limits")
    st.markdown(
        """
        - Sports uplift is a predictive association, not causal identification.
        - The event-effect clean sample covers 2022–2024 and only five cities.
        - Davis Cup support is based on three treated city-months, all in Bologna.
        - Davis Cup effects must not be transported to Torino, Milano, Roma or Napoli.
        - Direct economic impact includes international tourism spending only; indirect and induced effects are not included.
        """
    )

st.markdown("---")
st.caption("Sport Tourism Demand Predictor · Final V5 dashboard · Python / Streamlit")
