import streamlit as st
import pandas as pd
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

# ---- Emission factors, aligned with the main dashboard ----
# Electricity: implied by data/zone_energy.csv (institute-anchored Scope 2 reconstruction),
# i.e. total zone CO2e / total zone kWh, so both pages always agree.
_zone_energy = pd.read_csv(os.path.join(DATA_DIR, "zone_energy.csv"))
GRID_EF = (
    _zone_energy["annual_co2e_tonnes"].sum() * 1000
    / (_zone_energy["gross_electricity_gwh_yr"].sum() * 1_000_000)
)  # kg CO2e / kWh
LPG_EF = 2.99          # kg CO2e / kg LPG, combustion basis (IPCC 2006), same as dashboard
KG_PER_CYLINDER = 14.2  # commercial LPG cylinder
CAMPUS_GWH_2025_26 = 59.49  # campus gross electricity, 2025-26 actual

# Model was trained on anonymised mess codes. Fill in the real names here once the
# colleague confirms the mapping, e.g. "Mess_A": "Himalaya". Unmapped codes show as-is.
MESS_LABELS = {
    "Mess_A": "Himalaya 1 (Mess_A)", "Mess_B": "Himalaya 2 (Mess_B)", "Mess_C": "Himalaya 3 (Mess_C)",
    "Mess_D": "Himalaya 4 (Mess_D)", "Mess_E": "Himalaya 5 (Mess_E)",
    "Mess_F": "Nilgiri 1 (Mess_F)", "Mess_G": "Nilgiri 2 (Mess_G)",
    "Mess_H": "Vindhya 1 (Mess_H)", "Mess_I": "Vindhya 2 (Mess_I)",
}
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Forecast | Campus Digital Twin",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def _load(name):
    return joblib.load(os.path.join(MODELS, name))


electricity_model = _load("electricity_xgboost_model.pkl")

lpg_model = _load("lpg_xgboost_model.pkl")


# ============================================================
# CSS — ENTERPRISE DARK UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif;
}

.stApp {
    background: #041f0b;
    color: #eef8cf;
}

/* Remove default top padding */
.block-container {
    padding-top: 3.2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1600px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c2c16;
    border-right: 1px solid #123a1e;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

/* Sidebar text */
.sidebar-brand {
    font-size: 18px;
    font-weight: 700;
    color: #eef8cf;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #9dbf5e;
    margin-bottom: 22px;
}

.nav-title {
    font-size: 10px;
    font-weight: 600;
    color: #9dbf5e;
    letter-spacing: 1px;
    margin-top: 22px;
    margin-bottom: 8px;
}

.nav-item {
    font-size: 13px;
    color: #9dbf5e;
    padding: 7px 9px;
    border-radius: 4px;
    margin-bottom: 2px;
}

.nav-active {
    background: #123a1e;
    color: #d5f379;
    border-left: 2px solid #d5f379;
}

/* Header */
.page-title {
    font-size: 25px;
    font-weight: 650;
    color: #eef8cf;
    margin-bottom: 2px;
}

.page-subtitle {
    font-size: 11px;
    color: #9dbf5e;
}

/* Status bar */
.status-bar {
    background: #0c2c16;
    border: 1px solid #123a1e;
    border-radius: 4px;
    padding: 9px 13px;
    margin: 18px 0 14px 0;
    font-size: 11px;
    color: #9dbf5e;
}

.status-dot {
    color: #d5f379;
    font-weight: 700;
}

/* KPI cards */
.kpi {
    background: #0c2c16;
    border: 1px solid #123a1e;
    border-radius: 5px;
    padding: 15px;
    height: 105px;
}

.kpi-label {
    font-size: 9px;
    color: #9dbf5e;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 9px;
}

.kpi-value {
    font-size: 24px;
    font-weight: 650;
    color: #eef8cf;
}

.kpi-unit {
    font-size: 10px;
    color: #9dbf5e;
    margin-top: 4px;
}

.green {
    color: #d5f379;
}

.blue {
    color: #9dbf5e;
}

.orange {
    color: #bfe06d;
}

/* Section */
.section {
    font-size: 13px;
    font-weight: 600;
    color: #eef8cf;
    margin-top: 25px;
    margin-bottom: 9px;
}

/* Panel */
.panel {
    background: #0c2c16;
    border: 1px solid #123a1e;
    border-radius: 5px;
    padding: 14px;
}

/* Table */
.dataframe {
    font-size: 11px !important;
}

/* Badges */
.badge-healthy {
    color: #d5f379;
    background: #0c2c16;
    border: 1px solid #4d6932;
    padding: 3px 7px;
    border-radius: 3px;
    font-size: 9px;
}

.badge-active {
    color: #9dbf5e;
    background: #0c2c16;
    border: 1px solid #4d6932;
    padding: 3px 7px;
    border-radius: 3px;
    font-size: 9px;
}

.badge-info {
    color: #9dbf5e;
    background: #123a1e;
    border: 1px solid #2a4a2e;
    padding: 3px 7px;
    border-radius: 3px;
    font-size: 9px;
}

/* Small text */
.small-text {
    color: #9dbf5e;
    font-size: 10px;
}

/* Footer */
.footer {
    margin-top: 35px;
    padding-top: 12px;
    border-top: 1px solid #123a1e;
    color: #9dbf5e;
    font-size: 9px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">ML FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">XGBoost models · electricity &amp; mess LPG</div>', unsafe_allow_html=True)
    st.page_link("app.py", label="Campus map & carbon dashboard")
    st.page_link("pages/2_ML_Forecast.py", label="ML forecast & scenarios")
    st.markdown("---")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="page-title">ML Forecast &amp; Scenarios</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'IIT Madras · Carbon emission monitoring and energy forecasting'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STATUS BAR
# ============================================================

st.markdown(
    """
    <div class="status-bar">
        <span class="status-dot">●</span>
        &nbsp; Digital Twin operational
        &nbsp;&nbsp;·&nbsp;&nbsp;
        2 active prediction models
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Last data update: 2025-26
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="nav-title">SCENARIO INPUT</div>',
        unsafe_allow_html=True
    )

    mess = st.selectbox(
        "Mess",
        [
            "Mess_A",
            "Mess_B",
            "Mess_C",
            "Mess_D",
            "Mess_E",
            "Mess_F",
            "Mess_G",
            "Mess_H",
            "Mess_I"
        ],
        format_func=lambda m: MESS_LABELS.get(m, m),
        label_visibility="collapsed"
    )

    month = st.slider(
        "Prediction month",
        1,
        12,
        9
    )

    year = st.number_input(
        "Prediction year",
        2026,
        2035,
        2026
    )

    previous_lpg = st.number_input(
        "Previous month LPG",
        min_value=0.0,
        value=259.0,
        step=1.0
    )

    vacation_flag = st.selectbox(
        "Campus vacation",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    horizon = st.slider(
        "Electricity forecast horizon (years)",
        1,
        10,
        10
    )


# ============================================================
# ELECTRICITY MODEL
# ============================================================

# Year_Index 1 = 2022-23, so the selected year Y maps to academic year Y-(Y+1),
# e.g. 2026 -> index 5 -> 2026-27. (Was hard-coded to 5, so changing the year did nothing.)
year_index = int(year) - 2021
sel_ay = f"{int(year)}-{str(int(year) + 1)[-2:]}"

next_year = pd.DataFrame({
    "Year_Index": [year_index]
})

predicted_electricity = electricity_model.predict(
    next_year
)[0]

electricity_kwh = predicted_electricity * 1_000_000

electricity_co2 = (
    electricity_kwh * GRID_EF
) / 1000


# ============================================================
# LPG MODEL
# ============================================================

lpg_input = pd.DataFrame({
    "Month_Number": [month],
    "Year": [year],
    "Previous_Month_LPG": [previous_lpg],
    "Vacation_Flag": [vacation_flag],

    "Mess_Mess_A": [1 if mess == "Mess_A" else 0],
    "Mess_Mess_B": [1 if mess == "Mess_B" else 0],
    "Mess_Mess_C": [1 if mess == "Mess_C" else 0],
    "Mess_Mess_D": [1 if mess == "Mess_D" else 0],
    "Mess_Mess_E": [1 if mess == "Mess_E" else 0],
    "Mess_Mess_F": [1 if mess == "Mess_F" else 0],
    "Mess_Mess_G": [1 if mess == "Mess_G" else 0],
    "Mess_Mess_H": [1 if mess == "Mess_H" else 0],
    "Mess_Mess_I": [1 if mess == "Mess_I" else 0]
})

predicted_lpg = lpg_model.predict(
    lpg_input
)[0]

lpg_kg = predicted_lpg * KG_PER_CYLINDER

lpg_co2 = (
    lpg_kg * LPG_EF
) / 1000


# ============================================================
# TOTAL
# ============================================================

total_co2 = electricity_co2 + lpg_co2

electricity_share = (
    electricity_co2 / total_co2
) * 100

lpg_share = (
    lpg_co2 / total_co2
) * 100


# ============================================================
# KPI ROW
# ============================================================

st.markdown(
    '<div class="section">CURRENT FORECAST</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">ELECTRICITY DEMAND · {sel_ay}</div>
            <div class="kpi-value blue">{predicted_electricity:.2f}</div>
            <div class="kpi-unit">GWh / year</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">LPG · {MESS_LABELS.get(mess, mess).upper()} · {month:02d}/{year}</div>
            <div class="kpi-value orange">{predicted_lpg:.2f}</div>
            <div class="kpi-unit">cylinders / month · {lpg_co2:,.2f} t CO₂e</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">TOTAL CO₂</div>
            <div class="kpi-value green">{total_co2:,.2f}</div>
            <div class="kpi-unit">tonnes CO₂e</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">ELECTRICITY SHARE</div>
            <div class="kpi-value">{electricity_share:.1f}%</div>
            <div class="kpi-unit">of combined emissions</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TREND DATA
# ============================================================

trend_df = pd.DataFrame({
    "Year": [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"
    ],
    "Electricity": [
        49.16,
        52.15,
        58.21,
        59.49
    ]
})


# ============================================================
# ELECTRICITY TREND
# ============================================================

st.markdown(
    '<div class="section">ELECTRICITY DEMAND TREND</div>',
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=trend_df["Year"],
        y=trend_df["Electricity"],
        mode="lines+markers",
        line=dict(
            color="#9dbf5e",
            width=2
        ),
        marker=dict(
            size=6,
            color="#9dbf5e"
        ),
        name="Gross electricity"
    )
)

fig.add_trace(
    go.Scatter(
        x=["2025-26", sel_ay],
        y=[CAMPUS_GWH_2025_26, predicted_electricity],
        mode="lines+markers",
        line=dict(
            color="#d5f379",
            width=2,
            dash="dash"
        ),
        marker=dict(
            size=7,
            color="#d5f379"
        ),
        name="Forecast"
    )
)

fig.update_layout(
    height=290,
    margin=dict(l=10, r=10, t=15, b=10),
    paper_bgcolor="#0c2c16",
    plot_bgcolor="#0c2c16",
    font=dict(
        color="#9dbf5e",
        size=10
    ),
    xaxis=dict(
        gridcolor="#123a1e",
        showline=False
    ),
    yaxis=dict(
        title="GWh",
        gridcolor="#123a1e",
        showline=False
    ),
    legend=dict(
        orientation="h",
        y=1.08,
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)

# ============================================================
# FORECAST MONITOR
# ============================================================

st.markdown(
    '<div class="section">ELECTRICITY FORECAST</div>',
    unsafe_allow_html=True
)

forecast_change = (
    (predicted_electricity - CAMPUS_GWH_2025_26) / CAMPUS_GWH_2025_26
) * 100

fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{sel_ay} FORECAST</div>
            <div class="kpi-value blue">
                {predicted_electricity:.2f}
            </div>
            <div class="kpi-unit">GWh / year</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with fc2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">FORECAST CO₂</div>
            <div class="kpi-value green">
                {electricity_co2:,.2f}
            </div>
            <div class="kpi-unit">tonnes CO₂</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with fc3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">YEAR-OVER-YEAR CHANGE</div>
            <div class="kpi-value">
                {forecast_change:+.2f}%
            </div>
            <div class="kpi-unit">vs 2025-26 actual</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# MULTI-YEAR ELECTRICITY OUTLOOK
# ============================================================
# The electricity model was trained on 4 actual years + 20 trend-extrapolated
# estimates (see ml_training/electricity_model.py), so it covers Year_Index 5-24.
# Year_Index 1 = 2022-23, so index 5 = 2026-27 and index 14 = 2035-36.

def _ay(idx):
    start = 2021 + idx
    return f"{start}-{str(start + 1)[-2:]}"

outlook_idx = list(range(5, 5 + horizon))
outlook_gwh = electricity_model.predict(
    pd.DataFrame({"Year_Index": outlook_idx})
)
outlook = pd.DataFrame({
    "Academic year": [_ay(i) for i in outlook_idx],
    "GWh": outlook_gwh,
})
outlook["CO2e (t)"] = outlook["GWh"] * 1_000_000 * GRID_EF / 1000
outlook["vs 2025-26"] = (outlook["GWh"] / CAMPUS_GWH_2025_26 - 1) * 100

st.markdown(
    f'<div class="section">{horizon}-YEAR ELECTRICITY OUTLOOK</div>',
    unsafe_allow_html=True
)

last = outlook.iloc[-1]
oc1, oc2, oc3 = st.columns(3)
for col, label, val, unit in [
    (oc1, f"{last['Academic year']} DEMAND", f"{last['GWh']:.2f}", "GWh / year"),
    (oc2, f"{last['Academic year']} CO₂", f"{last['CO2e (t)']:,.0f}", "tonnes CO₂e / year"),
    (oc3, "CUMULATIVE CO₂", f"{outlook['CO2e (t)'].sum():,.0f}",
     f"tonnes CO₂e, {outlook['Academic year'].iloc[0]} to {last['Academic year']}"),
]:
    with col:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value green">{val}</div>
                <div class="kpi-unit">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

hist_years = ["2022-23", "2023-24", "2024-25", "2025-26"]
hist_gwh = [49.16, 52.15, 58.21, 59.49]

outlook_fig = go.Figure()
outlook_fig.add_trace(go.Scatter(
    x=hist_years, y=hist_gwh, mode="lines+markers", name="Actual",
    line=dict(color="#9dbf5e", width=2), marker=dict(size=7, color="#9dbf5e")
))
outlook_fig.add_trace(go.Scatter(
    x=["2025-26"] + list(outlook["Academic year"]),
    y=[CAMPUS_GWH_2025_26] + list(outlook["GWh"]),
    mode="lines+markers", name="XGBoost forecast",
    line=dict(color="#d5f379", width=2, dash="dash"), marker=dict(size=6, color="#d5f379")
))
try:
    _est = pd.read_csv(os.path.join(DATA_DIR, "electricity_training_data.csv"))
    _est = _est[(_est["Data_Type"] != "Actual") & (_est["Year_Index"].isin(outlook_idx))]
    outlook_fig.add_trace(go.Scatter(
        x=_est["Academic_Year"], y=_est["Total_Gross_GWh"], mode="markers",
        name="Extrapolated training estimate",
        marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color="#5f7f3e", width=1.5))
    ))
except Exception:
    pass
outlook_fig.update_layout(
    height=320, margin=dict(l=10, r=10, t=15, b=10),
    paper_bgcolor="#0c2c16", plot_bgcolor="#0c2c16",
    font=dict(color="#9dbf5e", size=10),
    xaxis=dict(gridcolor="#123a1e"), yaxis=dict(title="GWh / year", gridcolor="#123a1e"),
    legend=dict(orientation="h", y=1.1, x=0)
)
st.plotly_chart(outlook_fig, use_container_width=True, config={"displayModeBar": False})

outlook_display = outlook.copy()
outlook_display["GWh"] = outlook_display["GWh"].map(lambda x: f"{x:.2f}")
outlook_display["CO2e (t)"] = outlook_display["CO2e (t)"].map(lambda x: f"{x:,.0f}")
outlook_display["vs 2025-26"] = outlook_display["vs 2025-26"].map(lambda x: f"{x:+.1f}%")
st.dataframe(outlook_display, use_container_width=True, hide_index=True,
             height=min(40 + 35 * horizon, 400))
st.caption(
    "Only 4 actual years exist (2022-23 to 2025-26). With the professor's approval, the model was "
    "trained on those plus 20 estimated years extrapolated from their linear trend, so this outlook "
    "follows that trend. It assumes past growth continues and ignores new buildings, efficiency "
    f"measures or solar additions. CO₂e uses {GRID_EF:.3f} kg/kWh throughout."
)

# ============================================================
# EMISSION SOURCE MONITOR
# ============================================================

st.markdown(
    '<div class="section">EMISSION SOURCE MONITOR</div>',
    unsafe_allow_html=True
)

source_data = pd.DataFrame({
    "Source": [
        "Grid Electricity",
        "LPG — Mess Operations"
    ],
    "Status": [
        "ACTIVE",
        "ACTIVE"
    ],
    "Predicted": [
        f"{predicted_electricity:.2f} GWh",
        f"{predicted_lpg:.2f} cylinders"
    ],
    "CO₂": [
        f"{electricity_co2:,.2f} t",
        f"{lpg_co2:,.2f} t"
    ],
    "Share": [
        f"{electricity_share:.2f}%",
        f"{lpg_share:.2f}%"
    ]
})

st.dataframe(
    source_data,
    use_container_width=True,
    hide_index=True,
    height=115
)

# ============================================================
# CARBON ALERTS
# ============================================================

st.markdown(
    '<div class="section">ALERTS</div>',
    unsafe_allow_html=True
)

alerts = []

# Electricity forecast alert
if predicted_electricity > CAMPUS_GWH_2025_26 * 1.05:
    alerts.append(
        ("WARNING", "Electricity demand is more than 5% above 2025-26.")
    )
else:
    alerts.append(
        ("HEALTHY", "Electricity forecast is within the normal range.")
    )

# LPG alert
if predicted_lpg > previous_lpg * 1.10:
    alerts.append(
        ("WARNING", "LPG consumption is more than 10% above previous month.")
    )
else:
    alerts.append(
        ("HEALTHY", "LPG consumption is within the expected range.")
    )

# Total carbon alert
baseline_co2 = (
    CAMPUS_GWH_2025_26 * 1_000_000 * GRID_EF / 1000
    + previous_lpg * KG_PER_CYLINDER * LPG_EF / 1000
)  # 2025-26 electricity + this mess at last month's LPG

if total_co2 > baseline_co2 * 1.05:
    alerts.append(
        ("WARNING", "Projected carbon emissions exceed the baseline by more than 5%.")
    )
else:
    alerts.append(
        ("HEALTHY", "Projected carbon emissions are within the baseline range.")
    )

alert_df = pd.DataFrame(
    alerts,
    columns=["Status", "Message"]
)

st.dataframe(
    alert_df,
    use_container_width=True,
    hide_index=True,
    height=140
)

# ============================================================
# MODEL HEALTH
# ============================================================

st.markdown(
    '<div class="section">MODEL HEALTH</div>',
    unsafe_allow_html=True
)

model_data = pd.DataFrame({
    "MODEL": [
        "electricity-xgb-v2",
        "lpg-xgb-v1"
    ],
    "STATUS": [
        "HEALTHY",
        "HEALTHY"
    ],
    "TYPE": [
        "XGBoost Regressor",
        "XGBoost Regressor"
    ],
    "INPUT": [
        "Annual campus demand",
        "Mess-level monthly consumption"
    ],
    "OUTPUT": [
        "GWh",
        "Cylinders"
    ]
})

st.dataframe(
    model_data,
    use_container_width=True,
    hide_index=True,
    height=110
)

# ============================================================
# HISTORICAL CARBON TRENDS
# ============================================================

st.markdown(
    '<div class="section">HISTORICAL CARBON TREND</div>',
    unsafe_allow_html=True
)

historical_carbon = pd.DataFrame({
    "Year": [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"
    ],

    "Electricity_GWh": [
        49.16,
        52.15,
        58.21,
        59.49
    ]
})

# Convert electricity to CO2
historical_carbon["CO2_tonnes"] = (
    historical_carbon["Electricity_GWh"]
    * 1_000_000
    * GRID_EF
    / 1000
)

# Year-over-year change
historical_carbon["YoY"] = (
    historical_carbon["CO2_tonnes"]
    .pct_change()
    * 100
)

# Carbon trend chart
carbon_fig = go.Figure()

carbon_fig.add_trace(
    go.Scatter(
        x=historical_carbon["Year"],
        y=historical_carbon["CO2_tonnes"],
        mode="lines+markers",
        name="Historical CO₂",
        line=dict(
            color="#d5f379",
            width=2
        ),
        marker=dict(
            size=6
        )
    )
)

carbon_fig.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="#0c2c16",
    plot_bgcolor="#0c2c16",

    font=dict(
        color="#9dbf5e",
        size=10
    ),

    xaxis=dict(
        gridcolor="#123a1e"
    ),

    yaxis=dict(
        title="CO₂ (tonnes)",
        gridcolor="#123a1e"
    ),

    showlegend=False
)

st.plotly_chart(
    carbon_fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)

# Historical table

historical_display = historical_carbon.copy()

historical_display["Electricity_GWh"] = (
    historical_display["Electricity_GWh"]
    .map(lambda x: f"{x:.2f} GWh")
)

historical_display["CO2_tonnes"] = (
    historical_display["CO2_tonnes"]
    .map(lambda x: f"{x:,.2f} t")
)

historical_display["YoY"] = (
    historical_display["YoY"]
    .map(
        lambda x:
        "—" if pd.isna(x)
        else f"{x:+.2f}%"
    )
)

historical_display.columns = [
    "Academic Year",
    "Electricity",
    "CO₂ Emissions",
    "YoY Change"
]

st.dataframe(
    historical_display,
    use_container_width=True,
    hide_index=True,
    height=180
)

# ============================================================
# ZONE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section">ZONE ELECTRICITY ANALYSIS</div>',
    unsafe_allow_html=True
)

zone_df = pd.DataFrame({
    "Zone": _zone_energy["zone_name"],
    "Electricity (GWh)": _zone_energy["gross_electricity_gwh_yr"],
})  # read from data/zone_energy.csv, same figures as the main dashboard

zone_df["Share"] = (
    zone_df["Electricity (GWh)"]
    / zone_df["Electricity (GWh)"].sum()
) * 100

zone_df["CO2 (tonnes)"] = (
    zone_df["Electricity (GWh)"]
    * 1000000
    * GRID_EF
    / 1000
)

# Zone chart
zone_fig = go.Figure()

zone_fig.add_trace(
    go.Bar(
        x=zone_df["Zone"],
        y=zone_df["Electricity (GWh)"],
        marker_color="#9dbf5e",
        name="Electricity"
    )
)

zone_fig.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=15, b=10),
    paper_bgcolor="#0c2c16",
    plot_bgcolor="#0c2c16",
    font=dict(
        color="#9dbf5e",
        size=10
    ),
    xaxis=dict(
        gridcolor="#123a1e"
    ),
    yaxis=dict(
        title="GWh",
        gridcolor="#123a1e"
    ),
    showlegend=False
)

st.plotly_chart(
    zone_fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)

# Zone table
zone_display = zone_df.copy()

zone_display["Electricity (GWh)"] = (
    zone_display["Electricity (GWh)"]
    .map(lambda x: f"{x:.2f} GWh")
)

zone_display["Share"] = (
    zone_display["Share"]
    .map(lambda x: f"{x:.2f}%")
)

zone_display["CO2 (tonnes)"] = (
    zone_display["CO2 (tonnes)"]
    .map(lambda x: f"{x:,.2f} t")
)

st.dataframe(
    zone_display,
    use_container_width=True,
    hide_index=True,
    height=185
)

# ============================================================
# SCENARIO EXPLORER
# ============================================================

st.markdown(
    '<div class="section">SCENARIO EXPLORER</div>',
    unsafe_allow_html=True
)

scenario_reduction = st.slider(
    "Electricity demand reduction",
    min_value=0,
    max_value=30,
    value=10,
    step=1
)

lpg_reduction = st.slider(
    "LPG consumption reduction",
    min_value=0,
    max_value=30,
    value=0,
    step=1
)

scenario_electricity = (
    predicted_electricity
    * (1 - scenario_reduction / 100)
)

scenario_electricity_co2 = (
    scenario_electricity
    * 1_000_000
    * GRID_EF
    / 1000
)

scenario_lpg_co2 = (
    lpg_co2
    * (1 - lpg_reduction / 100)
)

scenario_total_co2 = (
    scenario_electricity_co2
    + scenario_lpg_co2
)
co2_saved = total_co2 - scenario_total_co2

scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

with scenario_col1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">SCENARIO ELECTRICITY</div>
            <div class="kpi-value blue">
                {scenario_electricity:.2f}
            </div>
            <div class="kpi-unit">GWh / year</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with scenario_col2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">SCENARIO CO₂</div>
            <div class="kpi-value green">
                {scenario_total_co2:,.2f}
            </div>
            <div class="kpi-unit">tonnes CO₂</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Scenario comparison chart

scenario_chart = go.Figure()

scenario_chart.add_trace(
    go.Bar(
        x=["Baseline", "Scenario"],
        y=[total_co2, scenario_total_co2],
        marker_color=["#5f7f3e", "#d5f379"],
        text=[
            f"{total_co2:,.2f} t",
            f"{scenario_total_co2:,.2f} t"
        ],
        textposition="outside"
    )
)

scenario_chart.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="#0c2c16",
    plot_bgcolor="#0c2c16",
    font=dict(
        color="#9dbf5e",
        size=10
    ),
    yaxis=dict(
        title="CO₂ (tonnes)",
        gridcolor="#123a1e"
    ),
    xaxis=dict(
        gridcolor="#123a1e"
    ),
    showlegend=False
)

st.plotly_chart(
    scenario_chart,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)

with scenario_col3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">CO₂ SAVED</div>
            <div class="kpi-value green">
                {co2_saved:,.2f}
            </div>
            <div class="kpi-unit">tonnes CO₂</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SCENARIO SUMMARY
# ============================================================

st.markdown(
    '<div class="section">ACTIVE SCENARIO</div>',
    unsafe_allow_html=True
)

scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

with scenario_col1:

    st.markdown(
        f"""
        <div class="panel">
            <div class="kpi-label">MESS</div>
            <div style="font-size:18px;color:#eef8cf;">
                {MESS_LABELS.get(mess, mess)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with scenario_col2:

    st.markdown(
        f"""
        <div class="panel">
            <div class="kpi-label">FORECAST PERIOD</div>
            <div style="font-size:18px;color:#eef8cf;">
                {month:02d} / {year}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with scenario_col3:

    st.markdown(
        f"""
        <div class="panel">
            <div class="kpi-label">VACATION STATUS</div>
            <div style="font-size:18px;color:#eef8cf;">
                {"VACATION" if vacation_flag else "NORMAL"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CARBONTWIN · IIT MADRAS · ENERGY INTELLIGENCE PLATFORM
    </div>
    """,
    unsafe_allow_html=True
)