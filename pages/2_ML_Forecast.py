from __future__ import annotations

import calendar
from datetime import date

import pandas as pd
import streamlit as st

from src.web_inference import ZONE_COLUMNS, predict_period

st.set_page_config(page_title="ML Carbon Forecast | IITM Carbon Twin", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      .forecast-note {padding: 0.85rem 1rem; border-radius: 0.6rem;
                      background: rgba(110, 160, 95, 0.12); border: 1px solid rgba(110,160,95,.28);}
      div[data-testid="stMetric"] {background: rgba(110,160,95,.08); padding: 12px 14px;
                                     border-radius: 10px; border: 1px solid rgba(110,160,95,.18);}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.page_link("app.py", label="Campus map & carbon dashboard", icon="🗺️")
    st.page_link("pages/2_ML_Forecast.py", label="ML carbon forecast", icon="📈")

st.title("ML-based Carbon Emission Prediction")
st.caption("Date-wise, monthly and yearly predictions using the IITM Date + Zone XGBoost models")

st.markdown(
    """
    <div class="forecast-note">
    Select a date, month or year. The app runs the model for <b>each calendar day</b> in the selected
    period and sums those daily estimates to produce the period total.
    </div>
    """,
    unsafe_allow_html=True,
)

period_mode = st.radio(
    "Prediction period",
    ["Specific date", "Month", "Calendar year", "Financial year (April–March)"],
    horizontal=True,
)

now = date.today()
if period_mode == "Specific date":
    selected_date = st.date_input("Select date", value=now)
    start_date = end_date = selected_date
    period_label = selected_date.strftime("%d %B %Y")
elif period_mode == "Month":
    c1, c2 = st.columns(2)
    with c1:
        selected_month = st.selectbox("Month", list(range(1, 13)), index=now.month - 1,
                                      format_func=lambda m: calendar.month_name[m])
    with c2:
        selected_year = st.number_input("Year", min_value=2022, max_value=2035, value=now.year, step=1)
    start_date = date(int(selected_year), int(selected_month), 1)
    end_date = date(int(selected_year), int(selected_month), calendar.monthrange(int(selected_year), int(selected_month))[1])
    period_label = f"{calendar.month_name[int(selected_month)]} {int(selected_year)}"
elif period_mode == "Calendar year":
    selected_year = st.number_input("Calendar year", min_value=2022, max_value=2035, value=now.year, step=1)
    start_date = date(int(selected_year), 1, 1)
    end_date = date(int(selected_year), 12, 31)
    period_label = str(int(selected_year))
else:
    fy_start = st.number_input("Financial year starting in", min_value=2022, max_value=2035,
                               value=now.year if now.month >= 4 else now.year - 1, step=1)
    start_date = date(int(fy_start), 4, 1)
    end_date = date(int(fy_start) + 1, 3, 31)
    period_label = f"FY{int(fy_start)}-{str((int(fy_start) + 1) % 100).zfill(2)}"

zone_options = ["All four zones", *ZONE_COLUMNS.keys()]
zone_labels = {"All four zones": "All four zones"}
zone_labels.update({key: value[0] for key, value in ZONE_COLUMNS.items()})
selected_zone = st.selectbox("Zone-level view", zone_options, format_func=lambda z: zone_labels[z])

st.caption(
    "Zone models cover Academic & Research, Hostel & Student Life, Residential, and Common / Utilities / Sports / Hospital. "
    "The model's four-zone classification is used consistently with its training data."
)

signature = (start_date.isoformat(), end_date.isoformat(), selected_zone)
run = st.button("Generate prediction", type="primary", use_container_width=False)
if run:
    try:
        with st.spinner(f"Generating daily predictions for {period_label}..."):
            campus_daily, zones_daily = predict_period(start_date, end_date)
        st.session_state["date_zone_forecast"] = {
            "signature": signature,
            "campus_daily": campus_daily,
            "zones_daily": zones_daily,
            "period_label": period_label,
        }
    except Exception as exc:
        st.error(f"Could not generate the prediction: {exc}")
        st.exception(exc)

result = st.session_state.get("date_zone_forecast")
if not result or result.get("signature") != signature:
    st.info("Choose the prediction period and zone, then click **Generate prediction**.")
    st.stop()

campus_daily = result["campus_daily"].copy()
zones_daily = result["zones_daily"].copy()
if selected_zone != "All four zones":
    zones_view = zones_daily.loc[zones_daily["zone"] == selected_zone].copy()
else:
    zones_view = zones_daily.copy()

campus_total = float(campus_daily["predicted_campus_net_selected_sources_tco2e_per_day"].sum())
campus_mean = float(campus_daily["predicted_campus_net_selected_sources_tco2e_per_day"].mean())
zone_totals = (
    zones_view.groupby(["zone", "zone_label"], as_index=False)["predicted_scope2_tco2e_per_day"]
    .sum()
    .rename(columns={"predicted_scope2_tco2e_per_day": "period_scope2_tco2e"})
)

st.divider()
st.subheader(f"Prediction results: {period_label}")

if period_mode == "Specific date":
    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "Campus net selected-source estimate",
        f"{campus_total:,.2f} tCO₂e/day",
    )
    if selected_zone == "All four zones":
        zone_day_total = float(zones_view["predicted_scope2_tco2e_per_day"].sum())
        metric_cols[1].metric("Sum of four zone Scope 2 estimates", f"{zone_day_total:,.2f} tCO₂e/day")
    else:
        zone_day_value = float(zones_view["predicted_scope2_tco2e_per_day"].iloc[0])
        metric_cols[1].metric("Selected-zone Scope 2 estimate", f"{zone_day_value:,.2f} tCO₂e/day")
else:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Campus net total for period", f"{campus_total:,.2f} tCO₂e")
    metric_cols[1].metric("Average campus estimate per day", f"{campus_mean:,.2f} tCO₂e/day")
    metric_cols[2].metric("Days included", f"{len(campus_daily):,}")

    if len(campus_daily) > 1:
        peak_idx = campus_daily["predicted_campus_net_selected_sources_tco2e_per_day"].idxmax()
        peak_row = campus_daily.loc[peak_idx]
        st.caption(
            f"Highest predicted campus day: {pd.Timestamp(peak_row['date']).strftime('%d %b %Y')} "
            f"({peak_row['predicted_campus_net_selected_sources_tco2e_per_day']:,.2f} tCO₂e/day)."
        )

st.markdown("#### Zone-wise Scope 2 total")
if selected_zone == "All four zones":
    st.dataframe(zone_totals, use_container_width=True, hide_index=True,
                 column_config={"period_scope2_tco2e": st.column_config.NumberColumn("Period Scope 2 (tCO₂e)", format="%.3f")})
    st.caption(
        f"Combined four-zone electricity Scope 2 subtotal: "
        f"{zone_totals['period_scope2_tco2e'].sum():,.2f} tCO₂e. This is not the campus net selected-source total below."
    )
else:
    st.metric(
        f"{zone_labels[selected_zone]} — Scope 2 total",
        f"{float(zone_totals['period_scope2_tco2e'].sum()):,.2f} tCO₂e" if not zone_totals.empty else "0.00 tCO₂e",
    )

if period_mode != "Specific date" and len(campus_daily) > 1:
    st.markdown("#### Daily campus emissions across the selected period")
    chart_data = campus_daily.set_index("date")[["predicted_campus_net_selected_sources_tco2e_per_day"]]
    st.line_chart(chart_data, y="predicted_campus_net_selected_sources_tco2e_per_day")

st.markdown("#### Daily prediction table")
zone_daily_pivot = zones_view.pivot(index="date", columns="zone_label", values="predicted_scope2_tco2e_per_day").reset_index()
daily_table = campus_daily.merge(zone_daily_pivot, on="date", how="left")
daily_table = daily_table.rename(columns={
    "predicted_campus_net_selected_sources_tco2e_per_day": "Campus net selected-source (tCO₂e/day)"
})
st.dataframe(daily_table, use_container_width=True, hide_index=True)

csv = daily_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download all daily predictions (CSV)",
    data=csv,
    file_name=f"iitm_carbon_predictions_{start_date.isoformat()}_to_{end_date.isoformat()}.csv",
    mime="text/csv",
)

with st.expander("How to interpret these predictions"):
    st.markdown(
        """
        - **Zone prediction:** electricity Scope 2 only, attributed to one of the four model zones.
        - **Campus prediction:** a separate campus-wide selected-source net subtotal: electricity Scope 2 + LPG combustion + selected vehicle, biomass and refrigerant priors − provisional tree sink. Natural gas is excluded pending overlap reconciliation.
        - **Monthly/yearly totals:** the app predicts each day in the selected date range and sums the daily model outputs. It does not multiply one daily value by the number of days.
        - **Important limitation:** the training targets are reconstructed estimates based on annual/monthly source data and documented assumptions, not independently measured daily emissions. Treat outputs as prototype estimates, not audited actual emissions.
        - For dates after the supplied annual anchors, electricity is extrapolated using the configured 2.2% annual growth assumption, while solar and grid factor are carried forward. The campus model also carries forward selected daily source priors / monthly LPG patterns.
        """
    )
