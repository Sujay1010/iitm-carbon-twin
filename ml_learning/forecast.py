import pandas as pd
import joblib
import plotly.graph_objects as go

# Load electricity model
model = joblib.load(
    "models/electricity_xgboost_model.pkl"
)

# Historical data
historical = pd.DataFrame({
    "Year": [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"
    ],
    "Year_Index": [1, 2, 3, 4],
    "Electricity_GWh": [
        49.16,
        52.15,
        58.21,
        59.49
    ]
})

# Forecast next year
future = pd.DataFrame({
    "Year_Index": [5]
})

prediction = model.predict(future)[0]

print("===================================")
print("ELECTRICITY FORECAST")
print("===================================")

print("2025-26 actual:", 59.49, "GWh")
print(
    "2026-27 forecast:",
    round(prediction, 2),
    "GWh"
)

# Carbon calculation
co2_tonnes = prediction * 1_000_000 * 0.76 / 1000

print(
    "Forecast CO2:",
    round(co2_tonnes, 2),
    "tonnes"
)

# Create forecast chart
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=historical["Year"],
        y=historical["Electricity_GWh"],
        mode="lines+markers",
        name="Historical",
        line=dict(
            color="#63a7ff",
            width=2
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=["2025-26", "2026-27"],
        y=[59.49, prediction],
        mode="lines+markers",
        name="Forecast",
        line=dict(
            color="#35c7a0",
            width=2,
            dash="dash"
        )
    )
)

fig.update_layout(
    title="Campus Electricity Forecast",
    xaxis_title="Academic Year",
    yaxis_title="Electricity (GWh)",
    template="plotly_dark"
)

fig.show()