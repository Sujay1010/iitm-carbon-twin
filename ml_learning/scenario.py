import joblib
import pandas as pd

# ============================================================
# LOAD MODELS
# ============================================================

electricity_model = joblib.load(
    "models/electricity_xgboost_model.pkl"
)

lpg_model = joblib.load(
    "models/lpg_xgboost_model.pkl"
)


# ============================================================
# BASELINE ELECTRICITY FORECAST
# ============================================================

baseline_input = pd.DataFrame({
    "Year_Index": [5]
})

baseline_electricity = electricity_model.predict(
    baseline_input
)[0]


# ============================================================
# SCENARIO INPUTS
# ============================================================

electricity_change_percent = -10

mess = "Mess_A"
month = 9
year = 2026
previous_lpg = 259
vacation_flag = 0


# ============================================================
# ELECTRICITY SCENARIO
# ============================================================

scenario_electricity = (
    baseline_electricity
    * (1 + electricity_change_percent / 100)
)


# ============================================================
# ELECTRICITY CO2
# ============================================================

electricity_co2 = (
    scenario_electricity
    * 1_000_000
    * 0.76
    / 1000
)


# ============================================================
# LPG PREDICTION
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


# ============================================================
# LPG CO2
# ============================================================

lpg_co2 = (
    predicted_lpg
    * 14.2
    * 3.0
    / 1000
)


# ============================================================
# TOTAL SCENARIO CO2
# ============================================================

total_co2 = electricity_co2 + lpg_co2


# ============================================================
# BASELINE CO2
# ============================================================

baseline_electricity_co2 = (
    baseline_electricity
    * 1_000_000
    * 0.76
    / 1000
)

baseline_lpg_co2 = (
    predicted_lpg
    * 14.2
    * 3.0
    / 1000
)

baseline_total_co2 = (
    baseline_electricity_co2
    + baseline_lpg_co2
)


# ============================================================
# IMPACT
# ============================================================

co2_change = (
    total_co2 - baseline_total_co2
)

co2_change_percent = (
    co2_change / baseline_total_co2
) * 100


# ============================================================
# DISPLAY
# ============================================================

print("===================================")
print("CARBON DIGITAL TWIN")
print("SCENARIO EXPLORER")
print("===================================")

print("\nBASELINE")
print(
    "Electricity:",
    round(baseline_electricity, 2),
    "GWh"
)

print(
    "Total CO2:",
    round(baseline_total_co2, 2),
    "tonnes"
)

print("\nSCENARIO")

print(
    "Electricity change:",
    electricity_change_percent,
    "%"
)

print(
    "Scenario electricity:",
    round(scenario_electricity, 2),
    "GWh"
)

print(
    "Scenario CO2:",
    round(total_co2, 2),
    "tonnes"
)

print("\n===================================")
print("SCENARIO IMPACT")
print("===================================")

print(
    "CO2 change:",
    round(co2_change, 2),
    "tonnes"
)

print(
    "CO2 change:",
    round(co2_change_percent, 2),
    "%"
)
