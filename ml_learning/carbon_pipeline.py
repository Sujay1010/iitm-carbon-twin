import joblib
import pandas as pd

# ==========================================
# 1. LOAD ELECTRICITY MODEL
# ==========================================

electricity_model = joblib.load(
    "models/electricity_xgboost_model.pkl"
)

next_year = pd.DataFrame({
    "Year_Index": [5]
})

predicted_electricity_gwh = electricity_model.predict(
    next_year
)[0]


# ==========================================
# 2. ELECTRICITY → CO2
# ==========================================

electricity_emission_factor = 0.76  # kg CO2/kWh

electricity_kwh = predicted_electricity_gwh * 1_000_000

electricity_co2_kg = (
    electricity_kwh * electricity_emission_factor
)

electricity_co2_tonnes = electricity_co2_kg / 1000


# ==========================================
# 3. LOAD LPG MODEL
# ==========================================

lpg_model = joblib.load(
    "models/lpg_xgboost_model.pkl"
)

mess = "Mess_A"
month = 9
year = 2026
previous_lpg = 259
vacation_flag = 0


# Prepare LPG input
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


# ==========================================
# 4. LPG PREDICTION
# ==========================================

predicted_lpg = lpg_model.predict(
    lpg_input
)[0]


# ==========================================
# 5. LPG → CO2
# ==========================================

lpg_per_cylinder_kg = 14.2
lpg_emission_factor = 3.0

lpg_kg = predicted_lpg * lpg_per_cylinder_kg

lpg_co2_kg = lpg_kg * lpg_emission_factor

lpg_co2_tonnes = lpg_co2_kg / 1000


# ==========================================
# 6. TOTAL CARBON FOOTPRINT
# ==========================================

total_co2_tonnes = (
    electricity_co2_tonnes +
    lpg_co2_tonnes
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("CARBON DIGITAL TWIN")
print("===================================")

print("\n--- ELECTRICITY ---")
print(
    "Predicted electricity:",
    round(predicted_electricity_gwh, 2),
    "GWh"
)

print(
    "Electricity CO2:",
    round(electricity_co2_tonnes, 2),
    "tonnes"
)


print("\n--- LPG ---")
print(
    "Mess:",
    mess
)

print(
    "Predicted LPG:",
    round(predicted_lpg, 2),
    "cylinders"
)

print(
    "LPG CO2:",
    round(lpg_co2_tonnes, 2),
    "tonnes"
)


print("\n===================================")
print("TOTAL CARBON FOOTPRINT")
print("===================================")

print(
    "Total CO2:",
    round(total_co2_tonnes, 2),
    "tonnes"
)