import joblib
import pandas as pd

# Load trained LPG model
model = joblib.load("models/lpg_xgboost_model.pkl")

# Input conditions
mess = "Mess_A"
month = 9
year = 2026
previous_lpg = 259
vacation_flag = 0

# Prepare input
input_data = pd.DataFrame({
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

# Predict LPG
predicted_lpg = model.predict(input_data)[0]

# LPG → CO2
lpg_per_cylinder_kg = 14.2
emission_factor = 3.0

lpg_kg = predicted_lpg * lpg_per_cylinder_kg
co2_kg = lpg_kg * emission_factor
co2_tonnes = co2_kg / 1000

# Display results
print("\n===================================")
print("CARBON DIGITAL TWIN - LPG")
print("===================================")

print("Mess:", mess)
print("Month:", month, year)

print("\n--- LPG PREDICTION ---")
print("Previous LPG:", previous_lpg, "cylinders")
print("Predicted LPG:", round(predicted_lpg, 2), "cylinders")

print("\n--- CARBON EMISSION ---")
print("LPG consumed:", round(lpg_kg, 2), "kg")
print("CO2 emission:", round(co2_kg, 2), "kg")
print("CO2 emission:", round(co2_tonnes, 2), "tonnes")