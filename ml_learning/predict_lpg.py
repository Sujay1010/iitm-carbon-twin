import joblib
import pandas as pd


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/lpg_xgboost_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# 2. INPUT FOR FUTURE PREDICTION
# ==========================================

mess = "Mess_A"

month = 9
year = 2026

previous_lpg = 259

# 0 = normal operation
# 1 = vacation
vacation_flag = 0


# ==========================================
# 3. CREATE INPUT DATA
# ==========================================

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


# ==========================================
# 4. MAKE PREDICTION
# ==========================================

prediction = model.predict(input_data)

predicted_lpg = prediction[0]


# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("\n===================================")
print("LPG PREDICTION")
print("===================================")

print("Mess:", mess)
print("Month:", month, year)
print("Previous LPG:", previous_lpg)
print("Vacation:", vacation_flag)

print(
    "Predicted LPG consumption:",
    round(predicted_lpg, 2),
    "cylinders"
)