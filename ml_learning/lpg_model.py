import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ==========================================
# 1. LOAD LPG DATA
# ==========================================

data = {
    "Month": [
        "Sep-25", "Oct-25", "Nov-25", "Dec-25",
        "Jan-26", "Feb-26", "Mar-26", "Apr-26",
        "May-26", "Jun-26", "Jul-26", "Aug-26"
    ],

    "Mess_A": [238, 228, 228, 233, 259, 215, 196, 225, 93, 0, 53, 259],
    "Mess_B": [202, 205, 146, 56, 245, 231, 205, 200, 171, 191, 216, 204],
    "Mess_C": [263, 249, 247, 222, 216, 248, 206, 219, 174, 181, 242, 303],
    "Mess_D": [251, 239, 186, 154, 177, 200, 180, 182, 160, 179, 206, 245],
    "Mess_E": [236, 244, 267, 319, 236, 238, 203, 210, 93, 0, 49, 290],
    "Mess_F": [267, 266, 204, 28, 206, 255, 249, 245, 131, 60, 89, 200],
    "Mess_G": [249, 232, 203, 81, 154, 266, 216, 219, 103, 0, 78, 294],
    "Mess_H": [216, 209, 163, 173, 196, 192, 191, 182, 143, 143, 150, 239],
    "Mess_I": [148, 154, 101, 27, 112, 113, 122, 118, 100, 131, 171, 211]
}


df = pd.DataFrame(data)


# ==========================================
# 2. RESHAPE
# ==========================================

df_long = df.melt(
    id_vars=["Month"],
    var_name="Mess",
    value_name="LPG_Consumption"
)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

df_long["Date"] = pd.to_datetime(
    df_long["Month"],
    format="%b-%y"
)

df_long["Month_Number"] = df_long["Date"].dt.month
df_long["Year"] = df_long["Date"].dt.year

# Vacation flag
df_long["Vacation_Flag"] = (
    df_long["LPG_Consumption"] < 150
).astype(int)

# Previous month consumption
df_long = df_long.sort_values(["Mess", "Date"])

df_long["Previous_Month_LPG"] = (
    df_long
    .groupby("Mess")["LPG_Consumption"]
    .shift(1)
)

# Remove first month because previous value doesn't exist
df_long = df_long.dropna(
    subset=["Previous_Month_LPG"]
)


# ==========================================
# 4. CREATE X AND y
# ==========================================

X = df_long[
    [
        "Month_Number",
        "Year",
        "Mess",
        "Previous_Month_LPG",
        "Vacation_Flag"
    ]
]

y = df_long["LPG_Consumption"]


# Convert Mess to numerical features
X = pd.get_dummies(
    X,
    columns=["Mess"],
    dtype=int
)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. CREATE XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    objective="reg:squarederror"
)


# ==========================================
# 7. TRAIN
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. PREDICT
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUATE
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==========================================")
print("LPG XGBOOST MODEL RESULTS")
print("==========================================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))


# ==========================================
# 10. SHOW ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual_LPG": y_test.values,
    "Predicted_LPG": np.round(y_pred, 2)
})

print("\nACTUAL VS PREDICTED")
print(results)

# ==========================================
# 11. SAVE MODEL
# ==========================================

import joblib

joblib.dump(model, "models/lpg_xgboost_model.pkl")

print("\nModel saved successfully!")