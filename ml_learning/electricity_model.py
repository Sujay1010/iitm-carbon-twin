
import os
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Original IIT Madras electricity data
actual_data = pd.DataFrame({
    "Year_Index": [1, 2, 3, 4],
    "Academic_Year": [
        "2022-23", "2023-24", "2024-25", "2025-26"
    ],
    "Total_Gross_GWh": [49.16, 52.15, 58.21, 59.49],
    "Data_Type": ["Actual"] * 4
})

# Generate 20 synthetic estimates from the original trend
trend_model = LinearRegression()

trend_model.fit(
    actual_data[["Year_Index"]],
    actual_data["Total_Gross_GWh"]
)

future_indices = list(range(5, 25))

future_values = trend_model.predict(
    pd.DataFrame({"Year_Index": future_indices})
)

synthetic_data = pd.DataFrame({
    "Year_Index": future_indices,
    "Academic_Year": [
        f"{2021 + i}-{str(2022 + i)[-2:]}"
        for i in future_indices
    ],
    "Total_Gross_GWh": future_values.round(2),
    "Data_Type": ["Synthetic estimate"] * 20
})

# Combine original and synthetic data
df = pd.concat(
    [actual_data, synthetic_data],
    ignore_index=True
)

# Save all 24 rows for inspection
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df.to_csv(
    "data/electricity_training_data_24_points.csv",
    index=False
)

# Train XGBoost
X = df[["Year_Index"]]
y = df["Total_Gross_GWh"]

model = XGBRegressor(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X, y)

# Predict 2026-27
prediction = model.predict(
    pd.DataFrame({"Year_Index": [5]})
)[0]

print("Total training rows:", len(df))
print("Actual observations: 4")
print("Synthetic estimates: 20")
print(f"2026-27 forecast: {prediction:.2f} GWh")

# Save trained model using the existing filename
joblib.dump(
    model,
    "models/electricity_xgboost_model.pkl"
)

print("Trained model saved successfully!")
