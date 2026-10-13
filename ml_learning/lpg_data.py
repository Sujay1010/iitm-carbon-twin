import pandas as pd

# ==========================================
# LPG GAS CYLINDER CONSUMPTION DATA
# Sep 2025 - Aug 2026
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

# Create a DataFrame
df = pd.DataFrame(data)

# Display basic information
print("\nLPG DATA")
print("=" * 50)

print(df)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ==========================================
# RESHAPE DATA
# ==========================================

df_long = df.melt(
    id_vars=["Month"],
    var_name="Mess",
    value_name="LPG_Consumption"
)

print("\nRESHAPED DATA")
print("=" * 50)
print(df_long)

print("\nNew dataset shape:")
print(df_long.shape)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Convert Month into a real date
df_long["Date"] = pd.to_datetime(
    df_long["Month"],
    format="%b-%y"
)

# Extract month number
df_long["Month_Number"] = df_long["Date"].dt.month

# Extract year
df_long["Year"] = df_long["Date"].dt.year

# Identify vacation months
# According to our data note:
# 0 or less than 150 cylinders = vacation
df_long["Vacation_Flag"] = (
    df_long["LPG_Consumption"] < 150
).astype(int)

# Previous month's LPG consumption for the same mess
df_long["Previous_Month_LPG"] = (
    df_long
    .sort_values(["Mess", "Date"])
    .groupby("Mess")["LPG_Consumption"]
    .shift(1)
)

print("\nFEATURE ENGINEERED DATA")
print("=" * 60)
print(df_long.head(20))

# ==========================================
# REMOVE ROWS WITH NO PREVIOUS MONTH DATA
# ==========================================

df_long = df_long.dropna(subset=["Previous_Month_LPG"])

print("\nAFTER CLEANING")
print("=" * 50)
print(df_long.head(15))

print("\nDataset shape after cleaning:")
print(df_long.shape)

# ==========================================
# CREATE X AND y
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

# Convert Mess names into numerical columns
X = pd.get_dummies(
    X,
    columns=["Mess"],
    dtype=int
)

print("\nFEATURES (X)")
print("=" * 50)
print(X.head())

print("\nTARGET (y)")
print("=" * 50)
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN / TEST DATA")
print("=" * 50)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)