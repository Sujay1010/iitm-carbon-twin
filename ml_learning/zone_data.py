import pandas as pd

data = {
    "Academic_Year": [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"
    ],

    "Academic": [
        30.01,
        32.26,
        36.79,
        37.50
    ],

    "Hostel": [
        10.17,
        10.77,
        11.32,
        11.49
    ],

    "Residential": [
        2.48,
        2.49,
        2.57,
        2.62
    ],

    "Common_Utility": [
        6.50,
        6.63,
        7.53,
        7.88
    ]
}

df = pd.DataFrame(data)

print("===================================")
print("ZONE ELECTRICITY DATA")
print("===================================")

print(df)

print("\nTotal by year:")

df["Total"] = (
    df["Academic"]
    + df["Hostel"]
    + df["Residential"]
    + df["Common_Utility"]
)

print(df[["Academic_Year", "Total"]])