import pandas as pd

# IIT Madras campus electricity data
data = {
    "Academic_Year": [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"
    ],

    "Academic_Gross_GWh": [
        30.01,
        32.26,
        36.79,
        37.50
    ],

    "Hostel_Gross_GWh": [
        10.17,
        10.77,
        11.32,
        11.49
    ],

    "Residential_Gross_GWh": [
        2.48,
        2.49,
        2.57,
        2.62
    ],

    "Common_Gross_GWh": [
        6.50,
        6.63,
        7.53,
        7.88
    ],

    "Total_Gross_GWh": [
        49.16,
        52.15,
        58.21,
        59.49
    ]
}

df = pd.DataFrame(data)

print("===================================")
print("ELECTRICITY DATA")
print("===================================")

print(df)

print("\nDataset shape:", df.shape)