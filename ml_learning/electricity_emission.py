# Electricity to CO2 emission calculation

predicted_electricity_gwh = 59.12

# IIT Madras grid emission factor
emission_factor = 0.76  # kg CO2 per kWh

# Convert GWh to kWh
electricity_kwh = predicted_electricity_gwh * 1_000_000

# Calculate CO2 emissions
co2_kg = electricity_kwh * emission_factor
co2_tonnes = co2_kg / 1000

print("===================================")
print("ELECTRICITY CARBON EMISSION")
print("===================================")

print(
    "Predicted electricity:",
    round(predicted_electricity_gwh, 2),
    "GWh"
)

print(
    "Electricity consumed:",
    round(electricity_kwh, 2),
    "kWh"
)

print(
    "CO2 emission:",
    round(co2_kg, 2),
    "kg"
)

print(
    "CO2 emission:",
    round(co2_tonnes, 2),
    "tonnes"
)