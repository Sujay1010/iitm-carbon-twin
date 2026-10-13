# LPG to CO2 emission calculation

predicted_lpg = 268.94

# LPG emission factor
# Approximate value: 3.0 kg CO2 per kg LPG
lpg_per_cylinder_kg = 14.2
emission_factor = 3.0

lpg_kg = predicted_lpg * lpg_per_cylinder_kg
co2_emission_kg = lpg_kg * emission_factor

print("\n===================================")
print("LPG CARBON EMISSION")
print("===================================")

print("Predicted LPG consumption:", round(predicted_lpg, 2), "cylinders")
print("LPG consumed:", round(lpg_kg, 2), "kg")
print("CO2 emission:", round(co2_emission_kg, 2), "kg")
print("CO2 emission:", round(co2_emission_kg / 1000, 2), "tonnes")