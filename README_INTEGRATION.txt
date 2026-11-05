IITM Carbon Twin Date + Zone XGBoost integration patch

Copy the folders/files in this patch into the ROOT of your cloned iitm-carbon-twin repository.
Allow the replacement of pages/2_ML_Forecast.py.

The patch replaces the existing ML_Forecast page in place (same route) and adds:
  src/features.py, src/__init__.py, src/web_inference.py
  config/calendar_rules.json
  data/Daily_CO2_Source_Dataset.csv
  data/annual_zone_anchors.csv
  models/production/xgboost_{academic,hostel,residential,common,campus_total}.joblib

No app.py edit is required because the existing app already links to pages/2_ML_Forecast.py.
No requirements.txt edit is required for this XGBoost-only integration: the website already lists pandas,
xgboost, joblib and Streamlit. PyTorch/LSTM dependencies and artifacts are intentionally not included.

The zone model predicts zone electricity Scope 2 only. The separate campus model predicts a campus-wide
selected-source net subtotal. Do not add the zone totals to the campus total; that would double-count electricity.

Monthly/yearly totals are calculated by making one prediction for every date in the period and summing those
daily predictions. All values are estimates against reconstructed targets, not measured daily emissions.

Test locally from the repo root:
  streamlit run app.py

Then open ML_Forecast, test one date, one month, one calendar year, and one financial year.
