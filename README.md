# IITM Campus Carbon Digital Twin — Dashboard

## What's in here
- `app.py` — the Streamlit app
- `data/` — your traced layers, pre-processed:
  - `buildings.geojson` (287 buildings, each tagged with `zone_name` via spatial join)
  - `zones.geojson` (5 zones: Academic, Hostel, Residential/Staff Quarters, Recreational/Sports, Green/Open)
  - `canopy.geojson`, `greencover.geojson`, `grounds.geojson`, `water.geojson`
  - `building_energy.csv` — editable data (starts at 0 kWh for every building; this is what the dashboard reads/writes)
  - `mess_co2e.csv` — **real computed data**: Himalaya/Nilgiri/Vindhya mess complex gas CO2e (Scope 1), both combustion and cylinder-manufacturing bases, Sep-25 to Aug-26
  - `zone_energy.csv` — **real computed data**: Academic/Hostel/Residential/Recreational zone electricity and Scope 2 CO2e for 2025-26, reconstructed from the institute's reported Scope 2 anchor
  - `green_cover_summary.csv` — **real computed data**: green zones A-E carbon stock, per India State of Forest Report 2019 (Tamil Nadu) density (82.25 tC/ha)
- `requirements.txt` — was zipped as `appp.txt` by mistake; renamed back so `pip install -r requirements.txt` works
- `.streamlit/config.toml` — sets Streamlit's own dark theme to the exact palette below, so native widgets
  (buttons, dropdowns, dataframes, sidebar) match the custom-styled parts instead of clashing with them.
  Keep this folder when you unzip — Streamlit picks it up automatically from next to `app.py`.

## Palette & type (exact match to your reference)
- **Colors** — sampled directly from your reference image, used as a strict two-color palette:
  - Dark `#041f0b` (page background, card/section surfaces are a lighter step of this)
  - Lime `#d5f379` (headline, primary accent, buttons, "attention" elements)
  - Every other color on the page (zone shapes, chart bars, table rows) is a shade mixed between
    those two — a green→lime ramp — rather than a new hue. Mess gas and a "net emitting" position use
    an inverted lime block (lime background, dark text) instead of a red, since the brief ruled out
    adding a warning color outside the two-color palette.
- **Font** — Space Grotesk (a bold geometric grotesk, close to the reference's "Overused Grotesk" in
  spirit) throughout: headline, section headers, stat numbers, body text, sidebar. Loaded from Google
  Fonts, so it needs your browser online the first time it loads (it'll cache after that).

## What was added to the original build
The original `app.py` only had the editable, all-zero `building_energy.csv`. The three CSVs
above wire in the actual computed numbers from the QGIS/Excel workflow:
- **Campus totals** row (top of the page): Scope 1 mess gas, Scope 2 zone electricity, and green-cover
  carbon stock, shown as three separate figures (deliberately not summed — different units and scopes).
- **Zone electricity & Scope 2 CO2e** bar chart, using the reconstructed 2025-26 numbers.
- **Mess complex gas CO2e** table with a sidebar toggle between the combustion basis (primary,
  2.99 kg CO2e/kg LPG, IPCC 2006) and the cylinder-manufacturing basis (secondary, 44.38 kg CO2e/cylinder).
- **Green cover carbon stock** table (A-E). Not drawn as separate polygons on the map, because this
  build's `greencover.geojson` is untagged OSM grass/wood, not the same hand-digitized A-E zones —
  see Limitations.
- On the map: the three real mess buildings (already present in `buildings.geojson`, matched by name)
  are outlined in red with a thicker border and a popup showing their real cylinder count and CO2e,
  independent of the (currently unused) building-level electricity table. Zone boundary tooltips now
  show the real electricity/CO2e figures for the four matched zones.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. Use the two links at the top of the left sidebar to switch between
the **campus map & carbon dashboard** and the **ML forecast & scenarios** page.

## ML forecast page (integrated from the colleague's `carbon-digital-twin` project)
- `pages/2_ML_Forecast.py` — the colleague's Streamlit app, now a second page of this dashboard,
  restyled to the same dark-green / lime palette and Space Grotesk font. Its logic is unchanged:
  - **Electricity model** (`models/electricity_xgboost_model.pkl`) — XGBoost, predicts next-year gross
    campus electricity (GWh) from the 2022-23 → 2025-26 trend.
  - **LPG model** (`models/lpg_xgboost_model.pkl`) — XGBoost, predicts monthly LPG cylinders per mess
    from month, year, previous month's LPG and a vacation flag.
  - KPIs, trend charts, alerts, zone analysis and a reduction-scenario explorer built on those predictions.
- **Electricity model updated** (v2): retrained on 24 rows — the 4 real years (2022-23 → 2025-26)
  plus 20 synthetic years extrapolated from a linear trend
  (`ml_training/electricity_model.py`, data in `ml_training/electricity_training_data_24_points.csv`).
  Same input (`Year_Index`) and filename, so the page uses it unchanged.
- **10-year electricity outlook**: a sidebar slider (1–10 years) on the ML page forecasts
  2026-27 → 2035-36 with the XGBoost model, showing yearly GWh, CO2e, change vs 2025-26 and cumulative
  CO2e, plotted against the 4 actual years and the extrapolated training estimates
  (`data/electricity_training_data.csv`). Because the training set is 4 actual + 20 trend-extrapolated
  years (approved by the professor), the outlook follows the historical linear trend.
- `ml_training/` — the colleague's training / data-prep scripts, kept for reference and retraining.
  They are **not** run by the website. To retrain, run them from inside `ml_training/` (they import
  each other as `src.*`, so either rename the folder to `src` in a copy or adjust the imports) and
  copy the new `.pkl` files into `models/`.

**Aligned with the main dashboard** (changed from the colleague's original):
- Grid factor is no longer a hard-coded 0.76; it is computed from `data/zone_energy.csv`
  (total zone CO2e / total zone kWh ≈ 0.719 kg CO2e/kWh), so both pages use the same basis.
  Editing that CSV updates both pages.
- LPG uses **2.99 kg CO2e/kg** (IPCC 2006 combustion, 14.2 kg cylinder), same as the dashboard's mess table.
- The zone table/chart reads `data/zone_energy.csv` (Recreational 3.25 GWh), replacing the old
  "Common / Utility 7.88 GWh" placeholder.
- The carbon alert baseline is computed (2025-26 electricity + selected mess at last month's LPG)
  instead of a fixed 44,942.86 t.

**Mess mapping** (`MESS_LABELS` near the top of `pages/2_ML_Forecast.py`): Mess_A–E = Himalaya,
Mess_F–G = Nilgiri, Mess_H–I = Vindhya.

## Publishing online
See `DEPLOY.md` — GitHub repo + Streamlit Community Cloud (free, public link).

## How it works (camera-ready pass, v2)
- **Palette**: green is the site's brand color throughout (headers, borders, the two carbon-stock cards) —
  fitting for a carbon/green-cover project. Emissions (mess gas, a net-emitting position) use a muted,
  desaturated terracotta red, never a bright "warning" red. Zone colors are muted, nature-toned hues
  (slate blue, amber, plum, sage) chosen to stay distinguishable without fighting the green brand color.
  No emoji anywhere — color and typography carry the design instead.
- **Top of the page — 4 stat cards, all in the same unit (t CO2e/yr or t CO2e):**
  1. **Net position** — total emissions (Scope 1 + Scope 2) minus green-cover stock: muted red if
     net-emitting, brand green if the stock is larger. A rough, illustrative comparison (it nets a flow
     against a stock) — the one-line caveat lives in its card, not a separate disclaimer block.
  2. Scope 2 zone electricity · Scope 1 mess gas · Green cover stock.
- **Map**: every shape — buildings, generic zones, and your real digitized layers — has a colored popup
  (a header bar in that zone/category's color) on click, plus a quick hover tooltip. Nothing is a plain
  white Leaflet popup any more. A small color-key legend sits above the map.
- **Layer switcher decluttered**: the old build listed every single polygon as its own checkbox
  ("macro_element_div_2", "_3", …) in Leaflet's layer control — that's now gone. Only the two basemaps
  (Satellite / OSM Standard) appear in the switcher, which itself is collapsed to a small icon by default.
  Which overlays are drawn at all is controlled from the sidebar, same as before.
- **Charts/tables, all in t CO2e/yr:**
  - Zone electricity bar chart — Altair, bars colored to match the map's zone colors (not default blue).
  - Mess complex table — row-tinted by complex, with the active CO2e basis (combustion/manufacturing,
    per the sidebar toggle) bolded.
  - Green cover table — row-shaded on a green scale by CO2e magnitude.
- **Notes & methodology** — the stock-vs-flow caveat, the hostel/recreational shared-total note, the
  unassigned-buildings count, etc. are all still here, just moved into a collapsed expander instead of
  sitting as captions on the main view — open it if a reviewer asks "isn't green cover a stock, not a rate?"
  or similar.
- The original per-building manual electricity editor (all-zero by default, not part of the real computed
  data) is likewise tucked into a collapsed **"🔧 Advanced" expander** at the very bottom. Its "Save changes"
  button still writes to `data/building_energy.csv`.
- Left sidebar: filter by zone, toggle extra layers, set the default grid emission factor, switch the mess
  CO2e basis.

## Real digitized layers (added on top of the teammate's original build)
Your own QGIS exports (`data/real_zones.geojson`, `data/real_greenzone.geojson`,
`data/real_messzone.geojson`) are drawn as thicker-outlined shapes on the map, on by
default via the "Show my real digitized layers" sidebar checkbox:
- **real_zones.geojson** — Academic, Hostel (2 blocks, same zone total — not double-counted),
  and Residential zone polygons, each with real `gwh_yr`/`co2e_t` (2025-26, Scope 2) baked
  directly into the shape's attributes.
- **real_greenzone.geojson** — your actual hand-digitized Green Zones A-E, replacing the
  table-only limitation from before: hectares, carbon stock (tC), and CO2e (t) per zone,
  drawn as real green polygons on the map, not just a table.
- **real_messzone.geojson** — all 5 traced mess buildings. The 3 real complexes (Himalaya,
  Nilgiri, Vindhya) are outlined in red with full combustion/manufacturing CO2e in their
  popups; the other 2 (Quark Canteen, Ishthara) are shown in grey with a "no data collected
  yet" popup rather than being hidden.
- **Recreational/Sports zone is now included** (`data/real_sports.geojson`) — re-exported
  with the correct CRS (EPSG:4326) and `gwh_yr`/`co2e_t` fields. It's 2 courts/grounds
  sharing one zone-level total (3.25 GWh/yr, 2,472 t CO2e/yr), same "don't double-count"
  pattern as the hostel blocks.

  **Note the number changed**: the real re-exported figure (3.25 GWh, 2,472 t) is lower
  than the placeholder used earlier (7.88 GWh, 5,646 t). That's expected — the placeholder
  was the broader "Common/Utility/Sports/Hospital" bucket from the original reconstruction
  (see limitation below), while this new figure is specific to the recreational/sports
  zone only, now that it's spatially defined. `zone_energy.csv` has been updated to this
  real figure, so the "Campus totals" metric and the zone bar chart both reflect it. Worth
  double-checking against your own source numbers before presenting, since it's a
  meaningfully different value from what was on the slide.

## Notes / limitations
- 4 buildings fell outside all drawn zone boundaries and show as "Unassigned" (grey) — worth checking against your zone sketch and either extending a zone or leaving them out of zone-level totals.
- Zone boundaries were hand-drawn and are approximate, matching the "not to scale" reference classification.
- Emission factor defaults to 0.82 kgCO2e/kWh (a general India grid average) — replace with whatever factor your course specifies once you have it. The zone-level Scope 2 numbers in `zone_energy.csv` do NOT use this default; they come from the separate campus-wide reconstruction anchored to the institute's own reported Scope 2 figure, and are independent of the building-level editable table.
- The green-cover carbon stock (82.25 tC/ha) is a standing **stock**, not an annual sequestration **rate** — don't read it as "tCO2e captured per year."
- The "Common/Utility/Sports/Hospital" electricity bucket from the source reconstruction was mapped onto "Recreational / Sports Zone" here as the closest match; it also covers utility and hospital load, so treat it as an approximation for that zone specifically.
- Green zones A-E don't have their own polygons in this build (only generic untagged OSM green-cover), so they show as a table only, not shaded on the map.
- Deploying online (e.g. Streamlit Community Cloud): its filesystem resets on restart, so "Save changes" won't persist between deploys/restarts there. Fine for local use or a live demo; for a persistent shared dataset you'd want something like a Google Sheet or small database instead of the CSV.
