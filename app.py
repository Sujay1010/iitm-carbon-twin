import json
import os

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "building_energy.csv")
MESS_CSV_PATH = os.path.join(DATA_DIR, "mess_co2e.csv")
ZONE_ENERGY_CSV_PATH = os.path.join(DATA_DIR, "zone_energy.csv")
GREEN_CSV_PATH = os.path.join(DATA_DIR, "green_cover_summary.csv")

REAL_ZONES_PATH = os.path.join(DATA_DIR, "real_zones.geojson")
REAL_GREENZONE_PATH = os.path.join(DATA_DIR, "real_greenzone.geojson")
REAL_MESSZONE_PATH = os.path.join(DATA_DIR, "real_messzone.geojson")
REAL_SPORTS_PATH = os.path.join(DATA_DIR, "real_sports.geojson")

# ---- palette: the exact two colors from the reference type specimen — a near-black forest green and a
# bright lime — used as a strict duotone. Everything else on the page is a shade mixed from these two
# (a green->lime ramp for zone identity) rather than a new hue; the one exception is that "emission /
# attention" elements (mess gas, a net-emitting position) get an INVERTED lime block instead of a new
# color, since we can't add red/orange without breaking the two-color brief. ----
DARK = "#041f0b"      # exact sampled background from the reference
DARK_2 = "#0c2c16"    # a touch lighter — card/section surfaces sitting on the DARK page
DARK_3 = "#123a1e"    # borders / hover states
LIME = "#d5f379"      # exact sampled accent from the reference — headline, primary CTA, "attention"
LIME_SOFT = "#eef8cf"  # near-white lime tint, used sparingly for very light text on dark

# a ramp mixed purely between DARK and LIME — used for zone identity so every color on the map is
# still "a shade of these two", nothing pulls in a new hue family.
RAMP = ["#4d6932", "#5f7f3e", "#7a9a4c", "#9dbf5e", "#bfe06d"]
C_ACADEMIC = RAMP[0]
C_HOSTEL = RAMP[1]
C_RESIDENTIAL = RAMP[2]
C_RECREATIONAL = RAMP[3]
FOREST = RAMP[4]      # green-cover specific — the "most lush" step, just short of pure LIME
C_GREY = "#3f4a3f"     # muted dark green-grey for "no data / unassigned" (still in the family)

MESS_COLOR = LIME      # gas emissions get the pure accent color so they read as the thing to notice

ZONE_COLORS = {
    "Academic Zone": C_ACADEMIC,
    "Hostel Zone": C_HOSTEL,
    "Residential Zone (Staff Quarters)": C_RESIDENTIAL,
    "Recreational / Sports Zone": C_RECREATIONAL,
    "Green / Open Zone": FOREST,
    "Unassigned": C_GREY,
}
REAL_ZONE_COLORS = {
    "Academic Zone": C_ACADEMIC,
    "Hostel Zone": C_HOSTEL,
    "Residential Zone": C_RESIDENTIAL,
    "Recreational Zone": C_RECREATIONAL,
}
# the sidebar Zone filter uses the teammate's original naming (from buildings.geojson); your real
# digitized layers use shorter names for the same zones — this maps one to the other so the filter
# can narrow the real layers too, not just the traced OSM buildings.
ZONE_FILTER_TO_REAL = {
    "Academic Zone": "Academic Zone",
    "Hostel Zone": "Hostel Zone",
    "Residential Zone (Staff Quarters)": "Residential Zone",
    "Recreational / Sports Zone": "Recreational Zone",
    "Unassigned": None,
}


def contrast_text(hex_color):
    """Pick DARK or LIME_SOFT text depending on the luminance of a ramp color, so header bars / chips
    stay legible whichever step of the duotone ramp they're colored with."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return DARK if lum > 0.42 else LIME_SOFT


st.set_page_config(page_title="Campus Digital Twin for Carbon Footprint Estimation", layout="wide", page_icon="\U0001F343")

# ---------------- Global styling ----------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif; font-weight: 400; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background-color: {DARK}; }}
    [data-testid="stHeader"] {{ background-color: transparent; }}
    [data-testid="stSidebar"] {{ background-color: {DARK_2}; border-right: 1px solid {DARK_3}; }}
    .block-container {{ padding-top: 2rem; max-width: 1280px; }}
    p, span, label, div {{ color: {LIME_SOFT}; }}

    h1 {{
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
        font-size: 2.6rem !important; letter-spacing: -0.5px; color: {LIME} !important;
    }}
    h3 {{ font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important; color: {LIME} !important; }}
    .title-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 10px; }}
    .title-row svg {{ flex-shrink: 0; }}

    .stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 24px 0; }}
    .stat-card {{
        flex: 1; min-width: 220px; border-radius: 10px; padding: 18px 20px 16px 20px;
        background: {DARK_2};
        border: 1px solid {DARK_3};
        border-top: 3px solid var(--accent, {LIME});
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    .stat-card.invert {{ background: {LIME}; border-top-color: {DARK}; }}
    .stat-card .stat-label {{ font-size: 0.74rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; color: {LIME}; opacity: 0.75; margin-bottom: 7px; }}
    .stat-card.invert .stat-label {{ color: {DARK}; opacity: 0.7; }}
    .stat-card .stat-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 600; line-height: 1.05; color: var(--accent, {LIME}); }}
    .stat-card.invert .stat-value {{ color: {DARK}; }}
    .stat-card .stat-sub {{ font-size: 0.76rem; opacity: 0.6; margin-top: 7px; line-height: 1.35; color: {LIME_SOFT}; }}
    .stat-card.invert .stat-sub {{ color: {DARK}; opacity: 0.75; }}

    .section-header {{ border-left: 3px solid {LIME}; padding-left: 12px; margin: 14px 0 8px 0; }}
    .section-header h3 {{ margin: 0; font-size: 1.4rem; }}

    .legend-chip {{ display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:999px;
        font-size:0.76rem; font-weight:500; margin:2px 6px 2px 0; background:{DARK_2};
        border:1px solid {DARK_3}; color:{LIME_SOFT}; }}
    .legend-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}

    .stButton > button[kind="primary"] {{
        background-color: {LIME} !important; border-color: {LIME} !important; color: {DARK} !important; font-weight: 600;
    }}
    .stButton > button[kind="primary"]:hover {{ background-color: {LIME_SOFT} !important; border-color: {LIME_SOFT} !important; }}
    a, .stMarkdown a {{ color: {LIME} !important; }}
    [data-testid="stMetricValue"] {{ color: {LIME}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_card(label, value, sub, accent, invert=False):
    cls = "stat-card invert" if invert else "stat-card"
    st.markdown(
        f"""
        <div class="{cls}" style="--accent:{accent};">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(f'<div class="section-header"><h3>{text}</h3></div>', unsafe_allow_html=True)


def legend_row(items):
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-dot" style="background:{color};"></span>{label}</span>'
        for label, color in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def styled_popup(title, accent, rows, note=None):
    """rows: list of (label, value, is_primary) tuples. A dark, duotone-matched popup — header bar in the
    zone/category's ramp color with auto-picked contrast text, dark body, lime numbers."""
    header_text = contrast_text(accent)
    rows_html = ""
    for label, value, is_primary in rows:
        weight = "600" if is_primary else "400"
        size = "16px" if is_primary else "13px"
        color = LIME if is_primary else LIME_SOFT
        opac = "1" if is_primary else "0.65"
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;gap:14px;padding:3px 0;">'
            f'<span style="opacity:{opac};font-size:13px;color:{LIME_SOFT};">{label}</span>'
            f'<span style="font-weight:{weight};font-size:{size};color:{color};">{value}</span></div>'
        )
    note_html = (
        f'<div style="font-size:11px;opacity:0.55;margin-top:6px;border-top:1px solid {DARK_3};padding-top:6px;color:{LIME_SOFT};">{note}</div>'
        if note else ""
    )
    return f"""
    <div style="font-family:'Space Grotesk',system-ui,sans-serif;min-width:220px;border-radius:10px;
                overflow:hidden;box-shadow:0 4px 18px rgba(0,0,0,0.4);border:1px solid {DARK_3};">
      <div style="background:{accent};color:{header_text};padding:9px 14px;font-weight:600;font-size:14px;">
        {title}
      </div>
      <div style="background:{DARK_2};padding:10px 14px 12px 14px;">
        {rows_html}
        {note_html}
      </div>
    </div>
    """


@st.cache_data
def load_geo():
    buildings = gpd.read_file(os.path.join(DATA_DIR, "buildings.geojson"))
    zones = gpd.read_file(os.path.join(DATA_DIR, "zones.geojson"))
    canopy = gpd.read_file(os.path.join(DATA_DIR, "canopy.geojson"))
    greencover = gpd.read_file(os.path.join(DATA_DIR, "greencover.geojson"))
    grounds = gpd.read_file(os.path.join(DATA_DIR, "grounds.geojson"))
    water = gpd.read_file(os.path.join(DATA_DIR, "water.geojson"))
    return buildings, zones, canopy, greencover, grounds, water


@st.cache_data
def load_real_layers():
    real_zones = gpd.read_file(REAL_ZONES_PATH) if os.path.exists(REAL_ZONES_PATH) else None
    real_greenzone = gpd.read_file(REAL_GREENZONE_PATH) if os.path.exists(REAL_GREENZONE_PATH) else None
    real_messzone = gpd.read_file(REAL_MESSZONE_PATH) if os.path.exists(REAL_MESSZONE_PATH) else None
    real_sports = gpd.read_file(REAL_SPORTS_PATH) if os.path.exists(REAL_SPORTS_PATH) else None
    return real_zones, real_greenzone, real_messzone, real_sports


EMPTY_ENERGY_COLUMNS = [
    "building_id", "name", "zone_name", "area_m2",
    "electricity_kwh_month", "emission_factor_kgco2_per_kwh", "notes",
]


def load_energy_csv():
    df = None
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    if df is None:
        df = pd.DataFrame(columns=EMPTY_ENERGY_COLUMNS)
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("").astype(str)
    else:
        df["notes"] = ""
    return df


@st.cache_data
def load_mess_csv():
    return pd.read_csv(MESS_CSV_PATH)


@st.cache_data
def load_zone_energy_csv():
    return pd.read_csv(ZONE_ENERGY_CSV_PATH)


@st.cache_data
def load_green_csv():
    return pd.read_csv(GREEN_CSV_PATH)


buildings, zones, canopy, greencover, grounds, water = load_geo()
real_zones, real_greenzone, real_messzone, real_sports = load_real_layers()
energy_df = load_energy_csv()
mess_df = load_mess_csv()
zone_energy_df = load_zone_energy_csv()
green_df = load_green_csv()

FOOTPRINT_SVG = f"""
<svg width="34" height="44" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
  <path d="M50,42 C30,42 15,64 15,95 C15,122 30,140 50,140
           C70,140 85,122 85,95 C85,64 70,42 50,42 Z" fill="{LIME}"/>
  <circle cx="24" cy="26" r="9.5" fill="{LIME}"/>
  <circle cx="41" cy="15" r="7.5" fill="{LIME}"/>
  <circle cx="58" cy="13" r="6.8" fill="{LIME}"/>
  <circle cx="73" cy="17" r="6" fill="{LIME}"/>
  <circle cx="85" cy="26" r="5" fill="{LIME}"/>
</svg>
"""
st.markdown(
    f'<div class="title-row">{FOOTPRINT_SVG}<h1>Campus Digital Twin for Carbon Footprint Estimation</h1></div>',
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
st.sidebar.page_link("app.py", label="Campus map & carbon dashboard")
st.sidebar.page_link("pages/2_ML_Forecast.py", label="ML forecast & scenarios")
st.sidebar.markdown("---")
st.sidebar.header("Filters")
zone_options = ["All"] + sorted(buildings["zone_name"].unique().tolist())
selected_zone = st.sidebar.selectbox("Zone", zone_options)

show_layers = st.sidebar.multiselect(
    "Extra layers",
    ["Zone boundaries", "Tree canopy", "Green cover", "Grounds/pitches", "Water bodies", "Mess zones (gas CO2e)"],
    default=["Zone boundaries", "Mess zones (gas CO2e)"],
)

st.sidebar.markdown("---")
use_real_layers = st.sidebar.checkbox(
    "Show my real digitized layers (QGIS export)", value=True,
    help="Your own hand-digitized zone, mess, and green-cover polygons with real numbers baked in.",
)

st.sidebar.markdown("---")
st.sidebar.header("Emission factor default")
default_ef = st.sidebar.number_input(
    "Grid emission factor (kgCO2e/kWh)", min_value=0.0, max_value=2.0, value=0.82, step=0.01,
    help="Applied to any building where you haven't set a custom factor in the advanced table.",
)

st.sidebar.markdown("---")
mess_basis = st.sidebar.radio(
    "Mess gas CO2e basis",
    ["Combustion (burning the LPG)", "Manufacturing (cylinder cradle-to-gate)"],
    help=(
        "Combustion = IPCC 2006, 2.99 kg CO2e/kg LPG. Manufacturing = cylinder embodied "
        "carbon (Vijayakumar & Ganeshkumar, 2026). Combustion is the primary basis."
    ),
)
mess_co2_col = (
    "annual_co2e_tonnes_combustion" if mess_basis.startswith("Combustion")
    else "annual_co2e_tonnes_manufacturing"
)

# ---------------- Top stat row: Net position + three scopes, all in tCO2e/yr ----------------
zone_energy_total_t = zone_energy_df["annual_co2e_tonnes"].sum()
mess_total_t = mess_df[mess_co2_col].sum()
green_total_t = green_df["co2e_tonnes"].sum()
total_emissions_t = zone_energy_total_t + mess_total_t
net_t = total_emissions_t - green_total_t
net_is_positive = net_t > 0  # emitting more than the green-cover stock offsets

st.markdown('<div class="stat-row">', unsafe_allow_html=True)
n1, n2, n3, n4 = st.columns(4)
with n1:
    stat_card(
        "Net position",
        f"{'+' if net_is_positive else ''}{net_t:,.0f} t",
        f"Emissions ({total_emissions_t:,.0f} t/yr) vs. green stock ({green_total_t:,.0f} t) — "
        f"{'net emitting' if net_is_positive else 'stock exceeds annual emissions'}, illustrative only.",
        LIME,
        invert=net_is_positive,
    )
with n2:
    stat_card(
        "Scope 2 · Zone electricity",
        f"{zone_energy_total_t:,.0f} t/yr",
        "Academic + Hostel + Residential + Recreational, 2025-26.",
        LIME,
    )
with n3:
    stat_card(
        f"Scope 1 · Mess gas ({mess_basis.split(' ')[0]})",
        f"{mess_total_t:,.0f} t/yr",
        "Himalaya + Nilgiri + Vindhya, Sep-25 to Aug-26.",
        LIME,
        invert=True,
    )
with n4:
    stat_card(
        "Green cover carbon stock",
        f"{green_total_t:,.0f} t",
        "Standing stock across Zones A–E (not an annual rate).",
        FOREST,
    )
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Map ----------------
section_header("Campus map")
legend_row([
    ("Academic", REAL_ZONE_COLORS["Academic Zone"]), ("Hostel", REAL_ZONE_COLORS["Hostel Zone"]),
    ("Residential", REAL_ZONE_COLORS["Residential Zone"]), ("Recreational", REAL_ZONE_COLORS["Recreational Zone"]),
    ("Green cover", FOREST), ("Mess (gas)", MESS_COLOR),
])
filter_note = f" · filtered to **{selected_zone}**" if selected_zone != "All" else ""
st.caption(f"Click any shape for details · hover for a quick preview{filter_note}")

center = [buildings.geometry.centroid.y.mean(), buildings.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=16, tiles=None)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google Satellite", name="Satellite",
).add_to(m)
folium.TileLayer("OpenStreetMap", name="OSM Standard").add_to(m)

zone_energy_lookup = zone_energy_df.set_index("zone_name")[["gross_electricity_gwh_yr", "annual_co2e_tonnes"]].to_dict("index")

if "Zone boundaries" in show_layers:
    for _, row in zones.iterrows():
        zname = row["zone_name"]
        if selected_zone != "All" and zname != selected_zone:
            continue
        zdata = zone_energy_lookup.get(zname)
        accent = ZONE_COLORS.get(zname, C_GREY)
        if zdata:
            popup_html = styled_popup(zname, accent, [
                ("Annual CO2e", f"{zdata['annual_co2e_tonnes']:,.0f} t/yr", True),
                ("Electricity", f"{zdata['gross_electricity_gwh_yr']:.2f} GWh/yr", False),
            ], note="Scope 2 · 2025-26")
        else:
            popup_html = styled_popup(zname, accent, [("Data", "not yet available", False)])
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, c=accent: {"color": c, "weight": 2, "fillOpacity": 0.06, "dashArray": "4 4"},
            tooltip=zname,
            popup=folium.Popup(popup_html, max_width=260),
            control=False,
        ).add_to(m)

layer_defs = [
    ("Tree canopy", canopy, RAMP[0], 0.35),
    ("Green cover", greencover, RAMP[2], 0.35),
    ("Grounds/pitches", grounds, RAMP[3], 0.4),
    ("Water bodies", water, LIME_SOFT, 0.35),
]
for label, gdf, color, opacity in layer_defs:
    if label in show_layers:
        folium.GeoJson(
            json.loads(gdf.to_json()),
            style_function=lambda x, c=color, o=opacity: {"color": c, "weight": 1, "fillColor": c, "fillOpacity": o},
            control=False,
        ).add_to(m)

if use_real_layers:
    if real_zones is not None:
        for _, row in real_zones.iterrows():
            zname = row["zone_name"]
            if selected_zone != "All" and ZONE_FILTER_TO_REAL.get(selected_zone) != zname:
                continue
            accent = REAL_ZONE_COLORS.get(zname, C_GREY)
            note = "2 hostel blocks share this total · Scope 2 · 2025-26" if zname == "Hostel Zone" else "Scope 2 · 2025-26"
            popup_html = styled_popup(f"{zname} (real)", accent, [
                ("Annual CO2e", f"{row['co2e_t']:,.0f} t/yr", True),
                ("Electricity", f"{row['gwh_yr']:.2f} GWh/yr", False),
            ], note=note)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=accent: {"color": c, "weight": 4, "fillColor": c, "fillOpacity": 0.22},
                tooltip=zname,
                popup=folium.Popup(popup_html, max_width=260),
                control=False,
            ).add_to(m)

    if real_greenzone is not None:
        for _, row in real_greenzone.iterrows():
            popup_html = styled_popup(f"Green Zone {row['zone_label']}", FOREST, [
                ("Carbon stock", f"{row['co2e_tonnes']:,.0f} t CO2e", True),
                ("Carbon (tC)", f"{row['carbon_stock_tc']:,.0f} tC", False),
                ("Area", f"{row['hectares']:.2f} ha", False),
            ], note="Standing stock, not an annual rate")
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {"color": DARK, "weight": 3, "fillColor": FOREST, "fillOpacity": 0.42},
                tooltip=f"Green Zone {row['zone_label']}",
                popup=folium.Popup(popup_html, max_width=260),
                control=False,
            ).add_to(m)

    if real_messzone is not None and "Mess zones (gas CO2e)" in show_layers:
        for _, row in real_messzone.iterrows():
            if row["has_data"]:
                popup_html = styled_popup(row["name"], MESS_COLOR, [
                    ("CO2e (combustion)", f"{row['annual_co2e_tonnes_combustion']:,.0f} t/yr", True),
                    ("CO2e (manufacturing)", f"{row['annual_co2e_tonnes_manufacturing']:,.0f} t/yr", False),
                    ("Cylinders/yr", f"{row['cylinders_per_yr']:,.0f}", False),
                ], note="Scope 1 · real digitized shape")
                color, weight, fill_opacity = MESS_COLOR, 4, 0.85
            else:
                popup_html = styled_popup(row["name"], C_GREY, [("Status", "no data collected yet", True)])
                color, weight, fill_opacity = C_GREY, 2, 0.3
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color, w=weight, o=fill_opacity: {"color": c, "weight": w, "fillColor": c, "fillOpacity": o},
                tooltip=row["name"],
                popup=folium.Popup(popup_html, max_width=260),
                control=False,
            ).add_to(m)

    if real_sports is not None and selected_zone in ("All", "Recreational / Sports Zone"):
        for _, row in real_sports.iterrows():
            popup_html = styled_popup("Recreational Zone (real)", REAL_ZONE_COLORS["Recreational Zone"], [
                ("Annual CO2e", f"{row.get('co2e_t', 0):,.0f} t/yr", True),
                ("Electricity", f"{row.get('gwh_yr', 0):.2f} GWh/yr", False),
            ], note="2 grounds share this total · Scope 2 · 2025-26")
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "color": REAL_ZONE_COLORS["Recreational Zone"], "weight": 4,
                    "fillColor": REAL_ZONE_COLORS["Recreational Zone"], "fillOpacity": 0.22,
                },
                tooltip="Recreational Zone",
                popup=folium.Popup(popup_html, max_width=260),
                control=False,
            ).add_to(m)

ef_all = energy_df["emission_factor_kgco2_per_kwh"].fillna(default_ef)
ef_all = ef_all.where(ef_all > 0, default_ef)
merged = energy_df.copy()
merged["annual_co2e_kg"] = merged["electricity_kwh_month"].fillna(0) * 12 * ef_all

map_buildings = buildings if selected_zone == "All" else buildings[buildings["zone_name"] == selected_zone]
co2_lookup = merged.set_index("building_id")["annual_co2e_kg"].to_dict()
mess_lookup = mess_df.set_index("building_id").to_dict("index")

for _, row in map_buildings.iterrows():
    bid = row["building_id"]
    is_mess = bid in mess_lookup and "Mess zones (gas CO2e)" in show_layers

    if is_mess:
        mrow = mess_lookup[bid]
        popup_html = styled_popup(f"{row['name']} ({mrow['complex']})", MESS_COLOR, [
            ("CO2e (combustion)", f"{mrow['annual_co2e_tonnes_combustion']:,.0f} t/yr", True),
            ("CO2e (manufacturing)", f"{mrow['annual_co2e_tonnes_manufacturing']:,.0f} t/yr", False),
            ("Cylinders/yr", f"{mrow['cylinders_per_yr']:,}", False),
        ], note="Scope 1")
        color, weight, fill_opacity = MESS_COLOR, 3, 0.8
        tooltip = row["name"]
    else:
        co2 = (co2_lookup.get(bid, 0) or 0) / 1000
        accent = ZONE_COLORS.get(row["zone_name"], C_GREY)
        popup_html = styled_popup(row["name"], accent, [
            ("Annual CO2e", f"{co2:,.2f} t/yr", True),
            ("Zone", row["zone_name"], False),
            ("Area", f"{row['area_m2']:.0f} m²", False),
        ])
        color, weight, fill_opacity = accent, 1, 0.6
        tooltip = row["name"]

    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, c=color, w=weight, o=fill_opacity: {"color": c, "weight": w, "fillColor": c, "fillOpacity": o},
        tooltip=tooltip,
        popup=folium.Popup(popup_html, max_width=260),
        control=False,
    ).add_to(m)

# only the two basemaps show up in the layer switcher now — every overlay above is control=False,
# and toggling overlays already happens via the sidebar, so this just swaps Satellite/OSM.
folium.LayerControl(collapsed=True).add_to(m)
st_folium(m, width=None, height=650, returned_objects=[])

# ---------------- Charts & tables (unified units: t CO2e/yr, colored by category) ----------------
st.markdown("---")
section_header("Zone electricity — Scope 2 CO2e (2025-26)")

zone_chart_df = zone_energy_df.copy()
zone_chart_df["zone_short"] = zone_chart_df["zone_name"].str.replace(r"\s*\(.*\)", "", regex=True)
domain = zone_chart_df["zone_name"].tolist()
range_ = [ZONE_COLORS.get(z, C_GREY) for z in domain]
chart = (
    alt.Chart(zone_chart_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=46)
    .encode(
        x=alt.X("zone_short:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("annual_co2e_tonnes:Q", title="t CO2e / yr"),
        color=alt.Color("zone_name:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
        tooltip=[
            alt.Tooltip("zone_name:N", title="Zone"),
            alt.Tooltip("annual_co2e_tonnes:Q", title="t CO2e/yr", format=",.0f"),
            alt.Tooltip("gross_electricity_gwh_yr:Q", title="GWh/yr", format=".2f"),
        ],
    )
    .properties(height=280, background=DARK_2)
    .configure_axis(
        labelFont="Space Grotesk", titleFont="Space Grotesk", labelColor=LIME_SOFT, titleColor=LIME,
        gridColor=DARK_3, domainColor=DARK_3, labelFontSize=12, titleFontSize=12,
    )
    .configure_view(strokeWidth=0)
)
st.altair_chart(chart, use_container_width=True)

zone_row_colors = [ZONE_COLORS.get(z, C_GREY) for z in zone_energy_df["zone_name"]]
zone_table = zone_energy_df.rename(columns={
    "zone_name": "Zone", "gross_electricity_gwh_yr": "Electricity (GWh/yr)",
    "annual_co2e_tonnes": "CO2e (t/yr)", "year": "Year",
})[["Zone", "Electricity (GWh/yr)", "CO2e (t/yr)", "Year"]]
zone_table_styler = (
    zone_table.style
    .apply(lambda r: [f"background-color:{DARK_2}; color:{LIME_SOFT}; border-left: 4px solid {zone_row_colors[r.name]};"] * len(r), axis=1)
    .format({"Electricity (GWh/yr)": "{:.2f}", "CO2e (t/yr)": "{:,.0f}"})
)
st.dataframe(zone_table_styler, hide_index=True, use_container_width=True)

section_header("Mess complex gas — Scope 1 CO2e")
mess_table = mess_df[["complex", "cylinders_per_yr", "annual_co2e_tonnes_combustion", "annual_co2e_tonnes_manufacturing"]].rename(columns={
    "complex": "Mess complex", "cylinders_per_yr": "Cylinders/yr",
    "annual_co2e_tonnes_combustion": "CO2e combustion (t/yr)",
    "annual_co2e_tonnes_manufacturing": "CO2e manufacturing (t/yr)",
})
active_col = "CO2e combustion (t/yr)" if mess_basis.startswith("Combustion") else "CO2e manufacturing (t/yr)"

mess_styler = (
    mess_table.style
    .apply(lambda r: [f"background-color:{DARK_2}; color:{LIME_SOFT}; border-left: 4px solid {LIME};"] * len(r), axis=1)
    .set_properties(subset=[active_col], **{"font-weight": "600", "color": LIME})
    .format({"Cylinders/yr": "{:,.0f}", "CO2e combustion (t/yr)": "{:,.1f}", "CO2e manufacturing (t/yr)": "{:,.1f}"})
)
st.dataframe(mess_styler, hide_index=True, use_container_width=True)

section_header("Green cover carbon stock — Zones A–E")
green_table = green_df.rename(columns={
    "zone_label": "Zone", "hectares": "Hectares", "carbon_stock_tc": "Carbon stock (tC)", "co2e_tonnes": "CO2e (t)",
})
green_ranked = green_table["CO2e (t)"].rank(method="first").astype(int) - 1
green_shades = [DARK_2, "#173c1f", "#245229", "#3a6d33", FOREST]
green_colors = [green_shades[min(r, len(green_shades) - 1)] for r in green_ranked]
green_styler = (
    green_table.style
    .apply(lambda r: [f"background-color:{green_colors[r.name]}; color:{LIME_SOFT}; border-left: 4px solid {FOREST};"] * len(r), axis=1)
    .format({"Hectares": "{:,.2f}", "Carbon stock (tC)": "{:,.0f}", "CO2e (t)": "{:,.0f}"})
)
st.dataframe(green_styler, hide_index=True, use_container_width=True)

# ---------------- Methodology notes (kept out of the way, available on demand) ----------------
with st.expander("Notes & methodology"):
    st.markdown(
        f"""
        - **Net position** above is illustrative: it nets an annual emissions *flow* ({total_emissions_t:,.0f} t/yr)
          against a green-cover *stock* ({green_total_t:,.0f} t), which are different kinds of quantities — useful
          as a rough comparison, not a formal carbon-neutral claim.
        - Hostel Zone (2 blocks) and Recreational Zone (2 grounds) each report one shared zone-level total —
          their popups say so; they are not double-counted in any total above.
        - Mess buildings without gas-consumption data (Quark Canteen, Ishthara) are shown muted on the map with
          "no data yet," not hidden.
        - Building outlines and generic zone boundaries are traced from OpenStreetMap/manual digitizing and are
          approximate; {int((buildings['zone_name'] == 'Unassigned').sum())} buildings fall outside all zone
          polygons and appear muted/unassigned.
        - Thicker-outlined map shapes are the real, hand-digitized QGIS layers with computed numbers baked
          directly into each feature (toggle in the sidebar).
        - **Palette note**: every color on this page is either the exact dark green / lime pair or a shade
          mixed between them. Mess gas and a net-emitting position use an inverted lime block rather than a
          red, since the brief was a strict two-color palette; zones are told apart by a green→lime ramp
          plus their labels, not by hue alone.
        """
    )

# ---------------- Optional: building-level manual electricity tracker ----------------
with st.expander("Advanced: building-level manual electricity tracker (optional)"):
    st.caption("Separate from the real data above — a per-building editor, 0 kWh by default. Not used in any total on this page.")

    filtered_ids = (
        buildings["building_id"].tolist() if selected_zone == "All"
        else buildings.loc[buildings["zone_name"] == selected_zone, "building_id"].tolist()
    )
    table_view = energy_df[energy_df["building_id"].isin(filtered_ids)].copy()

    edited = st.data_editor(
        table_view,
        column_config={
            "building_id": st.column_config.TextColumn(disabled=True),
            "name": st.column_config.TextColumn(disabled=True),
            "zone_name": st.column_config.TextColumn(disabled=True),
            "area_m2": st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "electricity_kwh_month": st.column_config.NumberColumn("Electricity (kWh/month)", min_value=0.0, step=10.0),
            "emission_factor_kgco2_per_kwh": st.column_config.NumberColumn(
                "Emission factor override", min_value=0.0, step=0.01, help="Leave as-is to use the sidebar default.",
            ),
            "notes": st.column_config.TextColumn("Notes"),
        },
        hide_index=True, use_container_width=True, num_rows="fixed", key="editor",
    )

    if st.button("Save changes", type="primary"):
        energy_df_idx = energy_df.set_index("building_id")
        edited_idx = edited.set_index("building_id")
        energy_df_idx.update(edited_idx)
        energy_df_idx.reset_index().to_csv(CSV_PATH, index=False)
        st.success("Saved to building_energy.csv")

    ef_edit = edited["emission_factor_kgco2_per_kwh"].fillna(default_ef)
    ef_edit = ef_edit.where(ef_edit > 0, default_ef)
    edited_co2e = edited["electricity_kwh_month"].fillna(0) * 12 * ef_edit
    e1, e2 = st.columns(2)
    e1.metric("Tracked CO2e (this view)", f"{edited_co2e.sum()/1000:,.1f} t")
    e2.metric("Buildings with data entered", f"{(edited['electricity_kwh_month'] > 0).sum()} / {len(edited)}")
