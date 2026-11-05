"""Calendar-driven feature and target preparation for the IITM zone model."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import math
from functools import lru_cache
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
SOURCE_FILE = DATA_DIR / "Daily_CO2_Source_Dataset.csv"
ANCHOR_FILE = DATA_DIR / "annual_zone_anchors.csv"

ZONE_COLUMNS = {
    "academic": ("Academic & Research", "academic_gross_kwh"),
    "hostel": ("Hostel & Student Life", "hostel_gross_kwh"),
    "residential": ("Residential", "residential_gross_kwh"),
    "common": ("Common / Utilities / Sports / Hospital", "common_gross_kwh"),
}
ZONE_ORDER = list(ZONE_COLUMNS)
FEATURE_COLUMNS = [
    "month_number", "day_of_month", "day_of_year", "day_of_week", "is_weekend",
    "summer_vacation_proxy", "winter_break_proxy", "school_activity_index",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "fiscal_year_offset", "annual_zone_gross_electricity_gwh_estimated",
    "annual_campus_gross_electricity_gwh_estimated",
    "annual_solar_generation_gwh_assumed", "grid_emission_factor_kgco2_per_kwh",
]
TARGET = "zone_scope2_co2e_tonnes_daily_reconstructed"
CAMPUS_TARGET = "net_selected_sources_after_tree_sink_co2e_tonnes_daily"
CAMPUS_FEATURE_COLUMNS = [
    "month_number", "day_of_month", "day_of_year", "day_of_week", "is_weekend",
    "summer_vacation_proxy", "winter_break_proxy", "school_activity_index",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos", "fiscal_year_offset",
    "annual_academic_gross_electricity_gwh_estimated",
    "annual_hostel_gross_electricity_gwh_estimated",
    "annual_residential_gross_electricity_gwh_estimated",
    "annual_common_gross_electricity_gwh_estimated",
    "annual_campus_gross_electricity_gwh_estimated",
    "annual_solar_generation_gwh_assumed", "grid_emission_factor_kgco2_per_kwh",
    "lpg_cylinders_daily_estimated", "lpg_population_index_or_seat_factor",
    "vehicles_scope1_tco2e_daily_prior", "biomass_scope1_tco2e_daily_prior",
    "refrigerants_scope1_tco2e_daily_prior", "tree_sink_tco2e_daily_prior",
]


def to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def fiscal_year_start(d: date) -> int:
    return d.year if d.month >= 4 else d.year - 1


def fiscal_year_label(start_year: int) -> str:
    return f"FY{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def load_calendar_rules() -> dict:
    path = CONFIG_DIR / "calendar_rules.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "summer_start": "05-15", "summer_end": "07-15",
        "winter_start": "12-20", "winter_end": "01-05",
        "summer_school_activity_index": 0.52,
        "winter_school_activity_index": 0.75,
        "normal_school_activity_index": 1.0,
        "annual_electricity_growth_after_last_anchor": 0.022,
    }


def calendar_features(value, rules: dict | None = None) -> dict:
    """Derive weekday/weekend and the project vacation proxies from the date."""
    d = to_date(value)
    rules = rules or load_calendar_rules()
    md = (d.month, d.day)
    s_month, s_day = map(int, rules.get("summer_start", "05-15").split("-"))
    e_month, e_day = map(int, rules.get("summer_end", "07-15").split("-"))
    w_s_month, w_s_day = map(int, rules.get("winter_start", "12-20").split("-"))
    w_e_month, w_e_day = map(int, rules.get("winter_end", "01-05").split("-"))
    summer = ((d.month, d.day) >= (s_month, s_day)) and ((d.month, d.day) <= (e_month, e_day))
    # Winter period crosses New Year.
    winter = ((d.month, d.day) >= (w_s_month, w_s_day)) or ((d.month, d.day) <= (w_e_month, w_e_day))
    dow = d.weekday()  # Monday=0, Sunday=6
    doy = d.timetuple().tm_yday
    activity = float(rules.get("normal_school_activity_index", 1.0))
    if summer:
        activity = float(rules.get("summer_school_activity_index", 0.52))
    elif winter:
        activity = float(rules.get("winter_school_activity_index", 0.75))
    return {
        "date": d.isoformat(),
        "day_name": d.strftime("%A"),
        "month_number": d.month,
        "day_of_month": d.day,
        "day_of_year": doy,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "summer_vacation_proxy": int(summer),
        "winter_break_proxy": int(winter),
        "school_activity_index": activity,
        "month_sin": math.sin(2 * math.pi * d.month / 12.0),
        "month_cos": math.cos(2 * math.pi * d.month / 12.0),
        "day_of_year_sin": math.sin(2 * math.pi * doy / 365.0),
        "day_of_year_cos": math.cos(2 * math.pi * doy / 365.0),
        "fiscal_year_start": fiscal_year_start(d),
        "fiscal_year": fiscal_year_label(fiscal_year_start(d)),
    }


def create_annual_anchors(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["fiscal_year_start"] = np.where(df["date"].dt.month >= 4, df["date"].dt.year, df["date"].dt.year - 1)
    rows = []
    for fy, g in df.groupby("fiscal_year_start", sort=True):
        campus = float(g["campus_gross_electricity_kwh"].sum())
        solar = float(g["solar_generation_kwh"].sum())
        grid_factor = float(g["cea_grid_factor_kg_per_kwh"].mean())
        annual_status = str(g["annual_status"].iloc[0]) if "annual_status" in g else "reconstructed_estimate"
        for zone, (_, col) in ZONE_COLUMNS.items():
            rows.append({
                "fiscal_year_start": int(fy),
                "fiscal_year": fiscal_year_label(int(fy)),
                "zone": zone,
                "zone_label": ZONE_COLUMNS[zone][0],
                "annual_zone_gross_electricity_gwh": float(g[col].sum()) / 1e6,
                "annual_campus_gross_electricity_gwh": campus / 1e6,
                "annual_solar_generation_gwh": solar / 1e6,
                "grid_emission_factor_kgco2_per_kwh": grid_factor,
                "anchor_status": annual_status,
                "source_start_date": g["date"].min().date().isoformat(),
                "source_end_date": g["date"].max().date().isoformat(),
            })
    return pd.DataFrame(rows)


def load_anchors() -> pd.DataFrame:
    if not ANCHOR_FILE.exists():
        prepare_data()
    anchors = pd.read_csv(ANCHOR_FILE)
    anchors["fiscal_year_start"] = anchors["fiscal_year_start"].astype(int)
    return anchors


def get_annual_anchor(d: date, zone: str, anchors: pd.DataFrame | None = None, rules: dict | None = None) -> dict:
    if zone not in ZONE_COLUMNS:
        raise ValueError(f"Unknown zone '{zone}'. Choose one of: {', '.join(ZONE_ORDER)}")
    anchors = anchors if anchors is not None else load_anchors()
    rules = rules or load_calendar_rules()
    fy = fiscal_year_start(to_date(d))
    zone_rows = anchors.loc[anchors["zone"] == zone].sort_values("fiscal_year_start")
    if zone_rows.empty:
        raise ValueError(f"No annual anchors found for zone {zone}")
    exact = zone_rows.loc[zone_rows["fiscal_year_start"] == fy]
    if not exact.empty:
        row = exact.iloc[0].to_dict()
        row["anchor_is_extrapolated"] = False
        return row
    all_fy = sorted(anchors["fiscal_year_start"].unique())
    max_fy = int(max(all_fy)); min_fy = int(min(all_fy))
    if fy > max_fy:
        row = zone_rows.loc[zone_rows["fiscal_year_start"] == max_fy].iloc[0].to_dict()
        delta = fy - max_fy
        growth = float(rules.get("annual_electricity_growth_after_last_anchor", 0.022))
        row["annual_zone_gross_electricity_gwh"] = float(row["annual_zone_gross_electricity_gwh"]) * ((1 + growth) ** delta)
        row["annual_campus_gross_electricity_gwh"] = float(row["annual_campus_gross_electricity_gwh"]) * ((1 + growth) ** delta)
        row["fiscal_year_start"] = fy
        row["fiscal_year"] = fiscal_year_label(fy)
        row["anchor_status"] = f"extrapolated_from_{max_fy}_at_{growth:.1%}_annual_growth"
        row["anchor_is_extrapolated"] = True
        # Solar and grid factor carry forward unchanged under the current assumption.
        return row
    if fy < min_fy:
        row = zone_rows.loc[zone_rows["fiscal_year_start"] == min_fy].iloc[0].to_dict()
        growth = float(rules.get("annual_electricity_growth_after_last_anchor", 0.022))
        delta = min_fy - fy
        row["annual_zone_gross_electricity_gwh"] = float(row["annual_zone_gross_electricity_gwh"]) / ((1 + growth) ** delta)
        row["annual_campus_gross_electricity_gwh"] = float(row["annual_campus_gross_electricity_gwh"]) / ((1 + growth) ** delta)
        row["fiscal_year_start"] = fy
        row["fiscal_year"] = fiscal_year_label(fy)
        row["anchor_status"] = f"backcast_before_{min_fy}_at_{growth:.1%}_annual_growth"
        row["anchor_is_extrapolated"] = True
        return row
    raise ValueError(f"Unable to find annual anchor for fiscal year {fy}")


def make_feature_row(value, zone: str, anchors: pd.DataFrame | None = None, rules: dict | None = None) -> dict:
    d = to_date(value)
    cal = calendar_features(d, rules)
    a = get_annual_anchor(d, zone, anchors=anchors, rules=rules)
    return {
        **{k: cal[k] for k in [
            "date", "day_name", "month_number", "day_of_month", "day_of_year", "day_of_week",
            "is_weekend", "summer_vacation_proxy", "winter_break_proxy", "school_activity_index",
            "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos", "fiscal_year_start", "fiscal_year"]},
        "fiscal_year_offset": int(cal["fiscal_year_start"] - 2022),
        "annual_zone_gross_electricity_gwh_estimated": float(a["annual_zone_gross_electricity_gwh"]),
        "annual_campus_gross_electricity_gwh_estimated": float(a["annual_campus_gross_electricity_gwh"]),
        "annual_solar_generation_gwh_assumed": float(a["annual_solar_generation_gwh"]),
        "grid_emission_factor_kgco2_per_kwh": float(a["grid_emission_factor_kgco2_per_kwh"]),
        "anchor_status": str(a.get("anchor_status", "unknown")),
        "anchor_is_extrapolated": bool(a.get("anchor_is_extrapolated", False)),
        "zone": zone,
        "zone_label": ZONE_COLUMNS[zone][0],
    }



@lru_cache(maxsize=1)
def load_source_cached() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_FILE, parse_dates=["date"])
    return df


def make_campus_feature_row(value, anchors: pd.DataFrame | None = None, source_row=None,
                            source_df: pd.DataFrame | None = None, rules: dict | None = None) -> dict:
    """Build date-only features for a whole-campus selected-source net estimate.

    Source-category priors (LPG/vehicles/biomass/refrigerants/tree sink) are lookup values
    from the reconstructed activity dataset. For dates beyond its range, monthly LPG
    patterns and the latest annual daily priors are carried forward as explicit estimates.
    """
    d = to_date(value)
    rules = rules or load_calendar_rules()
    anchors = anchors if anchors is not None else load_anchors()
    cal = calendar_features(d, rules)
    anchor_by_zone = {z: get_annual_anchor(d, z, anchors=anchors, rules=rules) for z in ZONE_ORDER}
    a0 = anchor_by_zone[ZONE_ORDER[0]]
    out = {k: cal[k] for k in [
        "month_number", "day_of_month", "day_of_year", "day_of_week", "is_weekend",
        "summer_vacation_proxy", "winter_break_proxy", "school_activity_index",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]}
    out["fiscal_year_offset"] = int(cal["fiscal_year_start"] - 2022)
    for z, colname in [
        ("academic", "annual_academic_gross_electricity_gwh_estimated"),
        ("hostel", "annual_hostel_gross_electricity_gwh_estimated"),
        ("residential", "annual_residential_gross_electricity_gwh_estimated"),
        ("common", "annual_common_gross_electricity_gwh_estimated"),
    ]:
        out[colname] = float(anchor_by_zone[z]["annual_zone_gross_electricity_gwh"])
    out["annual_campus_gross_electricity_gwh_estimated"] = float(a0["annual_campus_gross_electricity_gwh"])
    out["annual_solar_generation_gwh_assumed"] = float(a0["annual_solar_generation_gwh"])
    out["grid_emission_factor_kgco2_per_kwh"] = float(a0["grid_emission_factor_kgco2_per_kwh"])

    if source_row is None:
        src = source_df if source_df is not None else load_source_cached()
        tmp = src.copy()
        if not pd.api.types.is_datetime64_any_dtype(tmp["date"]):
            tmp["date"] = pd.to_datetime(tmp["date"])
        match = tmp.loc[tmp["date"].dt.date == d]
        if not match.empty:
            source_row = match.iloc[0]
        else:
            # Beyond supplied dates, repeat the latest available profile for the same calendar month.
            tmp["fiscal_year_start"] = np.where(tmp["date"].dt.month >= 4, tmp["date"].dt.year, tmp["date"].dt.year - 1)
            latest_fy = int(tmp["fiscal_year_start"].max())
            earliest_fy = int(tmp["fiscal_year_start"].min())
            fy_for_pattern = latest_fy if fiscal_year_start(d) > latest_fy else earliest_fy
            monthly = tmp.loc[(tmp["fiscal_year_start"] == fy_for_pattern) & (tmp["date"].dt.month == d.month)]
            if monthly.empty:
                monthly = tmp.loc[tmp["fiscal_year_start"] == fy_for_pattern]
            # Pick an arbitrary row then override monthly/annual values below.
            source_row = monthly.iloc[0].copy()
            for c in ["lpg_cylinders_daily_estimated", "lpg_population_index_or_seat_factor"]:
                source_row[c] = float(monthly[c].mean())
            for c in ["vehicles_scope1_tco2e_daily_prior", "biomass_scope1_tco2e_daily_prior",
                      "refrigerants_scope1_tco2e_daily_prior", "tree_sink_tco2e_daily_prior"]:
                fy_group = tmp.loc[tmp["fiscal_year_start"] == fy_for_pattern, c]
                source_row[c] = float(fy_group.mean())

    out["lpg_cylinders_daily_estimated"] = float(source_row["lpg_cylinders_daily_estimated"])
    out["lpg_population_index_or_seat_factor"] = float(source_row["lpg_population_index_or_seat_factor"])
    out["vehicles_scope1_tco2e_daily_prior"] = float(source_row["vehicles_scope1_tco2e_daily_prior"])
    out["biomass_scope1_tco2e_daily_prior"] = float(source_row["biomass_scope1_tco2e_daily_prior"])
    out["refrigerants_scope1_tco2e_daily_prior"] = float(source_row["refrigerants_scope1_tco2e_daily_prior"])
    out["tree_sink_tco2e_daily_prior"] = float(source_row["tree_sink_tco2e_daily_prior"])
    return out


def add_zone_targets(source: pd.DataFrame) -> pd.DataFrame:
    """Allocate campus grid imports to zones by each zone's gross electricity share.

    This is the declared modelling convention because daily solar generation is not
    available by zone. It preserves the campus grid-import total across all zones.
    """
    df = source.copy()
    df["date"] = pd.to_datetime(df["date"])
    factor = pd.to_numeric(df["cea_grid_factor_kg_per_kwh"], errors="raise")
    campus = pd.to_numeric(df["campus_gross_electricity_kwh"], errors="raise").replace(0, np.nan)
    grid = pd.to_numeric(df["grid_import_kwh"], errors="raise")
    rows = []
    for ix, r in df.iterrows():
        campus_gross = float(r["campus_gross_electricity_kwh"])
        daily_grid = float(r["grid_import_kwh"])
        grid_factor = float(r["cea_grid_factor_kg_per_kwh"])
        for zone, (label, col) in ZONE_COLUMNS.items():
            zone_gross = float(r[col])
            share = zone_gross / campus_gross if campus_gross else 0.0
            zone_grid = daily_grid * share
            target = zone_grid * grid_factor / 1000.0
            rows.append({"row_id": int(ix), "date": r["date"], "zone": zone, "zone_label": label,
                         "zone_gross_electricity_kwh_reconstructed": zone_gross,
                         "zone_grid_import_kwh_allocated": zone_grid,
                         TARGET: target,
                         "campus_scope2_co2e_tonnes_reference": daily_grid * grid_factor / 1000.0,
                         "daily_target_status": r.get("daily_target_status", "reconstructed"),
                         "annual_status": r.get("annual_status", "reconstructed")})
    return pd.DataFrame(rows)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild annual anchors, historical zone dataset, and projected references."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_FILE, parse_dates=["date"])
    anchors = create_annual_anchors(source)
    anchors.to_csv(ANCHOR_FILE, index=False)
    # Only through FY2025-26 goes into supervised training/evaluation. Future projection rows are kept separately.
    historical = source.loc[pd.to_datetime(source["date"]) <= pd.Timestamp("2026-03-31")].copy()
    historical["date"] = pd.to_datetime(historical["date"])
    target_table = add_zone_targets(historical)
    rules = load_calendar_rules()
    records = []
    anchor_df = anchors
    for _, row in target_table.iterrows():
        d = row["date"].date() if isinstance(row["date"], pd.Timestamp) else to_date(row["date"])
        f = make_feature_row(d, row["zone"], anchors=anchor_df, rules=rules)
        records.append({
            "date": d.isoformat(), "zone": row["zone"], "zone_label": row["zone_label"],
            **{c: f[c] for c in FEATURE_COLUMNS},
            "zone_gross_electricity_kwh_reconstructed": row["zone_gross_electricity_kwh_reconstructed"],
            "zone_grid_import_kwh_allocated": row["zone_grid_import_kwh_allocated"],
            TARGET: row[TARGET],
            "campus_scope2_co2e_tonnes_reference": row["campus_scope2_co2e_tonnes_reference"],
            "data_source_status": row["daily_target_status"],
            "annual_source_status": row["annual_status"],
        })
    model_df = pd.DataFrame(records).sort_values(["zone", "date"]).reset_index(drop=True)
    model_df.to_csv(DATA_DIR / "zone_daily_model_dataset.csv", index=False)

    # Export projected date/zone rows only as reference inputs; their generated targets must not be used for training/testing.
    future = source.loc[pd.to_datetime(source["date"]) > pd.Timestamp("2026-03-31")].copy()
    if not future.empty:
        future["date"] = pd.to_datetime(future["date"])
        ft = add_zone_targets(future)
        frecords = []
        for _, row in ft.iterrows():
            d = to_date(row["date"])
            f = make_feature_row(d, row["zone"], anchors=anchor_df, rules=rules)
            frecords.append({"date": d.isoformat(), "zone": row["zone"], "zone_label": row["zone_label"],
                             **{c: f[c] for c in FEATURE_COLUMNS},
                             "reference_projected_zone_scope2_co2e_tonnes": row[TARGET],
                             "status": "projected_reference_NOT_FOR_TRAINING_OR_TESTING"})
        pd.DataFrame(frecords).to_csv(DATA_DIR / "future_projection_reference_NOT_FOR_TRAINING.csv", index=False)

    # Whole-campus selected-source target uses the daily table's net subtotal, including
    # available LPG, selected Scope 1 priors and tree sink. It has no zone attribution.
    campus_records = []
    for _, row in historical.iterrows():
        d = row["date"].date()
        f = make_campus_feature_row(d, anchors=anchors, source_row=row, rules=rules)
        campus_records.append({
            "date": d.isoformat(), **{c: f[c] for c in CAMPUS_FEATURE_COLUMNS},
            CAMPUS_TARGET: float(row[CAMPUS_TARGET]),
            "scope2_cea_tco2e": float(row["scope2_cea_tco2e"]),
            "lpg_combustion_co2_tonnes_daily": float(row["lpg_combustion_co2_tonnes_daily"]),
            "vehicles_scope1_tco2e_daily_prior": float(row["vehicles_scope1_tco2e_daily_prior"]),
            "biomass_scope1_tco2e_daily_prior": float(row["biomass_scope1_tco2e_daily_prior"]),
            "refrigerants_scope1_tco2e_daily_prior": float(row["refrigerants_scope1_tco2e_daily_prior"]),
            "tree_sink_tco2e_daily_prior": float(row["tree_sink_tco2e_daily_prior"]),
            "data_source_status": str(row["daily_target_status"]),
        })
    campus_df = pd.DataFrame(campus_records).sort_values("date").reset_index(drop=True)
    campus_df.to_csv(DATA_DIR / "campus_daily_model_dataset.csv", index=False)
    if not future.empty:
        campus_future_rows = []
        for _, row in future.iterrows():
            d = row["date"].date()
            f = make_campus_feature_row(d, anchors=anchors, source_row=row, rules=rules)
            campus_future_rows.append({"date": d.isoformat(), **{c: f[c] for c in CAMPUS_FEATURE_COLUMNS},
                                       "reference_projected_campus_net_selected_sources_tco2e": float(row[CAMPUS_TARGET]),
                                       "status": "projected_reference_NOT_FOR_TRAINING_OR_TESTING"})
        pd.DataFrame(campus_future_rows).to_csv(DATA_DIR / "future_campus_reference_NOT_FOR_TRAINING.csv", index=False)
    return model_df, anchors, source
