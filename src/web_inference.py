"""Lightweight XGBoost-only inference for the Streamlit website.

Uses the exact feature-generation code and production XGBoost artifacts from
IITM_Carbon_Digital_Twin_Date_Zone_Model. PyTorch/LSTM is intentionally not
loaded: XGBoost was the preferred model in the project's chronological test.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import (
    CAMPUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    ZONE_COLUMNS,
    ZONE_ORDER,
    load_anchors,
    load_calendar_rules,
    load_source_cached,
    make_campus_feature_row,
    make_feature_row,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "production"


@lru_cache(maxsize=8)
def _load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required model is missing: {path}. Copy the production XGBoost model files into models/production/."
        )
    return joblib.load(path)


def predict_period(start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (campus_daily, zones_daily) for every day in an inclusive date range.

    campus_daily prediction is a campus-wide selected-source net subtotal.
    zones_daily predictions are zone-level electricity Scope 2 only. They are
    separate outputs and must not be added together.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError("End date must be on or after start date.")

    dates = pd.date_range(start=start, end=end, freq="D")
    anchors = load_anchors()
    rules = load_calendar_rules()
    source = load_source_cached()
    date_values = [ts.date() for ts in dates]

    zone_frames: list[pd.DataFrame] = []
    for zone in ZONE_ORDER:
        feature_rows = [make_feature_row(d, zone, anchors=anchors, rules=rules) for d in date_values]
        X_zone = pd.DataFrame(
            [{column: row[column] for column in FEATURE_COLUMNS} for row in feature_rows],
            columns=FEATURE_COLUMNS,
        ).astype(float)
        model = _load_model(str(MODEL_DIR / f"xgboost_{zone}.joblib"))
        predictions = model.predict(X_zone)
        zone_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "zone": zone,
                    "zone_label": ZONE_COLUMNS[zone][0],
                    "predicted_scope2_tco2e_per_day": [max(0.0, float(p)) for p in predictions],
                    "anchor_status": [row["anchor_status"] for row in feature_rows],
                    "anchor_is_extrapolated": [bool(row["anchor_is_extrapolated"]) for row in feature_rows],
                }
            )
        )

    campus_rows = [
        make_campus_feature_row(d, anchors=anchors, source_df=source, rules=rules)
        for d in date_values
    ]
    X_campus = pd.DataFrame(
        [{column: row[column] for column in CAMPUS_FEATURE_COLUMNS} for row in campus_rows],
        columns=CAMPUS_FEATURE_COLUMNS,
    ).astype(float)
    campus_model = _load_model(str(MODEL_DIR / "xgboost_campus_total.joblib"))
    campus_predictions = campus_model.predict(X_campus)
    campus_daily = pd.DataFrame(
        {
            "date": dates,
            "predicted_campus_net_selected_sources_tco2e_per_day": [
                max(0.0, float(p)) for p in campus_predictions
            ],
            "fiscal_year": [
                f"FY{d.year if d.month >= 4 else d.year - 1}-{str(((d.year if d.month >= 4 else d.year - 1) + 1) % 100).zfill(2)}"
                for d in date_values
            ],
        }
    )

    return campus_daily, pd.concat(zone_frames, ignore_index=True)
