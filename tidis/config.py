"""Central configuration. Everything that changes behaviour lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

SOURCE_URL = (
    "https://d1b4gd4m8561gs.cloudfront.net/sites/default/files/"
    "caracteristicas-financieras-titulos-vigentes.xlsx"
)

# Column names in the source file
CATEGORY_COL = "Categoría del Título"
BALANCE_COL = "Saldo en circulación"
MATURITY_COL = "Fecha de vencimiento"
CATEGORY_VALUE = "TIDIS"
# Optional columns: used if present, ignored otherwise
ID_CANDIDATES = ("ISIN", "Código ISIN", "Codigo ISIN", "Nemotécnico", "Nemotecnico")
ISSUE_CANDIDATES = ("Fecha de emisión", "Fecha de emision", "Fecha de expedición", "Fecha de expedicion")

# Colombia has no DST: fixed UTC-5
LOCAL_TZ = timezone(timedelta(hours=-5), name="America/Bogota")

# Where history is stored. Override with the TIDIS_DATA_DIR environment variable.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_dir() -> Path:
    return Path(os.environ.get("TIDIS_DATA_DIR", DEFAULT_DATA_DIR))


@dataclass(frozen=True)
class Settings:
    # --- storage ------------------------------------------------------------
    keep_raw: bool = True             # keep the original .xlsx of every day (allows re-parsing later)

    # --- imputation of missing days ---------------------------------------
    imputation: str = "linear"        # "linear" or "ffill" for titles alive on both sides of a gap
    tenor_days: int = 365             # TIDIS validity (1 year); used when no issue date is available

    # --- depreciation index & forecast --------------------------------------
    alpha: float = 0.05               # smoothing of the depreciation index: weight of the newest day
                                      # (small = softer trend, 1 = no smoothing)
    lookback_days: int = 180          # window for the days-to-maturity shape (multi-image)
    horizon_days: int = 365           # forecast horizon (covers the whole life of current titles)
    dtm_bin_days: int = 30            # days-to-maturity buckets of the shape
    min_bin_transitions: int = 20     # below this, a bucket uses the average rate
    include_new_issuance: bool = True
    issuance_window_days: int = 60    # new issuance = average of the last N days

    # --- alerts (all relative to the value they describe) ---------------------
    usage_rel_pct: float = 100.0      # daily usage (% of balance) vs its smoothed level: +100% = double
    issuance_rel_pct: float = 150.0   # daily issuance (% of balance) vs its smoothed level
    balance_change_pct: float = 2.0   # daily change of the total balance
    index_change_pct: float = 50.0    # change of the smoothed depreciation rate between images
    forecast_change_pct: float = 5.0  # revision between consecutive forecasts (same target date)
    unused_change_pct: float = 30.0   # revision of the balance expected to expire unused
    forecast_error_pct: float = 3.0   # previous forecast vs today's observed balance
    method_gap_pct: float = 15.0      # single-image vs multi-image forecast at +90 days
    min_history_flow_alerts: int = 10 # observed intervals needed before flow / index alerts fire
    min_history_forecast_alerts: int = 15  # images needed before forecast alerts fire
    check_horizons: tuple = (30, 90, 180)
    stale_days: int = 3               # warn if latest image is older than this


SETTINGS = Settings()
