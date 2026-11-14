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

    # --- forecast ---------------------------------------------------------
    lookback_days: int = 180          # history window used to estimate usage & issuance
    horizon_days: int = 365           # forecast horizon (covers the whole life of current titles)
    dtm_bin_days: int = 30            # usage rates are estimated per days-to-maturity bin
    min_bin_transitions: int = 20     # below this, a bin falls back to the global rate
    include_new_issuance: bool = True

    # --- alerts -----------------------------------------------------------
    z_threshold: float = 3.5          # modified z-score for unusual usage / issuance
    min_history_flow_alerts: int = 10 # observed intervals needed before flow alerts fire
    min_history_forecast_alerts: int = 15 # snapshots needed before forecast alerts fire
    forecast_change_pct: float = 5.0  # revision between consecutive forecasts
    forecast_error_pct: float = 3.0   # error of previous forecast vs today's actual
    check_horizons: tuple = (30, 90, 180)
    stale_days: int = 3               # warn if latest snapshot is older than this


SETTINGS = Settings()
