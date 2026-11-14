"""Orchestration shared by the Streamlit page and the command-line collector."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from . import source
from .alerts import compute_alerts
from .analysis import daily_flows, observed_intervals
from .config import SETTINGS, SOURCE_URL, Settings
from .forecast import run_forecast
from .imputation import build_panel, impute_rows, observed_matrix
from .storage import Store, now_local


def today() -> date:
    return now_local().date()


@dataclass
class History:
    catalog: pd.DataFrame
    snapshots: pd.DataFrame
    obs: pd.DataFrame          # observed images (snapshot date x maturity date)
    panel: pd.DataFrame        # daily images incl. imputed days
    imputed: pd.Series
    alive_on_maturity: bool
    flows: pd.DataFrame        # daily total + issued / redeemed / matured
    intervals: pd.DataFrame    # flows between observed snapshots
    vintages: pd.DataFrame
    fc_totals: pd.DataFrame
    fc_buckets: pd.DataFrame
    alerts: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.catalog.empty

    @property
    def actual_obs(self) -> pd.Series:
        return self.obs.sum(axis=1) if not self.obs.empty else pd.Series(dtype=float)


# --------------------------------------------------------------------------- ingestion
def ingest(store: Store, content: bytes, day: date | None = None, origin: str = "descarga",
           settings: Settings = SETTINGS) -> dict:
    # fresh downloads: only trust a cut date close to today; historical files: any date in the header
    reference = day if day is not None else (today() if origin == "descarga" else None)
    rows, cut = source.parse_file(content, reference=reference)
    if day is None and cut is None and origin != "descarga":
        raise ValueError("No se encontró la fecha de corte en el archivo: indique la fecha de la imagen")
    day = pd.Timestamp(day or cut or today()).normalize()
    rows, n_imp, n_drop = impute_rows(rows, store.previous_snapshot(day), settings)
    status = store.save_snapshot(day, rows, content if settings.keep_raw else None,
                                 origin=origin, source_cut_date=cut)
    if status == "updated" and store.has_forecast(day):
        _forecast_for(store, day, store.load_snapshots(), settings)   # image changed -> redo vintage
    return {"status": status, "date": day.date(), "rows": len(rows), "imputed_rows": n_imp,
            "dropped_rows": n_drop, "cut_date": cut, "total": float(rows["balance"].sum())}


def fetch_and_ingest(store: Store, url: str = SOURCE_URL, settings: Settings = SETTINGS) -> dict:
    try:
        content = source.download(url)
    except Exception as exc:  # network / server errors: keep working with stored history
        return {"status": "error", "error": f"No se pudo descargar el archivo: {exc}"}
    try:
        return ingest(store, content, origin="descarga", settings=settings)
    except Exception as exc:
        return {"status": "error", "error": f"No se pudo leer el archivo descargado: {exc}"}


# --------------------------------------------------------------------------- forecasts
def _forecast_for(store: Store, asof: pd.Timestamp, all_snaps: pd.DataFrame, settings: Settings) -> None:
    snaps = all_snaps[all_snaps["snapshot_date"] <= asof]       # no look-ahead
    panel, _, aom = build_panel(snaps, settings)
    if panel.empty:
        return
    total, buckets, _, meta = run_forecast(panel, aom, snaps, settings)
    store.save_forecast(asof, total, buckets, meta)


def refresh_forecasts(store: Store, settings: Settings = SETTINGS, rebuild: bool = False) -> int:
    """Create the forecast vintage of every snapshot date that does not have one yet.

    Vintages are write-once: they are the record of what was predicted at the time.
    """
    dates = store.snapshot_dates()
    todo = [d for d in dates if rebuild or not store.has_forecast(d)]
    if not todo:
        return 0
    snaps = store.load_snapshots()
    for d in todo:
        _forecast_for(store, d, snaps, settings)
    return len(todo)


# --------------------------------------------------------------------------- history
def load_history(store: Store, settings: Settings = SETTINGS) -> History:
    catalog = store.catalog()
    snaps = store.load_snapshots()
    obs = observed_matrix(snaps)
    panel, imputed, aom = build_panel(snaps, settings)
    flows = daily_flows(panel, imputed, aom)
    intervals = observed_intervals(obs, aom) if len(obs) > 1 else pd.DataFrame()
    totals, buckets = store.load_forecasts()
    return History(catalog, snaps, obs, panel, imputed, aom, flows, intervals,
                   store.vintages(), totals, buckets, store.alerts())


def refresh_alerts(store: Store, h: History, settings: Settings = SETTINGS) -> pd.DataFrame:
    intervals = h.intervals if not h.intervals.empty else pd.DataFrame(
        columns=["date", "redeemed_per_day", "issued_per_day"])
    fresh = compute_alerts(h.catalog, intervals, h.vintages, h.fc_totals, h.actual_obs, settings)
    return store.save_alerts(fresh)


def update(store: Store, fetch: bool = True, settings: Settings = SETTINGS,
           rebuild_forecasts: bool = False) -> tuple[dict | None, History]:
    """Full daily cycle: download -> save image -> forecasts -> alerts."""
    result = fetch_and_ingest(store, settings=settings) if fetch else None
    refresh_forecasts(store, settings, rebuild=rebuild_forecasts)
    h = load_history(store, settings)
    if not h.empty:
        h.alerts = refresh_alerts(store, h, settings)
    return result, h
