"""Forecast of the TIDIS stock based on how the images degrade.

Model (transparent, cohort style):
  * usage ("degradation") rate: daily fraction of a maturity bucket that is used
    (redeemed for taxes) as a function of its days-to-maturity, estimated from the
    daily panel over the last `lookback_days` (volume weighted);
  * maturity: a bucket disappears when it reaches its maturity date;
  * new issuance: average daily issuance in the lookback window, issued with the
    usual tenor (1 year) and degrading with the same curve.
With a single snapshot there is nothing to learn from, so the model falls back to
the contractual run-off (balances only disappear at maturity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .imputation import is_alive


@dataclass
class UsageModel:
    kind: str                          # "runoff" | "usage"
    bin_days: int
    rates: pd.Series = field(default_factory=pd.Series)   # daily rate per dtm bin (bin start)
    global_rate: float = 0.0
    issuance_per_day: float = 0.0
    tenor_days: int = 365
    n_transitions: int = 0
    window_start: pd.Timestamp | None = None
    _lookup: np.ndarray | None = field(default=None, repr=False)

    def rate_for(self, dtm: np.ndarray) -> np.ndarray:
        if self.kind == "runoff":
            return np.zeros(len(dtm))
        if self._lookup is None:
            size = int(max(self.rates.index.max() if len(self.rates) else 0, 0) // self.bin_days) + 1
            self._lookup = np.full(size, self.global_rate)
            self._lookup[(self.rates.index.values // self.bin_days).astype(int)] = self.rates.values
        idx = np.maximum(dtm, 0) // self.bin_days
        out = np.full(len(dtm), self.global_rate)
        inside = idx < len(self._lookup)
        out[inside] = self._lookup[idx[inside]]
        return out


def estimate_model(panel: pd.DataFrame, alive_on_maturity: bool, snapshots: pd.DataFrame | None = None,
                   settings: Settings = SETTINGS) -> UsageModel:
    tenor = settings.tenor_days
    if snapshots is not None and snapshots["issue_date"].notna().any():
        t = (snapshots["maturity_date"] - snapshots["issue_date"]).dt.days.dropna()
        t = t[(t > 30) & (t < 1200)]
        if len(t):
            tenor = int(t.median())
    if len(panel) < 2:
        return UsageModel("runoff", settings.dtm_bin_days, tenor_days=tenor)

    window = panel[panel.index >= panel.index[-1] - pd.Timedelta(days=settings.lookback_days)]
    vals = window.to_numpy()
    a, b = vals[:-1], vals[1:]                                  # consecutive days
    mats = window.columns.values[None, :]
    d0 = window.index.values[:-1, None]
    d1 = window.index.values[1:, None]
    alive = mats >= d1 if alive_on_maturity else mats > d1
    live = (a > 0) & alive                                      # exposed and not maturing
    days = len(window) - 1
    issuance = float(np.clip(b - a, 0, None).sum()) / days if days else 0.0
    exposure = a[live]
    if days == 0 or exposure.sum() <= 0:
        return UsageModel("runoff", settings.dtm_bin_days, issuance_per_day=issuance, tenor_days=tenor)
    used = np.clip(a - b, 0, None)[live]
    dtm = ((mats - d0) // np.timedelta64(1, "D"))[live]
    bins = (np.maximum(dtm, 0) // settings.dtm_bin_days).astype(int)
    used_b = np.bincount(bins, weights=used)
    exp_b = np.bincount(bins, weights=exposure)
    n_b = np.bincount(bins)
    ok = (n_b >= settings.min_bin_transitions) & (exp_b > 0)
    rates = pd.Series(np.divide(used_b, exp_b, out=np.zeros_like(used_b), where=exp_b > 0),
                      index=np.arange(len(n_b)) * settings.dtm_bin_days)[ok]
    global_rate = float(used.sum() / exposure.sum())
    return UsageModel("usage", settings.dtm_bin_days, rates.clip(0, 1), min(global_rate, 1.0),
                      issuance, tenor, int(live.sum()), window.index[0])


def project(image: pd.Series, asof: pd.Timestamp, model: UsageModel, alive_on_maturity: bool,
            settings: Settings = SETTINGS):
    """Project the image forward.

    Returns
      total    DataFrame(target_date, total, existing, new_issues, used, matured)
      buckets  DataFrame(maturity_date, balance_asof, expected_at_maturity, expected_used)
               for the titles outstanding at `asof`.
    """
    asof = pd.Timestamp(asof).normalize()
    image = image[(image > 0) & is_alive(image.index.values, asof, alive_on_maturity)]
    mats = image.index.values.astype("datetime64[ns]")
    bal = image.values.astype(float).copy()
    n_existing = len(bal)
    at_maturity = np.full(n_existing, np.nan)
    issue_new = settings.include_new_issuance and model.kind != "runoff" and model.issuance_per_day > 0
    rows = []
    for step in range(1, settings.horizon_days + 1):
        prev_day = asof + pd.Timedelta(days=step - 1)
        day = asof + pd.Timedelta(days=step)
        dtm = ((mats - np.datetime64(prev_day)) // np.timedelta64(1, "D")).astype(int)
        used = bal * model.rate_for(dtm)
        bal = bal - used
        expiring = ~is_alive(mats, day, alive_on_maturity) & (bal > 0)
        matured = float(bal[expiring].sum())
        exp_idx = np.where(expiring[:n_existing])[0]
        at_maturity[exp_idx] = bal[exp_idx]
        bal[expiring] = 0.0
        if issue_new:
            mats = np.append(mats, np.datetime64(day + pd.Timedelta(days=model.tenor_days), "ns"))
            bal = np.append(bal, model.issuance_per_day)
        rows.append({"target_date": day, "total": float(bal.sum()), "existing": float(bal[:n_existing].sum()),
                     "new_issues": float(bal[n_existing:].sum()), "used": float(used.sum()), "matured": matured})
    total = pd.DataFrame(rows)
    # titles still alive at the end of the horizon keep their projected balance
    still = np.isnan(at_maturity)
    at_maturity[still] = bal[:n_existing][still]
    buckets = pd.DataFrame({"maturity_date": image.index, "balance_asof": image.values,
                            "expected_at_maturity": at_maturity})
    buckets["expected_used"] = buckets["balance_asof"] - buckets["expected_at_maturity"]
    return total, buckets


def run_forecast(panel: pd.DataFrame, alive_on_maturity: bool, snapshots: pd.DataFrame,
                 settings: Settings = SETTINGS):
    asof = panel.index[-1]
    model = estimate_model(panel, alive_on_maturity, snapshots, settings)
    total, buckets = project(panel.loc[asof], asof, model, alive_on_maturity, settings)
    meta = {
        "model": model.kind, "global_daily_rate": model.global_rate,
        "issuance_per_day": model.issuance_per_day, "tenor_days": model.tenor_days,
        "n_snapshots": int(snapshots["snapshot_date"].nunique()), "n_transitions": model.n_transitions,
        "balance_asof": float(panel.loc[asof].sum()),
        "expected_unused_at_maturity": float(buckets["expected_at_maturity"].sum()),
    }
    return total, buckets, model, meta
