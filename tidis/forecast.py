"""Depreciation index of the TIDIS stock and forecasts built on it.

Two ways of measuring how fast the TIDIS balance "depreciates" (is used to pay taxes):

* single image ("una imagen")
    Inside one image, titles of different ages coexist: young titles (just issued) have
    barely been used, old titles have been used for months. The balance by *days since
    issue* therefore traces a depreciation curve. It is noisy (issuance varies day to day),
    so it is smoothed with the parameter alpha (zero-phase exponential smoothing in log
    space), forced to be non-increasing (isotonic fit) and normalised to 100 at issue.

* multiple images ("varias imágenes")
    Comparing consecutive daily images gives the fraction of the live stock used each day.
    Chaining those daily rates gives a depreciation index over calendar time (base 100).
    The daily rate series is smoothed with the same alpha (exponential smoothing); the last
    smoothed value is the level used to project. The relative shape by days-to-maturity
    (titles close to maturity are used faster) is estimated over the lookback window.

New issuance (both methods): average daily issuance of the last `issuance_window_days`.

alpha in (0, 1]: weight of the newest observation. Small alpha = softer trend (long memory),
alpha = 1 = no smoothing.

Both indices feed the same projection: each maturity date is depreciated day by day,
disappears at maturity (what is left "expires unused"), and new issuance is added.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .imputation import is_alive

METHODS = {"single": "Una imagen", "multi": "Varias imágenes"}


# --------------------------------------------------------------------------- smoothing
def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """One-sided exponential smoothing (NaNs keep the previous level)."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    level = np.nan
    for i, v in enumerate(x):
        if np.isfinite(v):
            level = v if not np.isfinite(level) else alpha * v + (1 - alpha) * level
        out[i] = level
    return out


def smooth_zero_phase(x: np.ndarray, alpha: float) -> np.ndarray:
    """Forward + backward exponential smoothing: softens without shifting the curve.

    The ends are padded with an odd reflection (as scipy's filtfilt does) so that a straight
    trend is preserved up to the edges instead of being bent towards the first/last point."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pad = min(n - 1, int(np.ceil(7 / alpha))) if n > 1 else 0
    if pad:
        # pivot each edge on a local linear fit (not on the single, noisy, edge value)
        k = max(2, min(n, int(np.ceil(1 / alpha))))
        t = np.arange(k)
        left = np.polyval(np.polyfit(t, x[:k], 1), 0)
        right = np.polyval(np.polyfit(t, x[-k:], 1), k - 1)
        x = np.concatenate([2 * left - x[pad:0:-1], x, 2 * right - x[-2:-pad - 2:-1]])
    out = (ewma(x, alpha) + ewma(x[::-1], alpha)[::-1]) / 2
    return out[pad:pad + n]


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Best non-increasing fit (least squares, pool-adjacent-violators)."""
    vals, weights, sizes = [], [], []
    for v in np.asarray(y, dtype=float):
        vals.append(v); weights.append(1.0); sizes.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:          # violation of "non-increasing"
            w = weights[-2] + weights[-1]
            vals[-2] = (vals[-2] * weights[-2] + vals[-1] * weights[-1]) / w
            weights[-2] = w; sizes[-2] += sizes[-1]
            vals.pop(); weights.pop(); sizes.pop()
    return np.repeat(vals, sizes)


def estimate_tenor(snapshots: pd.DataFrame | None, settings: Settings) -> int:
    if snapshots is not None and len(snapshots) and snapshots["issue_date"].notna().any():
        t = (snapshots["maturity_date"] - snapshots["issue_date"]).dt.days.dropna()
        t = t[(t > 30) & (t < 1200)]
        if len(t):
            return int(t.median())
    return settings.tenor_days


# --------------------------------------------------------------------------- single image
@dataclass
class SingleImageIndex:
    asof: pd.Timestamp
    alpha: float
    tenor: int
    kind: str                                   # "index" | "runoff"
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)  # by age (days since issue)
    daily_rate: float = 0.0                     # equivalent constant daily rate over the whole life
    issuance_per_day: float = 0.0
    _rates: np.ndarray | None = field(default=None, repr=False)

    def rate_for(self, dtm: np.ndarray) -> np.ndarray:
        if self.kind == "runoff" or self._rates is None:
            return np.zeros(len(dtm))
        age = np.clip(self.tenor - np.asarray(dtm), 0, len(self._rates) - 1)
        return self._rates[age]


def single_image_index(image: pd.Series, asof, alive_on_maturity: bool, alpha: float, tenor: int,
                       settings: Settings = SETTINGS) -> SingleImageIndex:
    asof = pd.Timestamp(asof).normalize()
    img = image[(image > 0) & is_alive(image.index.values, asof, alive_on_maturity)]
    dtm = (img.index - asof).days.values
    age = np.clip(tenor - dtm, 0, tenor)
    raw = np.zeros(tenor + 1)
    np.add.at(raw, age, img.values)
    if (raw > 0).sum() < 3:
        return SingleImageIndex(asof, alpha, tenor, "runoff")

    # days without balance = no issuance that day (weekends, holidays): interpolate them in log space
    ages = np.arange(tenor + 1)
    pos = raw > 0
    log_filled = np.interp(ages, ages[pos], np.log(raw[pos]))
    log_smooth = smooth_zero_phase(log_filled, alpha)        # alpha softens the curve
    # a title can only lose balance with age: best non-increasing fit, then smoothed again with alpha
    # (a positive-weight smoother keeps it non-increasing; cummin only removes rounding at the edges)
    log_mono = np.minimum.accumulate(smooth_zero_phase(isotonic_decreasing(log_smooth), alpha))
    base = log_mono[0]
    index = 100 * np.exp(log_smooth - base)                  # smoothed index, 100 at issue
    mono = 100 * np.exp(log_mono - base)                     # monotone version used to project
    index_raw = 100 * raw / np.exp(base)
    rates = np.clip(1 - mono[1:] / mono[:-1], 0, 1)
    rates = np.append(rates, rates[-1])
    avg_rate = 1 - (mono[-1] / 100) ** (1 / tenor)           # equivalent constant daily rate
    # issuance: young titles corrected by their (small) depreciation, per calendar day
    young = ages < settings.issuance_window_days
    issuance = float(np.mean(raw[young] / (mono[young] / 100)))
    profile = pd.DataFrame({"age": ages, "dtm": tenor - ages, "balance": raw, "index_raw": index_raw,
                            "index": index, "index_mono": mono, "daily_rate": rates})
    return SingleImageIndex(asof, alpha, tenor, "index", profile, float(avg_rate), issuance, rates)


# --------------------------------------------------------------------------- multiple images
@dataclass
class MultiImageIndex:
    alpha: float
    tenor: int
    kind: str                                   # "index" | "runoff"
    series: pd.DataFrame = field(default_factory=pd.DataFrame)   # by calendar day
    shape: pd.Series = field(default_factory=pd.Series)          # multiplier per dtm bin
    bin_days: int = 30
    level: float = 0.0                          # last smoothed daily rate
    issuance_per_day: float = 0.0
    n_transitions: int = 0
    _lookup: np.ndarray | None = field(default=None, repr=False)

    def rate_for(self, dtm: np.ndarray) -> np.ndarray:
        if self.kind == "runoff":
            return np.zeros(len(dtm))
        if self._lookup is None:
            size = int(max(self.shape.index.max() if len(self.shape) else 0, 0) // self.bin_days) + 1
            self._lookup = np.ones(size)
            self._lookup[(self.shape.index.values // self.bin_days).astype(int)] = self.shape.values
        idx = np.maximum(np.asarray(dtm), 0) // self.bin_days
        mult = np.ones(len(idx))
        inside = idx < len(self._lookup)
        mult[inside] = self._lookup[idx[inside]]
        return np.clip(mult * self.level, 0, 1)


def daily_depreciation(panel: pd.DataFrame, alive_on_maturity: bool) -> pd.DataFrame:
    """Daily usage of the live stock: exposure, used, issued and rate (used / exposure)."""
    if len(panel) < 2:
        return pd.DataFrame(columns=["exposure", "used", "issued", "rate"])
    vals = panel.to_numpy()
    a, b = vals[:-1], vals[1:]
    mats = panel.columns.values[None, :]
    d1 = panel.index.values[1:, None]
    live = (a > 0) & (mats >= d1 if alive_on_maturity else mats > d1)
    exposure = np.where(live, a, 0).sum(axis=1)
    used = np.where(live, np.clip(a - b, 0, None), 0).sum(axis=1)
    issued = np.clip(b - a, 0, None).sum(axis=1)
    rate = np.divide(used, exposure, out=np.full(len(used), np.nan), where=exposure > 0)
    return pd.DataFrame({"exposure": exposure, "used": used, "issued": issued, "rate": rate},
                        index=panel.index[1:])


def multi_image_index(panel: pd.DataFrame, alive_on_maturity: bool, alpha: float, tenor: int,
                      settings: Settings = SETTINGS) -> MultiImageIndex:
    dep = daily_depreciation(panel, alive_on_maturity)
    if dep.empty or not np.isfinite(dep["rate"]).any():
        return MultiImageIndex(alpha, tenor, "runoff", bin_days=settings.dtm_bin_days)
    dep["rate_smooth"] = ewma(dep["rate"].to_numpy(), alpha)
    dep["index"] = 100 * np.cumprod(1 - dep["rate"].fillna(0))
    dep["index_smooth"] = 100 * np.cumprod(1 - dep["rate_smooth"].fillna(0))

    # relative shape by days-to-maturity over the lookback window
    window = panel[panel.index >= panel.index[-1] - pd.Timedelta(days=settings.lookback_days)]
    vals = window.to_numpy()
    a, b = vals[:-1], vals[1:]
    mats = window.columns.values[None, :]
    live = (a > 0) & (mats >= window.index.values[1:, None] if alive_on_maturity else mats > window.index.values[1:, None])
    used = np.clip(a - b, 0, None)[live]
    exposure = a[live]
    dtm = ((mats - window.index.values[:-1, None]) // np.timedelta64(1, "D"))[live]
    bins = (np.maximum(dtm, 0) // settings.dtm_bin_days).astype(int)
    used_b, exp_b, n_b = np.bincount(bins, weights=used), np.bincount(bins, weights=exposure), np.bincount(bins)
    global_rate = used.sum() / exposure.sum() if exposure.sum() > 0 else 0.0
    ok = (n_b >= settings.min_bin_transitions) & (exp_b > 0) & (global_rate > 0)
    mult = np.divide(used_b, exp_b * global_rate, out=np.ones_like(used_b), where=ok)
    shape = pd.Series(mult, index=np.arange(len(n_b)) * settings.dtm_bin_days)[ok]
    recent = dep[dep.index > dep.index[-1] - pd.Timedelta(days=settings.issuance_window_days)]
    return MultiImageIndex(alpha, tenor, "index", dep, shape, settings.dtm_bin_days,
                           float(dep["rate_smooth"].iloc[-1]), float(recent["issued"].mean()), int(live.sum()))


# --------------------------------------------------------------------------- projection
def project(image: pd.Series, asof, rate_for: Callable[[np.ndarray], np.ndarray], issuance_per_day: float,
            tenor: int, alive_on_maturity: bool, settings: Settings = SETTINGS):
    """Project an image forward day by day.

    Returns
      total    DataFrame(target_date, total, existing, new_issues, used, matured)
      buckets  DataFrame(maturity_date, balance_asof, expected_at_maturity, expected_used)
    """
    asof = pd.Timestamp(asof).normalize()
    image = image[(image > 0) & is_alive(image.index.values, asof, alive_on_maturity)]
    mats = image.index.values.astype("datetime64[ns]")
    bal = image.values.astype(float).copy()
    n_existing = len(bal)
    at_maturity = np.full(n_existing, np.nan)
    issue_new = settings.include_new_issuance and issuance_per_day > 0
    rows = []
    for step in range(1, settings.horizon_days + 1):
        prev_day = asof + pd.Timedelta(days=step - 1)
        day = asof + pd.Timedelta(days=step)
        dtm = ((mats - np.datetime64(prev_day)) // np.timedelta64(1, "D")).astype(int)
        used = bal * rate_for(dtm)
        bal = bal - used
        expiring = ~is_alive(mats, day, alive_on_maturity) & (bal > 0)
        matured = float(bal[expiring].sum())
        exp_idx = np.where(expiring[:n_existing])[0]
        at_maturity[exp_idx] = bal[exp_idx]
        bal[expiring] = 0.0
        if issue_new:
            mats = np.append(mats, np.datetime64(day + pd.Timedelta(days=tenor), "ns"))
            bal = np.append(bal, issuance_per_day)
        rows.append({"target_date": day, "total": float(bal.sum()), "existing": float(bal[:n_existing].sum()),
                     "new_issues": float(bal[n_existing:].sum()), "used": float(used.sum()), "matured": matured})
    total = pd.DataFrame(rows)
    still = np.isnan(at_maturity)
    at_maturity[still] = bal[:n_existing][still]
    buckets = pd.DataFrame({"maturity_date": image.index, "balance_asof": image.values,
                            "expected_at_maturity": at_maturity})
    buckets["expected_used"] = buckets["balance_asof"] - buckets["expected_at_maturity"]
    return total, buckets


@dataclass
class ForecastResult:
    asof: pd.Timestamp
    alpha: float
    single: SingleImageIndex
    multi: MultiImageIndex
    totals: pd.DataFrame        # long, with column "method"
    buckets: pd.DataFrame       # long, with column "method"
    meta: dict


def run_forecasts(panel: pd.DataFrame, alive_on_maturity: bool, snapshots: pd.DataFrame,
                  settings: Settings = SETTINGS, alpha: float | None = None, asof=None) -> ForecastResult:
    """Both forecasts (single image and multiple images) made with data up to `asof`."""
    alpha = settings.alpha if alpha is None else alpha
    asof = panel.index[-1] if asof is None else pd.Timestamp(asof).normalize()
    panel = panel[panel.index <= asof]
    snaps = snapshots[snapshots["snapshot_date"] <= asof]
    tenor = estimate_tenor(snaps, settings)
    image = panel.loc[asof]
    single = single_image_index(image, asof, alive_on_maturity, alpha, tenor, settings)
    multi = multi_image_index(panel, alive_on_maturity, alpha, tenor, settings)
    totals, buckets = [], []
    for name, idx in (("single", single), ("multi", multi)):
        t, b = project(image, asof, idx.rate_for, idx.issuance_per_day, tenor, alive_on_maturity, settings)
        totals.append(t.assign(method=name))
        buckets.append(b.assign(method=name))
    totals, buckets = pd.concat(totals, ignore_index=True), pd.concat(buckets, ignore_index=True)
    unused = buckets.groupby("method")["expected_at_maturity"].sum()
    meta = {
        "alpha": alpha, "tenor_days": tenor, "n_snapshots": int(snaps["snapshot_date"].nunique()),
        "balance_asof": float(image.sum()),
        "single_model": single.kind, "single_daily_rate": single.daily_rate,
        "single_issuance_per_day": single.issuance_per_day, "single_expected_unused": float(unused.get("single", 0.0)),
        "multi_model": multi.kind, "multi_daily_rate": multi.level,
        "multi_daily_rate_raw": float(multi.series["rate"].iloc[-1]) if multi.kind == "index" else 0.0,
        "multi_issuance_per_day": multi.issuance_per_day, "multi_expected_unused": float(unused.get("multi", 0.0)),
        "n_transitions": multi.n_transitions,
    }
    return ForecastResult(asof, alpha, single, multi, totals, buckets, meta)


def with_options(settings: Settings, **kw) -> Settings:
    return replace(settings, **kw)


def calibrate_alpha(panel: pd.DataFrame, alive_on_maturity: bool, snapshots: pd.DataFrame,
                    observed_totals: pd.Series, alphas=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
                    n_points: int = 8, horizon: int = 60, min_history: int = 14,
                    settings: Settings = SETTINGS) -> pd.DataFrame:
    """Backtest: forecast from past images (only data up to each of them) and measure the
    error against images observed later, for each alpha and method."""
    days = list(observed_totals.index)
    cands = [d for d in days if (d - days[0]).days >= min_history and any(x > d for x in days)]
    if not cands:
        return pd.DataFrame(columns=["alpha", "method", "mape", "n"])
    origins = [cands[int(i)] for i in np.unique(np.linspace(0, len(cands) - 1, n_points).round())]
    short = replace(settings, horizon_days=horizon)
    rows = []
    for alpha in alphas:
        errs = {m: [] for m in METHODS}
        for a in origins:
            res = run_forecasts(panel, alive_on_maturity, snapshots, short, alpha=alpha, asof=a)
            for m, g in res.totals.groupby("method"):
                g = g.set_index("target_date")["total"]
                common = [d for d in observed_totals.index if a < d <= a + pd.Timedelta(days=horizon) and d in g.index]
                errs[m] += [abs(g[d] / observed_totals[d] - 1) * 100 for d in common]
        rows += [{"alpha": alpha, "method": m, "mape": float(np.mean(e)) if e else np.nan, "n": len(e)}
                 for m, e in errs.items()]
    return pd.DataFrame(rows)
