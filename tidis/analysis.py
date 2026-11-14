"""Compare images (snapshots) and decompose how the TIDIS stock changes.

Any change between two images is split into
    issued    new titles / increases of a maturity bucket
    redeemed  titles used (tax payments) before maturity -> the "degradation"
    matured   balance that disappeared because the title reached maturity (unused)
so that  total_B - total_A = issued - redeemed - matured  (exactly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .imputation import is_alive


def decompose(a: pd.Series, b: pd.Series, b_day: pd.Timestamp, alive_on_maturity: bool) -> dict:
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    delta = (b - a).values
    matured_mask = (delta < 0) & (b.values == 0) & ~is_alive(idx.values, b_day, alive_on_maturity)
    return {
        "issued": float(delta[delta > 0].sum()),
        "redeemed": float(-delta[(delta < 0) & ~matured_mask].sum()),
        "matured": float(-delta[matured_mask].sum()),
    }


def daily_flows(panel: pd.DataFrame, imputed: pd.Series, alive_on_maturity: bool) -> pd.DataFrame:
    """Total outstanding and its decomposition for every day of the panel."""
    if panel.empty:
        return pd.DataFrame(columns=["total", "issued", "redeemed", "matured", "imputed"])
    rows = []
    for i, d in enumerate(panel.index):
        r = {"date": d, "total": float(panel.loc[d].sum()), "imputed": bool(imputed.loc[d])}
        if i == 0:
            r.update(issued=np.nan, redeemed=np.nan, matured=np.nan)
        else:
            r.update(decompose(panel.iloc[i - 1], panel.loc[d], d, alive_on_maturity))
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def observed_intervals(obs: pd.DataFrame, alive_on_maturity: bool) -> pd.DataFrame:
    """Flows between consecutive *observed* snapshots, also expressed per day."""
    rows = []
    for a_day, b_day in zip(obs.index[:-1], obs.index[1:]):
        days = (b_day - a_day).days
        f = decompose(obs.loc[a_day], obs.loc[b_day], b_day, alive_on_maturity)
        prev_total = float(obs.loc[a_day].sum())
        rows.append({"date": b_day, "prev_date": a_day, "days": days, **f,
                     "redeemed_per_day": f["redeemed"] / days, "issued_per_day": f["issued"] / days,
                     "prev_total": prev_total, "total": float(obs.loc[b_day].sum())})
    return pd.DataFrame(rows)


def compare_images(a: pd.Series, b: pd.Series, b_day, alive_on_maturity: bool) -> pd.DataFrame:
    """Daily comparison of two images: one row per maturity date, with the change decomposed."""
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    delta = (b - a).to_numpy()
    matured = (delta < 0) & (b.to_numpy() == 0) & ~is_alive(idx.values, pd.Timestamp(b_day), alive_on_maturity)
    out = pd.DataFrame({
        "balance_a": a.to_numpy(), "balance_b": b.to_numpy(),
        "issued": np.clip(delta, 0, None),
        "redeemed": np.where(~matured, np.clip(-delta, 0, None), 0.0),
        "matured": np.where(matured, -delta, 0.0),
    }, index=pd.DatetimeIndex(idx, name="maturity_date"))
    out["change"] = out["balance_b"] - out["balance_a"]
    out["change_pct"] = 100 * out["change"] / out["balance_a"].where(out["balance_a"] > 0)
    return out[(out["balance_a"] > 0) | (out["balance_b"] > 0)]
