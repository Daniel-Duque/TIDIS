"""Imputation: (1) missing fields inside a snapshot, (2) missing days between snapshots.

The daily *panel* is the central object of the app: rows = calendar days,
columns = maturity dates ("buckets"), values = outstanding TIDIS balance.
Each observed snapshot is one row of the panel; the rows in between are imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings


# --------------------------------------------------------------------------- rows
def impute_rows(rows: pd.DataFrame, prev: pd.DataFrame | None, settings: Settings = SETTINGS):
    """Fill missing maturity / balance. Returns (rows, n_imputed, n_dropped)."""
    rows = rows.copy()
    rows["imputed"] = rows["imputed"].astype(bool)
    has_id = rows["title_id"].notna() & (rows["title_id"].astype(str) != "") & (rows["title_id"].astype(str) != "<NA>")
    prev_by_id = None
    if prev is not None and prev["title_id"].notna().any():
        prev_by_id = prev.dropna(subset=["title_id"]).drop_duplicates("title_id").set_index("title_id")

    # maturity: previous snapshot of the same title, else issue date + tenor
    miss = rows["maturity_date"].isna()
    if miss.any() and prev_by_id is not None:
        fill = rows.loc[miss & has_id, "title_id"].map(prev_by_id["maturity_date"])
        rows.loc[fill.dropna().index, "maturity_date"] = fill.dropna()
        rows.loc[fill.dropna().index, "imputed"] = True
    miss = rows["maturity_date"].isna() & rows["issue_date"].notna()
    rows.loc[miss, "maturity_date"] = rows.loc[miss, "issue_date"] + pd.Timedelta(days=settings.tenor_days)
    rows.loc[miss, "imputed"] = True

    # balance: previous snapshot of the same title, else previous balance of the bucket
    miss = rows["balance"].isna()
    if miss.any() and prev_by_id is not None:
        fill = rows.loc[miss & has_id, "title_id"].map(prev_by_id["balance"])
        rows.loc[fill.dropna().index, "balance"] = fill.dropna()
        rows.loc[fill.dropna().index, "imputed"] = True
    miss = rows["balance"].isna() & rows["maturity_date"].notna()
    if miss.any() and prev is not None:
        prev_bucket = prev.groupby("maturity_date")["balance"].sum()
        known = rows[~rows["balance"].isna()].groupby("maturity_date")["balance"].sum()
        for i in rows.index[miss]:
            m = rows.at[i, "maturity_date"]
            # only safe if this is the only unknown row of the bucket
            if (rows.loc[miss, "maturity_date"] == m).sum() == 1 and m in prev_bucket.index:
                rows.at[i, "balance"] = max(prev_bucket[m] - known.get(m, 0.0), 0.0)
                rows.at[i, "imputed"] = True

    bad = rows["maturity_date"].isna() | rows["balance"].isna()
    rows = rows[~bad].reset_index(drop=True)
    return rows, int(rows["imputed"].sum()), int(bad.sum())


# --------------------------------------------------------------------------- panel
def observed_matrix(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Observed snapshots as a matrix: index = snapshot date, columns = maturity date."""
    if snapshots.empty:
        return pd.DataFrame()
    m = snapshots.pivot_table(index="snapshot_date", columns="maturity_date", values="balance",
                              aggfunc="sum", fill_value=0.0)
    m.columns = pd.DatetimeIndex(m.columns)
    return m.sort_index().sort_index(axis=1).astype(float)


def detect_alive_on_maturity(obs: pd.DataFrame) -> bool:
    """Does the source still list a title on its maturity date? Learned from the data."""
    listed, dropped = 0, 0
    dates = list(obs.index)
    for i, d in enumerate(dates):
        if d in obs.columns:
            if obs.at[d, d] > 0:
                listed += 1
            elif i > 0 and obs.at[dates[i - 1], d] > 0:
                dropped += 1
    return listed >= dropped  # default (no evidence): still listed on maturity day


def is_alive(maturities: np.ndarray, day: pd.Timestamp, alive_on_maturity: bool) -> np.ndarray:
    return maturities >= day if alive_on_maturity else maturities > day


def issue_dates_by_bucket(snapshots: pd.DataFrame, settings: Settings = SETTINGS) -> pd.Series:
    """Estimated issue date per maturity bucket (reported issue date, else maturity - tenor)."""
    mats = pd.DatetimeIndex(sorted(snapshots["maturity_date"].dropna().unique()))
    est = pd.Series(mats - pd.Timedelta(days=settings.tenor_days), index=mats)
    if snapshots["issue_date"].notna().any():
        rep = snapshots.dropna(subset=["issue_date"]).groupby("maturity_date")["issue_date"].min()
        est.loc[rep.index] = rep
    return est


def build_panel(snapshots: pd.DataFrame, settings: Settings = SETTINGS):
    """Daily panel with imputed gaps.

    Returns (panel, imputed_flag, alive_on_maturity).
    """
    obs = observed_matrix(snapshots)
    if obs.empty:
        return obs, pd.Series(dtype=bool), True
    aom = detect_alive_on_maturity(obs)
    issue = issue_dates_by_bucket(snapshots, settings).reindex(obs.columns)
    mats = obs.columns.values
    days = pd.date_range(obs.index[0], obs.index[-1], freq="D")
    panel = obs.reindex(days)
    imputed = pd.Series(~days.isin(obs.index), index=days)

    for a_day, b_day in zip(obs.index[:-1], obs.index[1:]):
        gap = (b_day - a_day).days
        if gap <= 1:
            continue
        a, b = obs.loc[a_day].values, obs.loc[b_day].values
        matured = ~is_alive(mats, b_day, aom) & (b == 0) & (a > 0)
        new = (a == 0) & (b > 0)
        step_issue = new & (issue.values > a_day) & (issue.values < b_day)
        for k in range(1, gap):
            d = a_day + pd.Timedelta(days=k)
            frac = k / gap
            vals = a + (b - a) * frac if settings.imputation == "linear" else a.copy()
            vals = np.where(matured, np.where(is_alive(mats, d, aom), a, 0.0), vals)
            vals = np.where(step_issue, np.where(issue.values <= d, b, 0.0), vals)
            panel.loc[d] = vals
    return panel.astype(float), imputed, aom
