"""Synthetic files with the same layout as the source, for tests and demos."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd


def simulate(start="2025-01-01", days=200, warmup=400, seed=0, shock_day=None, shock_rate=0.15):
    """Title-level simulation. Returns {date: DataFrame of outstanding TIDIS titles}."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start) - pd.Timedelta(days=warmup)
    titles = {}   # isin -> [issue, maturity, balance]
    n = 0
    out = {}
    for k in range(warmup + days):
        d = t0 + pd.Timedelta(days=k)
        # usage: higher near maturity, and spikes on tax deadlines (days 8-12 of even months)
        deadline = d.month % 2 == 0 and 8 <= d.day <= 12
        for isin, (iss, mat, bal) in list(titles.items()):
            dtm = (mat - d).days
            if dtm < 0:
                del titles[isin]            # matured
                continue
            rate = 0.002 + 0.004 * (1 - dtm / 365)
            if deadline:
                rate *= 4
            if shock_day is not None and d == pd.Timestamp(start) + pd.Timedelta(days=shock_day):
                rate = shock_rate
            titles[isin][2] = bal * (1 - min(rate * rng.uniform(0.5, 1.5), 1))
        if d.weekday() < 5:                 # issuance on business days
            for _ in range(rng.integers(1, 3)):
                n += 1
                titles[f"COT{n:07d}"] = [d, d + pd.Timedelta(days=365), float(rng.lognormal(22, 0.6))]
        if d >= pd.Timestamp(start):
            out[d] = pd.DataFrame([(i, *v) for i, v in titles.items()],
                                  columns=["ISIN", "Fecha de emisión", "Fecha de vencimiento", "Saldo en circulación"])
    return out


def to_excel(titles: pd.DataFrame, day: pd.Timestamp, drop_balance_rows: int = 0) -> bytes:
    """Write a file like the real one: 3 title rows, header, TIDIS + other categories."""
    t = titles.copy()
    t["Categoría del Título"] = "TIDIS"
    tes = pd.DataFrame({"ISIN": ["COL17CT0001", "COL17CT0002"], "Fecha de emisión": [pd.Timestamp("2020-01-01")] * 2,
                        "Fecha de vencimiento": [pd.Timestamp("2030-01-01")] * 2,
                        "Saldo en circulación": [1e13, 2e13], "Categoría del Título": ["TES", "TES"]})
    t = pd.concat([t, tes], ignore_index=True)
    if drop_balance_rows:
        t.loc[:drop_balance_rows - 1, "Saldo en circulación"] = np.nan
    cols = ["ISIN", "Categoría del Título", "Fecha de emisión", "Fecha de vencimiento", "Saldo en circulación"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([["Características financieras títulos vigentes"], [f"Fecha de corte: {day:%d/%m/%Y}"], [""]]
                     ).to_excel(xw, header=False, index=False, startrow=0)
        t[cols].to_excel(xw, index=False, startrow=3)
    return buf.getvalue()
