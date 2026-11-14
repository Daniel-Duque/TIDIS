"""Flags for significant changes: in the data, in the flows and in the predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import robust_z
from .config import SETTINGS, Settings

SEV_ORDER = {"alerta": 0, "advertencia": 1, "info": 2}


def _fmt(x: float) -> str:
    return f"${x:,.0f}"


def _alert(date, category, severity, metric, value, reference, message):
    key = f"{pd.Timestamp(date).date()}|{category}|{metric}"
    return dict(date=pd.Timestamp(date), category=category, severity=severity, metric=metric,
                value=value, reference=reference, message=message, key=key)


def data_alerts(catalog: pd.DataFrame) -> list[dict]:
    out = []
    dates = list(catalog["snapshot_date"])
    for a, b in zip(dates[:-1], dates[1:]):
        gap = (b - a).days
        if gap > 1:
            out.append(_alert(b, "datos", "info", "dias_imputados", gap - 1, np.nan,
                              f"{gap - 1} día(s) sin imagen entre {a.date()} y {b.date()}: imputados"))
    for _, r in catalog.iterrows():
        if r.get("n_imputed_rows", 0) and r["n_imputed_rows"] > 0:
            out.append(_alert(r["snapshot_date"], "datos", "advertencia", "filas_imputadas", r["n_imputed_rows"], np.nan,
                              f"{int(r['n_imputed_rows'])} título(s) con campos faltantes completados"))
    for a, b in zip(catalog.index[:-1], catalog.index[1:]):
        if catalog.at[a, "sha256"] and catalog.at[a, "sha256"] == catalog.at[b, "sha256"]:
            out.append(_alert(catalog.at[b, "snapshot_date"], "datos", "info", "imagen_sin_cambios", np.nan, np.nan,
                              "La imagen de TIDIS es idéntica a la anterior (¿la fuente no se actualizó?)"))
    return out


def flow_alerts(intervals: pd.DataFrame, settings: Settings = SETTINGS) -> list[dict]:
    """Unusual usage or issuance compared with the recent history (robust z-score)."""
    out = []
    labels = {"redeemed_per_day": ("uso_inusual", "Uso (redención) de TIDIS"),
              "issued_per_day": ("emision_inusual", "Emisión de TIDIS")}
    for i in range(len(intervals)):
        hist = intervals.iloc[max(0, i - 60):i]
        if len(hist) < settings.min_history_flow_alerts:
            continue
        r = intervals.iloc[i]
        for col, (metric, label) in labels.items():
            z = robust_z(r[col], hist[col])
            if np.isfinite(z) and z > settings.z_threshold:
                sev = "alerta" if z > 2 * settings.z_threshold else "advertencia"
                out.append(_alert(r["date"], "flujos", sev, metric, r[col], float(hist[col].median()),
                                  f"{label} de {_fmt(r[col])}/día vs mediana {_fmt(hist[col].median())}/día (z={z:.1f})"))
    return out


def forecast_alerts(vintages: pd.DataFrame, totals: pd.DataFrame, actual_obs: pd.Series,
                    settings: Settings = SETTINGS) -> list[dict]:
    """Revisions between consecutive forecasts, and errors of past forecasts."""
    out = []
    if vintages.empty or totals.empty:
        return out
    by_vintage = {a: g.set_index("target_date")["total"] for a, g in totals.groupby("asof")}
    v = vintages.set_index("asof")
    asofs = sorted(by_vintage)
    for prev, cur in zip(asofs[:-1], asofs[1:]):
        if v.at[prev, "n_snapshots"] < settings.min_history_forecast_alerts or v.at[prev, "model"] == "runoff":
            continue
        p, c = by_vintage[prev], by_vintage[cur]
        # revision of the forecast for the same target dates
        for h in settings.check_horizons:
            target = cur + pd.Timedelta(days=h)
            if target in p.index and target in c.index and p[target] > 0:
                chg = 100 * (c[target] - p[target]) / p[target]
                if abs(chg) > settings.forecast_change_pct:
                    sev = "alerta" if abs(chg) > 2 * settings.forecast_change_pct else "advertencia"
                    out.append(_alert(cur, "predicción", sev, f"revision_{h}d", c[target], p[target],
                                      f"La predicción para {target.date()} (+{h}d) cambió {chg:+.1f}%: "
                                      f"{_fmt(p[target])} → {_fmt(c[target])}"))
        # revision of the expected balance that will expire unused
        pu, cu = v.at[prev, "expected_unused_at_maturity"], v.at[cur, "expected_unused_at_maturity"]
        if pu > 0 and abs(100 * (cu - pu) / pu) > settings.forecast_change_pct:
            chg = 100 * (cu - pu) / pu
            out.append(_alert(cur, "predicción", "advertencia", "revision_no_usado", cu, pu,
                              f"Saldo esperado que vencerá sin usarse cambió {chg:+.1f}%: {_fmt(pu)} → {_fmt(cu)}"))
        # error of the previous forecast against the value observed today
        if cur in actual_obs.index and cur in p.index and actual_obs[cur] > 0:
            err = 100 * (p[cur] - actual_obs[cur]) / actual_obs[cur]
            if abs(err) > settings.forecast_error_pct:
                sev = "alerta" if abs(err) > 2 * settings.forecast_error_pct else "advertencia"
                out.append(_alert(cur, "error", sev, "error_prediccion", p[cur], actual_obs[cur],
                                  f"La predicción del {prev.date()} para hoy fue {_fmt(p[cur])}, "
                                  f"el saldo real es {_fmt(actual_obs[cur])} (error {err:+.1f}%)"))
    return out


def compute_alerts(catalog, intervals, vintages, totals, actual_obs, settings: Settings = SETTINGS) -> pd.DataFrame:
    rows = data_alerts(catalog) + flow_alerts(intervals, settings) + forecast_alerts(vintages, totals, actual_obs, settings)
    cols = ["date", "category", "severity", "metric", "value", "reference", "message", "key"]
    return pd.DataFrame(rows, columns=cols)


def forecast_accuracy(totals: pd.DataFrame, actual_obs: pd.Series) -> pd.DataFrame:
    """Error of every saved forecast against what was later observed, by horizon."""
    if totals.empty or actual_obs.empty:
        return pd.DataFrame()
    t = totals[totals["target_date"].isin(actual_obs.index)].copy()
    if t.empty:
        return pd.DataFrame()
    t["actual"] = t["target_date"].map(actual_obs)
    t["horizon"] = (t["target_date"] - t["asof"]).dt.days
    t["error_pct"] = 100 * (t["total"] - t["actual"]) / t["actual"]
    t["horizon_bin"] = pd.cut(t["horizon"], [0, 7, 30, 90, 180, 366],
                              labels=["1-7 d", "8-30 d", "31-90 d", "91-180 d", "181-365 d"])
    return t
