"""Flags for significant changes. Every alert is *relative*: it compares a value with its
reference (previous value, smoothed level, previous forecast, other method) and both the
threshold and the text are expressed as a percentage of that reference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .forecast import METHODS, ewma

SEV_ORDER = {"alerta": 0, "advertencia": 1, "info": 2}
METRIC_LABELS = {
    "dias_imputados": "Días imputados", "filas_imputadas": "Títulos imputados",
    "imagen_sin_cambios": "Imagen sin cambios", "uso_inusual": "Uso inusual",
    "emision_inusual": "Emisión inusual", "cambio_saldo": "Cambio del saldo",
    "cambio_indice": "Cambio del índice de depreciación", "revision": "Revisión de la predicción",
    "revision_no_usado": "Revisión del saldo que vencerá sin usarse", "error_prediccion": "Error de la predicción",
    "discrepancia_metodos": "Discrepancia entre métodos",
}


def _cop(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def _sev(rel: float, threshold: float) -> str:
    return "alerta" if abs(rel) > 2 * threshold else "advertencia"


def _alert(date, category, severity, metric, value, reference, change_pct, message, suffix=""):
    key = f"{pd.Timestamp(date).date()}|{category}|{metric}{suffix}"
    return dict(date=pd.Timestamp(date), category=category, severity=severity, metric=metric,
                value=value, reference=reference, change_pct=change_pct, message=message, key=key)


def _rel(value: float, reference: float) -> float:
    return 100 * (value - reference) / abs(reference) if reference else np.nan


# --------------------------------------------------------------------------- data quality
def data_alerts(catalog: pd.DataFrame) -> list[dict]:
    out = []
    dates = list(catalog["snapshot_date"])
    for a, b in zip(dates[:-1], dates[1:]):
        gap = (b - a).days
        if gap > 1:
            share = 100 * (gap - 1) / gap
            out.append(_alert(b, "datos", "info", "dias_imputados", gap - 1, gap, share,
                              f"{gap - 1} día(s) sin imagen entre {a.date()} y {b.date()} "
                              f"({share:.0f}% del intervalo): imputados"))
    for _, r in catalog.iterrows():
        n, total = r.get("n_imputed_rows", 0), r.get("n_rows", 0)
        if n and n > 0:
            share = 100 * n / total if total else np.nan
            out.append(_alert(r["snapshot_date"], "datos", "advertencia", "filas_imputadas", n, total, share,
                              f"{int(n)} de {int(total)} títulos ({share:.1f}%) con saldo o vencimiento faltante: "
                              "completados con la imagen anterior"))
    for a, b in zip(catalog.index[:-1], catalog.index[1:]):
        if catalog.at[a, "sha256"] and catalog.at[a, "sha256"] == catalog.at[b, "sha256"]:
            out.append(_alert(catalog.at[b, "snapshot_date"], "datos", "info", "imagen_sin_cambios", 0.0, 0.0, 0.0,
                              f"Imagen idéntica a la del {catalog.at[a, 'snapshot_date'].date()} (0.0% de cambio): "
                              "¿la fuente no se actualizó?"))
    return out


# --------------------------------------------------------------------------- flows & index
def flow_alerts(intervals: pd.DataFrame, settings: Settings = SETTINGS) -> list[dict]:
    """Usage / issuance as % of the balance vs their smoothed level, and total balance change."""
    out = []
    if intervals.empty:
        return out
    iv = intervals.reset_index(drop=True)
    usage = 100 * iv["redeemed_per_day"] / iv["prev_total"]
    issue = 100 * iv["issued_per_day"] / iv["prev_total"]
    # level known *before* each interval (no look-ahead)
    usage_lvl = pd.Series(ewma(usage.to_numpy(), settings.alpha)).shift(1)
    issue_lvl = pd.Series(ewma(issue.to_numpy(), settings.alpha)).shift(1)
    for i, r in iv.iterrows():
        span = f"entre {r['prev_date'].date()} y {r['date'].date()}" if r["days"] > 1 else "en el día"
        chg = _rel(r["total"], r["prev_total"])
        if np.isfinite(chg) and abs(chg) > settings.balance_change_pct:
            out.append(_alert(r["date"], "flujos", _sev(chg, settings.balance_change_pct), "cambio_saldo",
                              r["total"], r["prev_total"], chg,
                              f"El saldo total cambió {chg:+.1f}% {span}: {_cop(r['prev_total'])} → {_cop(r['total'])} "
                              f"(usado {_cop(r['redeemed'])}, emitido {_cop(r['issued'])}, vencido {_cop(r['matured'])})"))
        if i < settings.min_history_flow_alerts:
            continue
        for metric, val, lvl, thr, label, amount in (
                ("uso_inusual", usage[i], usage_lvl[i], settings.usage_rel_pct, "Uso diario", r["redeemed"]),
                ("emision_inusual", issue[i], issue_lvl[i], settings.issuance_rel_pct, "Emisión diaria", r["issued"])):
            rel = _rel(val, lvl)
            if np.isfinite(rel) and rel > thr:
                out.append(_alert(r["date"], "flujos", _sev(rel, thr), metric, val, lvl, rel,
                                  f"{label} de {val:.2f}% del saldo, {rel:+.0f}% sobre su nivel suavizado "
                                  f"({lvl:.2f}%/día); {_cop(amount)} {span}"))
    return out


def index_alerts(series: pd.DataFrame, observed_dates, alpha: float, settings: Settings = SETTINGS) -> list[dict]:
    """Shifts of the smoothed depreciation rate (multi-image index) between observed images."""
    out = []
    if series is None or series.empty or "rate_smooth" not in series:
        return out
    obs = [d for d in observed_dates if d in series.index]
    for k, (a, b) in enumerate(zip(obs[:-1], obs[1:])):
        if k + 1 < settings.min_history_flow_alerts:
            continue
        ra, rb = series.at[a, "rate_smooth"], series.at[b, "rate_smooth"]
        rel = _rel(rb, ra)
        if np.isfinite(rel) and abs(rel) > settings.index_change_pct:
            out.append(_alert(b, "índice", _sev(rel, settings.index_change_pct), "cambio_indice", 100 * rb, 100 * ra, rel,
                              f"El ritmo de depreciación suavizado (α={alpha:g}) pasó de {100 * ra:.3f}%/día a "
                              f"{100 * rb:.3f}%/día ({rel:+.0f}%); índice: {series.at[b, 'index_smooth']:.1f}"))
    return out


# --------------------------------------------------------------------------- forecasts
def _vget(v: pd.DataFrame, asof, col, default=np.nan):
    return v.at[asof, col] if col in v.columns and pd.notna(v.at[asof, col]) else default


def forecast_alerts(vintages: pd.DataFrame, totals: pd.DataFrame, actual_obs: pd.Series,
                    settings: Settings = SETTINGS) -> list[dict]:
    out = []
    if vintages.empty or totals.empty:
        return out
    v = vintages.set_index("asof")
    paths = {(a, m): g.set_index("target_date")["total"] for (a, m), g in totals.groupby(["asof", "method"])}
    asofs = sorted(set(a for a, _ in paths))
    for prev, cur in zip(asofs[:-1], asofs[1:]):
        if _vget(v, prev, "n_snapshots", 0) < settings.min_history_forecast_alerts:
            continue
        for method, mname in METHODS.items():
            if (prev, method) not in paths or (cur, method) not in paths:
                continue
            if _vget(v, prev, f"{method}_model", "index") == "runoff":
                continue
            p, c = paths[(prev, method)], paths[(cur, method)]
            # one alert per method and day, listing every horizon; the largest revision sets the severity
            revs = []
            for h in settings.check_horizons:
                t = cur + pd.Timedelta(days=h)
                if t in p.index and t in c.index and p[t] > 0:
                    revs.append((h, t, _rel(c[t], p[t])))
            big = [r for r in revs if abs(r[2]) > settings.forecast_change_pct]
            if big:
                h, t, rel = max(big, key=lambda r: abs(r[2]))
                detail = ", ".join(f"+{hh} d {rr:+.1f}%" for hh, _, rr in revs)
                out.append(_alert(cur, "predicción", _sev(rel, settings.forecast_change_pct), "revision",
                                  c[t], p[t], rel,
                                  f"[{mname}] Revisión frente a la predicción del {prev.date()}: {detail}. "
                                  f"Mayor cambio a +{h} d ({t.date()}): {_cop(p[t])} → {_cop(c[t])}",
                                  suffix=f"_{method}"))
            pu, cu = _vget(v, prev, f"{method}_expected_unused"), _vget(v, cur, f"{method}_expected_unused")
            rel = _rel(cu, pu) if np.isfinite(pu) and np.isfinite(cu) else np.nan
            if np.isfinite(rel) and abs(rel) > settings.unused_change_pct:
                out.append(_alert(cur, "predicción", _sev(rel, settings.unused_change_pct), "revision_no_usado", cu, pu, rel,
                                  f"[{mname}] El saldo que vencerá sin usarse cambió {rel:+.1f}%: "
                                  f"{_cop(pu)} → {_cop(cu)}", suffix=f"_{method}"))
            if cur in actual_obs.index and cur in p.index and actual_obs[cur] > 0:
                rel = _rel(p[cur], actual_obs[cur])
                if abs(rel) > settings.forecast_error_pct:
                    out.append(_alert(cur, "error", _sev(rel, settings.forecast_error_pct), "error_prediccion",
                                      p[cur], actual_obs[cur], rel,
                                      f"[{mname}] La predicción del {prev.date()} para este día se desvió {rel:+.1f}% "
                                      f"del saldo observado: {_cop(p[cur])} vs {_cop(actual_obs[cur])}",
                                      suffix=f"_{method}"))
    # single vs multi: flag when the gap at +90 d crosses the threshold
    h = 90
    prev_gap = None
    for a in asofs:
        if (a, "single") not in paths or (a, "multi") not in paths:
            continue
        t = a + pd.Timedelta(days=h)
        s, m = paths[(a, "single")], paths[(a, "multi")]
        if t not in s.index or t not in m.index or m[t] <= 0:
            continue
        gap = _rel(s[t], m[t])
        ready = _vget(v, a, "n_snapshots", 0) >= settings.min_history_forecast_alerts
        if ready and abs(gap) > settings.method_gap_pct and (prev_gap is None or abs(prev_gap) <= settings.method_gap_pct):
            out.append(_alert(a, "predicción", "advertencia", "discrepancia_metodos", s[t], m[t], gap,
                              f"Los métodos discrepan {gap:+.1f}% a +{h} días ({t.date()}): una imagen {_cop(s[t])} "
                              f"vs varias imágenes {_cop(m[t])}"))
        prev_gap = gap if ready else None
    return out


def compute_alerts(catalog, intervals, index_series, alpha, vintages, totals, actual_obs,
                   settings: Settings = SETTINGS) -> pd.DataFrame:
    rows = (data_alerts(catalog) + flow_alerts(intervals, settings)
            + index_alerts(index_series, list(catalog["snapshot_date"]), alpha, settings)
            + forecast_alerts(vintages, totals, actual_obs, settings))
    cols = ["date", "category", "severity", "metric", "value", "reference", "change_pct", "message", "key"]
    return pd.DataFrame(rows, columns=cols)


def forecast_accuracy(totals: pd.DataFrame, actual_obs: pd.Series) -> pd.DataFrame:
    """Error of every saved forecast against what was later observed, by method and horizon."""
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
