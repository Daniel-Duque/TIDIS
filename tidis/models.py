"""Projection models for the drivers of the multi-image forecast.

Every model receives the daily history of the two drivers of the TIDIS stock
    rate    fraction of the live balance used (redeemed) that day   -> depreciation
    issued  new TIDIS issued that day (COP)
and returns their daily path over the horizon. The path then feeds the same projection by
maturity date (depreciation, maturity, new issuance), so every model respects the known
maturity schedule. "Suavizado α" (the default) keeps the smoothed level flat, exactly as before.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings

DRIVERS = ["rate", "issued"]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    description: str
    min_days: int
    uses_alpha: bool
    requires: str | None        # optional package


# --------------------------------------------------------------------------- helpers
def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical calendar features: tax deadlines depend on the day of the month and the month."""
    d = pd.DatetimeIndex(dates)
    dom, mon, dow = d.day.values, d.month.values, d.dayofweek.values
    return np.column_stack([np.sin(2 * np.pi * dom / 31), np.cos(2 * np.pi * dom / 31),
                            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
                            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float)])


def _prepare(hist: pd.DataFrame) -> pd.DataFrame:
    y = hist[DRIVERS].astype(float).copy()
    y["rate"] = y["rate"].ffill().fillna(0.0)
    y["issued"] = y["issued"].fillna(0.0)
    return y


def _scale(y: pd.DataFrame, clip: float = 10.0):
    """Robust standardisation (median / MAD) with winsorising: a one-off shock (e.g. a day with
    15% usage) must not squash the normal variation nor dominate the fit."""
    med = y.median().values
    mad = 1.4826 * (y - med).abs().median().values
    sd = y.std().fillna(1).values
    scale = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
    z = np.clip((y.values - med) / scale, -clip, clip)
    return z, med, scale


def _future_dates(hist: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _finish(hist: pd.DataFrame, values: np.ndarray, horizon: int) -> pd.DataFrame:
    """Clip to plausible ranges (no negative usage/issuance, no explosions)."""
    y = _prepare(hist)
    out = pd.DataFrame(np.asarray(values, dtype=float).reshape(horizon, 2), columns=DRIVERS,
                       index=_future_dates(hist, horizon))
    for c in DRIVERS:
        hi = max(float(y[c].max()) * 1.5, 0.0)
        out[c] = out[c].clip(0, hi if hi > 0 else None)
    out["rate"] = out["rate"].clip(upper=0.5)
    return out


# --------------------------------------------------------------------------- models
def _ewma(hist, horizon, alpha, settings):
    from .forecast import ewma
    y = _prepare(hist)
    level = ewma(hist["rate"].to_numpy(dtype=float), alpha)[-1]
    recent = y[y.index > y.index[-1] - pd.Timedelta(days=settings.issuance_window_days)]
    return _finish(hist, np.column_stack([np.full(horizon, level), np.full(horizon, recent["issued"].mean())]), horizon)


def _holt(hist, horizon, alpha, settings):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = _prepare(hist)
    cols = []
    for c in DRIVERS:
        fit = ExponentialSmoothing(y[c].values, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit(smoothing_level=alpha, optimized=True)
        cols.append(fit.forecast(horizon))
    return _finish(hist, np.column_stack(cols), horizon)


def _sarima(hist, horizon, alpha, settings):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = _prepare(hist)
    z, mu, sd = _scale(y)
    cols = []
    for j in range(2):
        res = SARIMAX(z[:, j], order=(1, 0, 1), seasonal_order=(1, 0, 0, 7), trend="c").fit(disp=False, maxiter=100)
        cols.append(res.forecast(horizon) * sd[j] + mu[j])
    return _finish(hist, np.column_stack(cols), horizon)


def _var(hist, horizon, alpha, settings):
    from statsmodels.tsa.api import VAR
    y = _prepare(hist)
    z, mu, sd = _scale(y)
    res = VAR(z).fit(maxlags=max(1, min(14, len(z) // 6)), ic="aic")
    k = res.k_ar
    fc = res.forecast(z[-k:], horizon) if k > 0 else np.tile(res.params[0], (horizon, 1))
    return _finish(hist, fc * sd + mu, horizon)


_LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)


def _lag_features(z: np.ndarray, t: int, date: pd.Timestamp) -> np.ndarray:
    lags = [z[t - l] for l in _LAGS]
    return np.concatenate(lags + [z[t - 7:t].mean(axis=0), z[t - 28:t].mean(axis=0),
                                  [date.day, date.month, date.dayofweek]])


def _gbm(hist, horizon, alpha, settings):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = _prepare(hist)
    z, mu, sd = _scale(y)
    dates = y.index
    X = np.array([_lag_features(z, t, dates[t]) for t in range(28, len(z))])
    models = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=8, random_state=0).fit(X, z[28:, j]) for j in range(2)]
    buf = list(z)
    for d in _future_dates(hist, horizon):
        f = _lag_features(np.array(buf), len(buf), d)[None, :]
        buf.append(np.array([m.predict(f)[0] for m in models]))
    return _finish(hist, np.array(buf[len(z):]) * sd + mu, horizon)


def _sequence_model(build, epochs, lr, val_frac=0.0):
    def fit_predict(hist, horizon, alpha, settings):
        from .nn import T, train
        y = _prepare(hist)
        z, mu, sd = _scale(y)
        dates = y.index
        W = int(min(28, max(7, len(z) // 4)))
        # position i carries the values of day i and the calendar of day i+1 (the day to predict)
        cal = calendar_features(dates.append(_future_dates(hist, horizon)))
        feats = lambda zz, start: np.column_stack([zz[start:start + W], cal[start + 1:start + W + 1]])  # noqa: E731
        X = np.array([feats(z, j - W) for j in range(W, len(z))])
        Y = z[W:]
        model = build(X.shape[2], W)
        train(model, X, Y, epochs=epochs, lr=lr, val_frac=val_frac, seed=0)
        buf = np.vstack([z, np.zeros((horizon, 2))])
        for s in range(horizon):
            t = len(z) + s
            buf[t] = model(T(feats(buf, t - W)[None, :, :])).data[0]
        return _finish(hist, buf[len(z):] * sd + mu, horizon)
    return fit_predict


def _build_gru(n_in, window):
    from .nn import GRU
    return GRU(n_in, 2, hidden=24, seed=0)


def _build_transformer(n_in, window):
    from .nn import Transformer
    # small on purpose: with a few hundred daily observations bigger encoders overfit (backtested)
    return Transformer(n_in, 2, d_model=16, heads=2, layers=1, window=window, seed=0)


PROJECTION_MODELS: dict[str, ModelSpec] = {
    "ewma": ModelSpec("Suavizado α (EWMA)", "Nivel suavizado exponencialmente con α, constante hacia adelante.",
                      1, True, None),
    "holt": ModelSpec("Holt amortiguado", "Nivel suavizado con α más una tendencia que se amortigua.",
                      14, True, "statsmodels"),
    "sarima": ModelSpec("SARIMA semanal", "ARIMA(1,0,1) con estacionalidad semanal (1,0,0)₇, por serie.",
                        30, False, "statsmodels"),
    "var": ModelSpec("VAR", "Vector autorregresivo: uso y emisión se explican juntos (rezagos por AIC).",
                     30, False, "statsmodels"),
    "gbm": ModelSpec("Gradient boosting", "Árboles con rezagos (1-28 días) y calendario (día, mes, día de la semana).",
                     60, False, "sklearn"),
    "rnn": ModelSpec("RNN (GRU)", "Red recurrente GRU sobre ventanas de 28 días con calendario.",
                     60, False, None),
    "transformer": ModelSpec("Transformer", "Encoder (1 capa, atención de 2 cabezas) sobre ventanas de 28 días con calendario.",
                             60, False, None),
}

_FITTERS: dict[str, Callable] = {
    "ewma": _ewma, "holt": _holt, "sarima": _sarima, "var": _var, "gbm": _gbm,
    # fixed epochs, all windows used, final parameters (early stopping on the few most recent
    # windows was worse in the backtest)
    "rnn": _sequence_model(_build_gru, epochs=60, lr=0.01),
    "transformer": _sequence_model(_build_transformer, epochs=60, lr=0.005),
}


def is_available(key: str) -> bool:
    req = PROJECTION_MODELS[key].requires
    if req is None:
        return True
    try:
        __import__(req)
        return True
    except ImportError:
        return False


def forecast_drivers(hist: pd.DataFrame, horizon: int, model: str = "ewma", alpha: float | None = None,
                     settings: Settings = SETTINGS) -> tuple[pd.DataFrame, str, str]:
    """Daily path of the drivers. Returns (path, model actually used, note).

    Falls back to the α smoothing when the model is unavailable, the history is too short or
    the fit fails - the page says so instead of failing."""
    alpha = settings.alpha if alpha is None else alpha
    spec = PROJECTION_MODELS.get(model)
    note = ""
    if spec is None:
        model, note = "ewma", f"Modelo '{model}' desconocido: se usa el suavizado α."
    elif not is_available(model):
        model, note = "ewma", f"{spec.label} requiere el paquete '{spec.requires}': se usa el suavizado α."
    elif len(hist) < spec.min_days:
        model, note = "ewma", (f"{spec.label} necesita al menos {spec.min_days} días de historia "
                               f"(hay {len(hist)}): se usa el suavizado α.")
    if model != "ewma":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = _FITTERS[model](hist, horizon, alpha, settings)
            if np.isfinite(path.values).all():
                return path, model, note
            note = f"{PROJECTION_MODELS[model].label} produjo valores no válidos: se usa el suavizado α."
        except Exception as exc:                      # never break the page because of a model
            note = f"{PROJECTION_MODELS[model].label} falló ({type(exc).__name__}): se usa el suavizado α."
        model = "ewma"
    return _ewma(hist, horizon, alpha, settings), "ewma", note
