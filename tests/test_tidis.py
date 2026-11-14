"""Run with:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import simulate, to_excel  # noqa: E402
from tidis import pipeline, source  # noqa: E402
from tidis.analysis import daily_flows, decompose  # noqa: E402
from tidis.config import Settings  # noqa: E402
from tidis.forecast import (ewma, isotonic_decreasing, multi_image_index, run_forecasts,  # noqa: E402
                            single_image_index)
from tidis.imputation import build_panel, impute_rows  # noqa: E402
from tidis.storage import Store  # noqa: E402

D = pd.Timestamp


def snap(day, rows):
    """rows: list of (maturity, balance)"""
    return pd.DataFrame({"title_id": [f"T{i}" for i in range(len(rows))], "issue_date": pd.NaT,
                         "maturity_date": [D(m) for m, _ in rows], "balance": [b for _, b in rows],
                         "imputed": False, "snapshot_date": D(day)})


# ----------------------------------------------------------------------------- parsing
def test_parse_synthetic_file():
    sim = simulate(days=2, warmup=30)
    day = list(sim)[1]
    rows, cut = source.parse_file(to_excel(sim[day], day), reference=day.date())
    assert cut == day.date()
    assert len(rows) == len(sim[day])                        # TES rows filtered out
    assert rows["balance"].sum() == pytest.approx(sim[day]["Saldo en circulación"].sum())
    assert rows["maturity_date"].notna().all()


@pytest.mark.parametrize("txt,val", [("1.234.567,89", 1234567.89), ("1,234,567.89", 1234567.89),
                                     ("$ 1.234.567", 1234567), ("1234,5", 1234.5), (12.5, 12.5)])
def test_parse_number(txt, val):
    assert source._parse_number(txt) == pytest.approx(val)


# ----------------------------------------------------------------------------- imputation
def test_row_imputation_uses_previous_image_of_same_title():
    prev = snap("2025-01-01", [("2025-06-01", 100.0), ("2025-07-01", 50.0)])
    cur = snap("2025-01-02", [("2025-06-01", np.nan), ("2025-07-01", 40.0)])
    rows, n_imp, n_drop = impute_rows(cur.drop(columns="snapshot_date"), prev)
    assert n_imp == 1 and n_drop == 0
    assert rows.loc[0, "balance"] == 100.0 and rows.loc[0, "imputed"]


def test_gap_imputation_follows_title_life_cycle():
    a = snap("2025-01-01", [("2025-01-03", 10.0), ("2025-06-01", 100.0)])
    b = snap("2025-01-05", [("2025-06-01", 60.0), ("2026-01-03", 30.0)])   # new bucket issued ~Jan 3
    panel, imputed, aom = build_panel(pd.concat([a, b]), Settings(imputation="linear"))
    assert list(imputed) == [False, True, True, True, False]
    # maturing bucket: flat until maturity, then gone
    assert list(panel[D("2025-01-03")]) == [10, 10, 10, 0, 0]
    # continuing bucket: linear 100 -> 60
    assert list(panel[D("2025-06-01")]) == [100, 90, 80, 70, 60]
    # new bucket appears on its estimated issue date (maturity - 365 d = 2025-01-03)
    assert list(panel[D("2026-01-03")]) == [0, 0, 30, 30, 30]


def test_flow_decomposition_is_exact():
    a = pd.Series({D("2025-01-02"): 10.0, D("2025-03-01"): 100.0})
    b = pd.Series({D("2025-03-01"): 80.0, D("2026-01-01"): 25.0})
    f = decompose(a, b, D("2025-01-03"), True)
    assert f == {"issued": 25.0, "redeemed": 20.0, "matured": 10.0}
    assert b.sum() - a.sum() == f["issued"] - f["redeemed"] - f["matured"]


# ----------------------------------------------------------------------------- depreciation index & forecast
def test_single_image_with_few_dates_gives_contractual_runoff():
    s = snap("2025-01-01", [("2025-02-01", 100.0), ("2025-05-01", 50.0)])
    panel, _, aom = build_panel(s)
    res = run_forecasts(panel, aom, s, Settings(horizon_days=200))
    assert res.single.kind == "runoff" and res.multi.kind == "runoff"
    for _, g in res.totals.groupby("method"):
        t = g.set_index("target_date")["total"]
        assert t[D("2025-01-15")] == 150 and t[D("2025-03-01")] == 50 and t[D("2025-06-01")] == 0


def test_single_image_index_recovers_known_depreciation():
    rate, asof = 0.004, D("2025-01-01")
    # one title per day of issue; a title of age `a` has kept (1-rate)^a of its issue amount
    rows = [(asof + pd.Timedelta(days=365 - a), 1000 * (1 - rate) ** a) for a in range(0, 366)]
    image = pd.Series({m: b for m, b in rows})
    idx = single_image_index(image, asof, True, alpha=0.1, tenor=365)
    assert idx.kind == "index"
    assert idx.profile["index_mono"].iloc[0] == pytest.approx(100)
    assert idx.daily_rate == pytest.approx(rate, rel=1e-3)
    assert np.allclose(idx.rate_for(np.arange(10, 300)), rate, rtol=1e-3)
    assert idx.issuance_per_day == pytest.approx(1000, rel=1e-4)


def test_multi_image_index_recovers_known_rate():
    rate = 0.004
    days = pd.date_range("2025-01-01", periods=60)
    mats = [D("2025-08-01"), D("2025-10-01"), D("2025-12-01")]
    s = pd.concat([snap(d, [(m, 1000 * (1 - rate) ** i) for m in mats]) for i, d in enumerate(days)])
    panel, _, aom = build_panel(s)
    idx = multi_image_index(panel, aom, alpha=0.3, tenor=365, settings=Settings(min_bin_transitions=5))
    assert idx.kind == "index"
    assert idx.level == pytest.approx(rate, rel=1e-9)
    assert idx.series["index"].iloc[-1] == pytest.approx(100 * (1 - rate) ** 59)
    assert idx.issuance_per_day == 0


def test_alpha_softens_the_trend():
    rng = np.random.default_rng(0)
    x = 0.005 + rng.normal(0, 0.002, 300)
    assert np.allclose(ewma(x, 1.0), x)                       # alpha = 1: no smoothing
    assert np.std(ewma(x, 0.05)[50:]) < np.std(ewma(x, 0.5)[50:]) < np.std(x)


def test_isotonic_fit_is_non_increasing():
    y = np.array([5, 6, 4, 4.5, 3, 3.2, 1.0])
    z = isotonic_decreasing(y)
    assert np.all(np.diff(z) <= 0) and z.mean() == pytest.approx(y.mean())


# ----------------------------------------------------------------------------- storage + pipeline
@pytest.fixture(scope="module")
def history(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("data"))
    sim = simulate(days=70, warmup=370, seed=3, shock_day=60, shock_rate=0.2)
    for i, d in enumerate(sim):
        if 30 <= i < 33:
            continue                                      # outage -> imputed days
        pipeline.ingest(store, to_excel(sim[d], d), day=d.date())
    _, h = pipeline.update(store, fetch=False)
    return store, sim, h


def test_history_matches_source_on_observed_days(history):
    store, sim, h = history
    for d in h.obs.index:
        assert h.panel.loc[d].sum() == pytest.approx(sim[d]["Saldo en circulación"].sum())
    assert h.imputed.sum() == 3


def test_daily_flows_reconcile_with_total(history):
    _, _, h = history
    f = h.flows
    net = (f["issued"] - f["redeemed"] - f["matured"]).iloc[1:]
    assert np.allclose(f["total"].diff().iloc[1:], net)


def test_same_file_is_not_saved_twice_and_vintages_are_write_once(history):
    store, sim, h = history
    d = h.obs.index[-1]
    before = store.vintages().set_index("asof").loc[d, "created_at"]
    assert pipeline.ingest(store, to_excel(sim[d], d), day=d.date())["status"] == "unchanged"
    pipeline.refresh_forecasts(store)
    assert store.vintages().set_index("asof").loc[d, "created_at"] == before
    assert len(store.vintages()) == len(h.obs)


def test_both_methods_are_saved_every_day(history):
    store, sim, h = history
    assert set(h.fc_totals["method"]) == {"single", "multi"}
    assert {"single_expected_unused", "multi_expected_unused", "alpha"} <= set(h.vintages.columns)


def test_shock_is_flagged_with_relative_changes(history):
    store, sim, h = history
    shock = list(sim)[60]
    flagged = h.alerts[h.alerts["date"] == shock]
    assert {"uso_inusual", "cambio_saldo", "error_prediccion", "cambio_indice"} <= set(flagged["metric"])
    assert (flagged["severity"] == "alerta").any()
    drop = flagged[flagged["metric"] == "cambio_saldo"].iloc[0]
    assert drop["change_pct"] == pytest.approx(100 * (drop["value"] / drop["reference"] - 1))
    assert f"{drop['change_pct']:+.1f}%" in drop["message"]
    # every non-informative alert carries a relative change
    assert h.alerts.loc[h.alerts["severity"] != "info", "change_pct"].notna().all()


def test_old_format_forecasts_still_load(tmp_path):
    store = Store(tmp_path)
    store.save_forecast("2025-01-01", pd.DataFrame({"target_date": [D("2025-01-02")], "total": [1.0]}),
                        pd.DataFrame({"maturity_date": [D("2025-06-01")], "expected_at_maturity": [1.0]}),
                        {"model": "usage", "expected_unused_at_maturity": 1.0})
    t, b = store.load_forecasts()
    assert list(t["method"]) == ["multi"] and list(b["method"]) == ["multi"]


def test_streamlit_page_renders(history, monkeypatch):
    from streamlit.testing.v1 import AppTest
    store, _, _ = history
    monkeypatch.setenv("TIDIS_DATA_DIR", str(store.root))
    monkeypatch.setattr(source, "download", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7
