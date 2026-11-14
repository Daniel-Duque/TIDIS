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
from tidis.forecast import estimate_model, project  # noqa: E402
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


# ----------------------------------------------------------------------------- forecast
def test_single_image_gives_contractual_runoff():
    s = snap("2025-01-01", [("2025-02-01", 100.0), ("2025-05-01", 50.0)])
    panel, _, aom = build_panel(s)
    model = estimate_model(panel, aom, s)
    assert model.kind == "runoff"
    total, buckets = project(panel.iloc[-1], panel.index[-1], model, aom, Settings(horizon_days=200))
    t = total.set_index("target_date")["total"]
    assert t[D("2025-01-15")] == 150 and t[D("2025-03-01")] == 50 and t[D("2025-06-01")] == 0
    assert buckets["expected_at_maturity"].sum() == 150


def test_model_recovers_known_usage_rate():
    rate = 0.004
    days = pd.date_range("2025-01-01", periods=60)
    mats = [D("2025-08-01"), D("2025-10-01"), D("2025-12-01")]
    frames = [snap(d, [(m, 1000 * (1 - rate) ** i) for m in mats]) for i, d in enumerate(days)]
    s = pd.concat(frames)
    panel, _, aom = build_panel(s)
    model = estimate_model(panel, aom, s, Settings(min_bin_transitions=5))
    assert model.kind == "usage"
    assert model.global_rate == pytest.approx(rate, rel=1e-9)
    assert model.issuance_per_day == 0


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


def test_shock_is_flagged(history):
    store, sim, h = history
    shock = list(sim)[60]
    flagged = h.alerts[h.alerts["date"] == shock]
    assert {"uso_inusual", "error_prediccion"} <= set(flagged["metric"])
    assert (flagged["severity"] == "alerta").any()


def test_streamlit_page_renders(history, monkeypatch):
    from streamlit.testing.v1 import AppTest
    store, _, _ = history
    monkeypatch.setenv("TIDIS_DATA_DIR", str(store.root))
    monkeypatch.setattr(source, "download", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
