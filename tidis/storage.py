"""File-based history store.

data/
  raw/YYYY-MM-DD.xlsx              original file as downloaded (allows re-parsing later)
  snapshots/YYYY-MM-DD.parquet     TIDIS rows of that day (the daily "image")
  forecasts/YYYY-MM-DD_total.parquet    forecast vintage made with data up to that date
  forecasts/YYYY-MM-DD_buckets.parquet  expected balance at maturity per maturity date
  catalog.csv                      one row per snapshot (image hash, totals, origin ...)
  forecasts/vintages.csv           one row per forecast vintage (model summary)
  alerts.csv                       every flag ever raised, with first detection time
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import LOCAL_TZ, data_dir
from .source import SNAPSHOT_COLUMNS

CATALOG_COLUMNS = ["snapshot_date", "saved_at", "source_cut_date", "origin", "sha256",
                   "n_rows", "n_imputed_rows", "total_balance"]
ALERT_COLUMNS = ["date", "category", "severity", "metric", "value", "reference", "message",
                 "key", "first_detected"]


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_PARQUET_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def _read_parquet(path: Path) -> pd.DataFrame:
    """Parquet files are immutable once written (new writes change mtime): cache in-process."""
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _PARQUET_CACHE:
        _PARQUET_CACHE[key] = pd.read_parquet(path)
    return _PARQUET_CACHE[key].copy()


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def image_digest(rows: pd.DataFrame) -> str:
    """Hash of the TIDIS image itself (not of the file: xlsx metadata changes on every export)."""
    cols = ["title_id", "issue_date", "maturity_date", "balance"]
    canon = rows[cols].astype(str).sort_values(cols).to_csv(index=False)
    return sha256(canon.encode("utf-8"))


def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


class Store:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else data_dir()
        self.raw_dir = self.root / "raw"
        self.snap_dir = self.root / "snapshots"
        self.fc_dir = self.root / "forecasts"
        self.catalog_path = self.root / "catalog.csv"
        self.vintages_path = self.fc_dir / "vintages.csv"
        self.alerts_path = self.root / "alerts.csv"

    # ------------------------------------------------------------------ catalog / snapshots
    def catalog(self) -> pd.DataFrame:
        if not self.catalog_path.exists():
            return pd.DataFrame(columns=CATALOG_COLUMNS)
        cat = pd.read_csv(self.catalog_path, dtype={"sha256": str, "origin": str})
        cat["snapshot_date"] = pd.to_datetime(cat["snapshot_date"])
        return cat.sort_values("snapshot_date").reset_index(drop=True)

    def snapshot_dates(self) -> list[pd.Timestamp]:
        return list(self.catalog()["snapshot_date"])

    def version(self) -> float:
        """Changes whenever something is written: used as a cache key by the app."""
        paths = [self.catalog_path, self.vintages_path, self.alerts_path]
        return sum(p.stat().st_mtime_ns for p in paths if p.exists())

    def save_snapshot(self, day: date, rows: pd.DataFrame, raw: bytes | None, *, origin: str,
                      source_cut_date: date | None = None, digest: str | None = None) -> str:
        """Save the image of `day`. Returns 'new', 'updated' or 'unchanged'."""
        day = pd.Timestamp(day).normalize()
        tag = day.strftime("%Y-%m-%d")
        digest = digest or image_digest(rows)
        cat = self.catalog()
        existing = cat[cat["snapshot_date"] == day]
        if len(existing) and existing["sha256"].iloc[0] == digest:
            return "unchanged"
        if raw is not None:
            _atomic_write(self.raw_dir / f"{tag}.xlsx", lambda p: Path(p).write_bytes(raw))
        rows = rows[SNAPSHOT_COLUMNS].copy()
        _atomic_write(self.snap_dir / f"{tag}.parquet", lambda p: rows.to_parquet(p, index=False))
        entry = {
            "snapshot_date": day, "saved_at": now_local().isoformat(timespec="seconds"),
            "source_cut_date": source_cut_date.isoformat() if source_cut_date else "",
            "origin": origin, "sha256": digest, "n_rows": len(rows),
            "n_imputed_rows": int(rows["imputed"].sum()), "total_balance": float(rows["balance"].sum()),
        }
        cat = _concat([cat[cat["snapshot_date"] != day], pd.DataFrame([entry])])
        cat = cat.sort_values("snapshot_date")
        out = cat.assign(snapshot_date=cat["snapshot_date"].dt.strftime("%Y-%m-%d"))
        _atomic_write(self.catalog_path, lambda p: out.to_csv(p, index=False))
        return "updated" if len(existing) else "new"

    def load_snapshot(self, day) -> pd.DataFrame:
        tag = pd.Timestamp(day).strftime("%Y-%m-%d")
        df = _read_parquet(self.snap_dir / f"{tag}.parquet")
        df["snapshot_date"] = pd.Timestamp(day).normalize()
        return df

    def load_snapshots(self, until=None) -> pd.DataFrame:
        dates = self.snapshot_dates()
        if until is not None:
            dates = [d for d in dates if d <= pd.Timestamp(until)]
        if not dates:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS + ["snapshot_date"])
        return pd.concat([self.load_snapshot(d) for d in dates], ignore_index=True)

    def previous_snapshot(self, day) -> pd.DataFrame | None:
        prev = [d for d in self.snapshot_dates() if d < pd.Timestamp(day)]
        return self.load_snapshot(prev[-1]) if prev else None

    def raw_path(self, day) -> Path:
        return self.raw_dir / f"{pd.Timestamp(day).strftime('%Y-%m-%d')}.xlsx"

    # ------------------------------------------------------------------ forecasts
    def vintages(self) -> pd.DataFrame:
        if not self.vintages_path.exists():
            return pd.DataFrame(columns=["asof"])
        v = pd.read_csv(self.vintages_path)
        v["asof"] = pd.to_datetime(v["asof"])
        return v.sort_values("asof").reset_index(drop=True)

    def has_forecast(self, asof) -> bool:
        tag = pd.Timestamp(asof).strftime("%Y-%m-%d")
        return (self.fc_dir / f"{tag}_total.parquet").exists()

    def save_forecast(self, asof, total: pd.DataFrame, buckets: pd.DataFrame, meta: dict) -> None:
        tag = pd.Timestamp(asof).strftime("%Y-%m-%d")
        _atomic_write(self.fc_dir / f"{tag}_total.parquet", lambda p: total.to_parquet(p, index=False))
        _atomic_write(self.fc_dir / f"{tag}_buckets.parquet", lambda p: buckets.to_parquet(p, index=False))
        v = self.vintages()
        row = {"asof": pd.Timestamp(asof).normalize(), "created_at": now_local().isoformat(timespec="seconds"), **meta}
        v = _concat([v[v["asof"] != row["asof"]], pd.DataFrame([row])]).sort_values("asof")
        out = v.assign(asof=v["asof"].dt.strftime("%Y-%m-%d"))
        _atomic_write(self.vintages_path, lambda p: out.to_csv(p, index=False))

    def load_forecasts(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        totals, buckets = [], []
        for asof in self.vintages()["asof"]:
            tag = asof.strftime("%Y-%m-%d")
            t_path, b_path = self.fc_dir / f"{tag}_total.parquet", self.fc_dir / f"{tag}_buckets.parquet"
            if t_path.exists():
                totals.append(_read_parquet(t_path).assign(asof=asof))
            if b_path.exists():
                buckets.append(_read_parquet(b_path).assign(asof=asof))
        t = pd.concat(totals, ignore_index=True) if totals else pd.DataFrame(columns=["asof", "target_date", "total"])
        b = pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame(columns=["asof", "maturity_date"])
        return t, b

    # ------------------------------------------------------------------ alerts
    def alerts(self) -> pd.DataFrame:
        if not self.alerts_path.exists():
            return pd.DataFrame(columns=ALERT_COLUMNS)
        a = pd.read_csv(self.alerts_path)
        a["date"] = pd.to_datetime(a["date"])
        return a

    def save_alerts(self, fresh: pd.DataFrame) -> pd.DataFrame:
        """Merge freshly computed alerts with the log, preserving first-detection times."""
        old = self.alerts()
        first = dict(zip(old["key"], old["first_detected"])) if len(old) else {}
        stamp = now_local().isoformat(timespec="seconds")
        fresh = fresh.copy()
        fresh["first_detected"] = [first.get(k, stamp) for k in fresh["key"]]
        # keep alerts from the log that can no longer be recomputed (e.g. history rewritten)
        merged = _concat([fresh, old[~old["key"].isin(fresh["key"])]])
        if merged.empty:
            merged = pd.DataFrame(columns=ALERT_COLUMNS)
        merged = merged[ALERT_COLUMNS].sort_values(["date", "category"], ascending=[False, True])
        out = merged.assign(date=pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d"))
        _atomic_write(self.alerts_path, lambda p: out.to_csv(p, index=False))
        return merged.reset_index(drop=True)
