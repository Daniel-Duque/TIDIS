"""File-based history store.

data/
  raw/YYYY-MM-DD.xlsx              original file as downloaded (allows re-parsing later)
  snapshots/YYYY-MM-DD.parquet     TIDIS rows of that day (the daily "image")
  forecasts/YYYY-MM-DD_total.parquet    forecast vintage made with data up to that date (both methods)
  forecasts/YYYY-MM-DD_buckets.parquet  expected balance at maturity per maturity date (both methods)
  catalog.csv                      one row per snapshot (image hash, totals, origin ...)
  forecasts/vintages.csv           one row per forecast vintage (model summary)
  alerts.csv                       every flag ever raised, with first detection time
  cache/*.parquet                  consolidated copies of the daily files (derived, safe to delete)
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
ALERT_COLUMNS = ["date", "category", "severity", "metric", "value", "reference", "change_pct", "message",
                 "key", "first_detected"]


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.chmod(tmp, 0o644)
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

    def _consolidated(self, name: str, keys: dict, date_col: str, read_one) -> pd.DataFrame:
        """Single-file cache of many daily files. `keys` maps date -> version tag (hash / timestamp);
        only dates that are new or whose tag changed are re-read from their daily file."""
        path = self.root / "cache" / f"{name}.parquet"
        cached = pd.DataFrame()
        if path.exists():
            try:
                cached = _read_parquet(path)
            except Exception:          # corrupt / incompatible cache: rebuild it
                cached = pd.DataFrame()
        if not cached.empty:
            tags = cached.groupby(date_col)["_tag"].first()
            valid = [d for d, t in keys.items() if d in tags.index and tags[d] == t]
            cached = cached[cached[date_col].isin(valid)]
        else:
            valid = []
        missing = [d for d in keys if d not in set(valid)]
        if missing:
            fresh = [read_one(d).assign(_tag=keys[d]) for d in missing]
            cached = _concat([cached] + fresh)
            if not cached.empty:
                cached = cached.sort_values(date_col, kind="stable").reset_index(drop=True)
                _atomic_write(path, lambda p: cached.to_parquet(p, index=False))
        return cached.drop(columns="_tag", errors="ignore")

    def load_snapshots(self, until=None) -> pd.DataFrame:
        cat = self.catalog()
        if cat.empty:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS + ["snapshot_date"])
        keys = dict(zip(cat["snapshot_date"], cat["sha256"].fillna("").astype(str)))
        df = self._consolidated("snapshots", keys, "snapshot_date", self.load_snapshot)
        if until is not None:
            df = df[df["snapshot_date"] <= pd.Timestamp(until)]
        return df.reset_index(drop=True)

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
        v = self.vintages()
        keys = dict(zip(v["asof"], v["created_at"].astype(str))) if len(v) else {}

        def reader(kind):
            def read_one(asof):
                path = self.fc_dir / f"{asof.strftime('%Y-%m-%d')}_{kind}.parquet"
                return _read_parquet(path).assign(asof=asof) if path.exists() else pd.DataFrame({"asof": [asof]}).iloc[:0]
            return read_one

        t = self._consolidated("forecast_totals", keys, "asof", reader("total")) if keys else pd.DataFrame()
        b = self._consolidated("forecast_buckets", keys, "asof", reader("buckets")) if keys else pd.DataFrame()
        if t.empty:
            t = pd.DataFrame(columns=["asof", "target_date", "total", "method"])
        if b.empty:
            b = pd.DataFrame(columns=["asof", "maturity_date", "method"])
        for df in (t, b):                       # vintages saved before there were two methods
            df["method"] = df["method"].fillna("multi") if "method" in df else "multi"
        return t, b

    # ------------------------------------------------------------------ alerts
    def alerts(self) -> pd.DataFrame:
        if not self.alerts_path.exists():
            return pd.DataFrame(columns=ALERT_COLUMNS)
        a = pd.read_csv(self.alerts_path).reindex(columns=ALERT_COLUMNS)   # older logs: no change_pct
        a["date"] = pd.to_datetime(a["date"])
        return a

    def save_alerts(self, fresh: pd.DataFrame) -> pd.DataFrame:
        """Replace the log with the freshly computed alerts, preserving first-detection times.

        Alerts are fully reproducible from the saved images and forecasts, so the log always
        matches the current rules (no stale alerts after a rule change or a rebuild)."""
        old = self.alerts()
        first = dict(zip(old["key"], old["first_detected"])) if len(old) else {}
        stamp = now_local().isoformat(timespec="seconds")
        merged = fresh.copy() if len(fresh) else pd.DataFrame(columns=ALERT_COLUMNS)
        merged["first_detected"] = [first.get(k, stamp) for k in merged["key"]]
        merged = merged[ALERT_COLUMNS].sort_values(["date", "category"], ascending=[False, True])
        out = merged.assign(date=pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d"))
        _atomic_write(self.alerts_path, lambda p: out.to_csv(p, index=False))
        return merged.reset_index(drop=True)
