"""Download and parse the 'características financieras títulos vigentes' file."""
from __future__ import annotations

import io
import re
import urllib.request
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import (BALANCE_COL, CATEGORY_COL, CATEGORY_VALUE, ID_CANDIDATES,
                     ISSUE_CANDIDATES, MATURITY_COL, SOURCE_URL)

SNAPSHOT_COLUMNS = ["title_id", "issue_date", "maturity_date", "balance", "imputed"]

_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def download(url: str = SOURCE_URL, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TIDIS monitor)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------- parsing helpers
def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.normalize()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for i, v in s.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (datetime, date, pd.Timestamp)):
            out[i] = pd.Timestamp(v).normalize()
        elif isinstance(v, (int, float)) and 20000 < v < 80000:   # Excel serial date
            out[i] = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))
        else:
            txt = str(v).strip()
            iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", txt) is not None
            out[i] = pd.to_datetime(txt, dayfirst=not iso, errors="coerce")
    return out.dt.normalize()


def _parse_number(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    txt = re.sub(r"[^\d,.\-]", "", str(v))
    if not txt:
        return np.nan
    if "," in txt and "." in txt:                     # last separator is the decimal one
        if txt.rfind(",") > txt.rfind("."):
            txt = txt.replace(".", "").replace(",", ".")
        else:
            txt = txt.replace(",", "")
    elif "," in txt:                                  # 1,234,567 or 1234,5
        parts = txt.split(",")
        txt = txt.replace(",", "") if all(len(p) == 3 for p in parts[1:]) else txt.replace(",", ".")
    elif txt.count(".") > 1:                          # 1.234.567
        txt = txt.replace(".", "")
    try:
        return float(txt)
    except ValueError:
        return np.nan


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def _find_col(columns, candidates) -> str | None:
    lookup = {_norm(c): c for c in columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


def _find_cut_date(top: pd.DataFrame, reference: date | None) -> date | None:
    """Best effort: look for a 'fecha de corte' in the rows above the header."""
    found = []
    for v in top.to_numpy().ravel():
        if isinstance(v, (datetime, pd.Timestamp)):
            found.append(pd.Timestamp(v).date())
            continue
        if not isinstance(v, str):
            continue
        t = v.lower()
        for m in re.finditer(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", t):
            try:
                found.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
            except ValueError:
                pass
        for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", t):
            try:
                found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
        for m in re.finditer(r"(\d{1,2}) de ([a-z]+) de (\d{4})", t):
            if m.group(2) in _MONTHS_ES:
                try:
                    found.append(date(int(m.group(3)), _MONTHS_ES[m.group(2)], int(m.group(1))))
                except ValueError:
                    pass
    if reference is not None:   # only trust plausible dates (up to a month before the download)
        found = [d for d in found if 0 <= (reference - d).days <= 31]
    return max(found) if found else None


# --------------------------------------------------------------------------- public API
def read_table(content: bytes, reference: date | None = None) -> tuple[pd.DataFrame, date | None]:
    """Return the full table and the cut date found in the header block (if any)."""
    raw = pd.read_excel(io.BytesIO(content), header=None)
    header_row = None
    for i in range(min(len(raw), 30)):
        if any(_norm(v) == _norm(CATEGORY_COL) for v in raw.iloc[i].tolist()):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Column '{CATEGORY_COL}' not found in the first 30 rows of the file")
    table = raw.iloc[header_row + 1:].copy()
    table.columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    table = table.dropna(how="all").reset_index(drop=True)
    return table, _find_cut_date(raw.iloc[:header_row], reference)


def extract_tidis(table: pd.DataFrame) -> pd.DataFrame:
    """Normalise the TIDIS rows to the snapshot schema (one row per title)."""
    cat = _find_col(table.columns, [CATEGORY_COL])
    bal = _find_col(table.columns, [BALANCE_COL])
    mat = _find_col(table.columns, [MATURITY_COL])
    missing = [n for n, c in [(CATEGORY_COL, cat), (BALANCE_COL, bal), (MATURITY_COL, mat)] if c is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    idc = _find_col(table.columns, ID_CANDIDATES)
    isc = _find_col(table.columns, ISSUE_CANDIDATES)

    t = table[table[cat].astype(str).str.strip().str.upper() == CATEGORY_VALUE]
    out = pd.DataFrame({
        "title_id": t[idc].astype(str).str.strip() if idc else pd.Series(pd.NA, index=t.index, dtype="string"),
        "issue_date": _parse_dates(t[isc]) if isc else pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]"),
        "maturity_date": _parse_dates(t[mat]),
        "balance": t[bal].map(_parse_number).astype(float),
        "imputed": False,
    })
    out["title_id"] = out["title_id"].astype("string")
    return out.reset_index(drop=True)[SNAPSHOT_COLUMNS]


def parse_file(content: bytes, reference: date | None = None) -> tuple[pd.DataFrame, date | None]:
    table, cut = read_table(content, reference)
    return extract_tidis(table), cut
