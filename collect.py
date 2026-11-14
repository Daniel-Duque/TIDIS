"""Headless daily job: download today's image, save it, forecast, flag.

    python collect.py                          # daily run (cron / GitHub Actions / Task Scheduler)
    python collect.py --file old.xlsx --date 2026-03-31   # add a historical image
    python collect.py --rebuild-forecasts      # recompute every forecast vintage (after model changes)

Exit code 0 = ok, 1 = download failed, 2 = new alerts of severity 'alerta' for the latest image.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from tidis import pipeline
from tidis.storage import Store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="ingest a local .xlsx instead of downloading")
    ap.add_argument("--date", type=date.fromisoformat, help="date of the image given with --file (YYYY-MM-DD)")
    ap.add_argument("--data-dir", type=Path, help="history folder (default: ./data or $TIDIS_DATA_DIR)")
    ap.add_argument("--rebuild-forecasts", action="store_true")
    args = ap.parse_args()

    store = Store(args.data_dir) if args.data_dir else Store()
    code = 0
    if args.file:
        res = pipeline.ingest(store, args.file.read_bytes(), day=args.date, origin="carga manual")
    else:
        res = pipeline.fetch_and_ingest(store)
    if res["status"] == "error":
        print(res["error"], file=sys.stderr)
        code = 1
    else:
        print(f"[{res['status']}] image {res['date']}: {res['rows']} titles, total ${res['total']:,.0f}, "
              f"imputed rows {res['imputed_rows']}, dropped rows {res['dropped_rows']}")

    _, h = pipeline.update(store, fetch=False, rebuild_forecasts=args.rebuild_forecasts)
    if h.empty:
        return code or 1
    last = h.obs.index[-1]
    today_alerts = h.alerts[h.alerts["date"] == last]
    for _, a in today_alerts.iterrows():
        print(f"  {a['severity'].upper():12s} {a['category']:11s} {a['message']}")
    if code == 0 and (today_alerts["severity"] == "alerta").any():
        code = 2
    return code


if __name__ == "__main__":
    sys.exit(main())
