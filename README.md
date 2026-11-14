# Monitor TIDIS

Streamlit page that follows the outstanding **TIDIS** balance day by day:
it saves a daily *image* (balance by maturity date), rebuilds the full history,
imputes missing days, forecasts how the stock will degrade, and flags significant changes.

```
streamlit run main.py      # the page (downloads + saves at most once per hour)
python collect.py          # the same daily cycle without the page (for a scheduler)
python -m pytest tests -q  # tests (synthetic data, no internet needed)
```

## How it works

| Step | What happens | Where |
|---|---|---|
| Download | `caracteristicas-financieras-titulos-vigentes.xlsx`; the header row is located automatically, the cut date is read from the title rows if present | `tidis/source.py` |
| Image of the day | TIDIS rows (title, issue date, maturity, balance) saved as `data/snapshots/YYYY-MM-DD.parquet`, plus the original file in `data/raw/` | `tidis/storage.py` |
| Row imputation | a title with missing balance/maturity is completed from the previous image of the same title | `tidis/imputation.py` |
| Day imputation | days without image are rebuilt title by title: continuing titles interpolated, titles that mature in the gap stay flat until maturity, new titles appear on their issue date | `tidis/imputation.py` |
| Comparing images | every change is split into **issued**, **used** (redeemed for taxes before maturity) and **matured unused**; the three add up exactly to the change | `tidis/analysis.py` |
| Forecast | usage rate per days-to-maturity bucket estimated from the history, projected per maturity date, plus new issuance at the average rate. One forecast per image is saved once and never overwritten (`data/forecasts/`), so the forecasts themselves have a history | `tidis/forecast.py` |
| Alerts | unusual usage/issuance (robust z-score), revisions between consecutive forecasts (+30/+90/+180 d and "will expire unused"), error of yesterday's forecast vs today's image, imputed days/rows. Kept in `data/alerts.csv` with first-detection time | `tidis/alerts.py` |

All thresholds and model parameters are in `tidis/config.py`.

## Daily images: schedule `collect.py`

The page only saves when someone opens it. For a full daily history, run the collector once a day.
Exit code is `2` when the new image raises an *alerta* (useful for notifications).

* cron: `0 9 * * * cd /path/TIDIS && python collect.py >> collect.log 2>&1`
* GitHub Actions (also persists the history in the repo, see note below):

```yaml
# .github/workflows/daily.yml
on:
  schedule: [{cron: "0 14 * * *"}]   # 09:00 Bogotá
  workflow_dispatch:
permissions: {contents: write}
jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -r requirements.txt
      - run: python collect.py || [ $? -eq 2 ]
      - run: |
          git config user.name "tidis-bot" && git config user.email "bot@users.noreply.github.com"
          git add data && git commit -m "TIDIS image $(date +%F)" && git push || echo "nothing new"
```

**Persistence note:** on Streamlit Community Cloud the disk is wiped on every restart/redeploy.
Keep `data/` in the repository (as in the workflow above) or point `TIDIS_DATA_DIR` to a persistent
volume. If the repo grows too much, set `keep_raw=False` in `tidis/config.py` (only the TIDIS images are kept).

Past files can be added from the page (sidebar → *Agregar archivo histórico*) or with
`python collect.py --file old.xlsx --date 2026-03-31`; forecasts for those dates are made using only data
available up to that date. After changing the model, `python collect.py --rebuild-forecasts` recomputes them.
