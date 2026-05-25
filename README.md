# Turkish Electricity Market — System Imbalance Direction Classifier

Course project for **IE 48B (Spring 2026)**. The task: for each hour of the next trading day, predict whether the Turkish power grid's system imbalance will be **Positive**, **Negative**, or **Neutral** — 24 labels submitted at noon of day *d*, scored against EPİAŞ-published ground truth.

Final leaderboard submission lives in `classify.py` as a single self-contained file (the grading harness copies only `classify.py` + `classify_helper*` into its workspace).

---

## Approach

A **RandomForestClassifier** trained directly on the categorical label (no regression-then-threshold), on a rolling 180-day window, using:

- **Weather features** (Open-Meteo) for three demand centers: Istanbul, Ankara, Izmir — temperature, humidity, wind, etc., spliced from the historical-forecast archive plus 2-day forecast horizon.
- **Calendar features**: hour-of-day, day-of-week, month, holiday flags, cyclical sin/cos encodings.
- **Lagged net imbalance & direction**: `net_lag_{168, 336}` + one-hot `dir_lag_{168, 336}`.

### Why lags ≥ 168h?
EPİAŞ publishes realized imbalance with a variable 6–12h delay, so at the noon-of-*d* cutoff the most recent reliable observation is ~35h old. Keeping every lag at ≥ 168h (one week) gives a wide safety margin against the publish-delay jitter that would otherwise leak NaNs into the feature matrix at inference time.

### Safety chain
```
RandomForest  →  seasonal-naive (168h lag)  →  all-Neutral
```
Every step wraps the next in a try/except so a transient API failure or feature-extraction bug degrades gracefully instead of crashing the submission (an exception in the grader collapses to all-Neutral ≈ 4.17% accuracy — to be avoided at all costs).

---

## Repo layout

```
classify.py              Final leaderboard submission (self-contained)
classify_helper.py       Data-fetching helpers (Python)
classify_helper.r        R-language helpers (parallel implementation)

src/
  data.py                Grid construction, weather/imbalance ingestion
  features.py            Feature engineering (calendar, cyclical, lags)
  backtest.py            Walk-forward backtest harness
models/
  m_seasonal_naive.py    Baseline: 168h-lag direction
  m_dt_baseline.py       Decision tree baseline
  m_weather_lr.py        Logistic regression on weather (winner of model search)
  m_gam_baseline.py      Generalized additive model
  m_gam_plus.py          GAM with extended interactions
  m_lgbm_reg.py          LightGBM regression-to-label
  m_diff_lag.py          Differenced-lag variant
scripts/
  experiment.py          Variant sweep on the leaderboard window
  preflight.py           Sanity checks before submission
notebooks/
  01_eda.ipynb           Exploratory analysis
  02_backtest_compare.ipynb   Variant comparison
results/
  backtest_summary.csv   Per-model walk-forward scores
  backtest_per_day.csv   Per-day breakdown
  winner.txt             Selected model identifier
```

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Walk-forward backtest over the last N days
python -m src.backtest --days 30

# Run the variant sweep (mirrors leaderboard scoring window)
python -m scripts.experiment

# Submission entry point (the grader calls this)
python classify.py
```

---

## Notes

- All external data (EPİAŞ imbalance, Open-Meteo weather) is fetched live at inference time — no embedded datasets.
- The imbalance source is the course-provided shared spreadsheet referenced in `classify_helper.py`.
- Python 3.11+; key deps: `scikit-learn`, `pandas`, `numpy`, `openmeteo-requests`, `statsmodels`.
