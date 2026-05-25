"""Walk-forward backtest harness.

Mirrors the leaderboard scoring (mean daily accuracy) over the last N days of
labeled imbalance data. For each day d in the test window:
  * pick `as_of_dt = d - 1 day` (mimicking the noon-of-day-d data cutoff with
    EPIAS publish delay)
  * call model.predict(d, as_of_dt=as_of_dt) -> 24 labels
  * compare against ground-truth `system_direction` for the 24 hours of d
  * record per-day accuracy

Writes/updates `results/backtest_summary.csv` with one row per (model, run).
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data import VALID_LABELS, load_imbalance_grid, predict_date_window

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "backtest_summary.csv"
PER_DAY_CSV = RESULTS_DIR / "backtest_per_day.csv"

DEFAULT_MODELS = (
    "m_seasonal_naive",
    "m_diff_lag",
    "m_dt_baseline",
    "m_weather_lr",
)


def _load_predict(name: str):
    mod = importlib.import_module(f"models.{name}")
    return getattr(mod, "predict")


def _truth_for(predict_date: pd.Timestamp) -> list[str] | None:
    start_hour, end_hour = predict_date_window(predict_date)
    grid = load_imbalance_grid(end_dt=end_hour)
    rows = grid[(grid["dt"] >= start_hour) & (grid["dt"] <= end_hour)].sort_values("dt")
    if len(rows) != 24 or rows["system_direction"].isna().any():
        return None
    return rows["system_direction"].tolist()


def _last_full_truth_dates(n: int) -> list[pd.Timestamp]:
    """Find the last `n` calendar dates that have all 24 hours of labels."""
    grid = load_imbalance_grid()
    fully_labeled = (
        grid.dropna(subset=["system_direction"])
        .assign(date=lambda d: d["dt"].dt.normalize())
        .groupby("date")
        .size()
        .loc[lambda s: s == 24]
        .index
        .sort_values()
    )
    return list(fully_labeled[-n:])


def run_backtest(
    models: Iterable[str] = DEFAULT_MODELS,
    days: int = 14,
    cutoff_offset_hours: int = 24,
) -> pd.DataFrame:
    """Score each model on the last `days` fully-labeled days.

    `cutoff_offset_hours` controls the as_of_dt: predict_date_start - offset.
    24h is conservative (assumes a full day of EPIAS delay).
    """
    test_dates = _last_full_truth_dates(days)
    if not test_dates:
        raise RuntimeError("No fully-labeled days found — is imbalance data loaded?")

    print(f"Backtest window: {test_dates[0].date()} … {test_dates[-1].date()} ({len(test_dates)} days)")

    per_day_records: list[dict] = []
    summary_records: list[dict] = []
    run_ts = pd.Timestamp.now(tz="Europe/Istanbul").isoformat()

    for name in models:
        print(f"\n=== {name} ===")
        try:
            predict_fn = _load_predict(name)
        except Exception as e:
            print(f"  load failed: {e!r}")
            continue

        per_day_acc: list[float] = []
        t0 = time.time()
        for d in test_dates:
            as_of = d - pd.Timedelta(hours=cutoff_offset_hours)
            truth = _truth_for(d)
            if truth is None:
                continue
            try:
                preds = predict_fn(d, as_of_dt=as_of)
            except Exception as e:
                print(f"  {d.date()}: predict failed ({e!r})")
                continue
            if len(preds) != 24 or any(p not in VALID_LABELS for p in preds):
                print(f"  {d.date()}: invalid output, skipping")
                continue
            acc = sum(p == t for p, t in zip(preds, truth)) / 24
            per_day_acc.append(acc)
            per_day_records.append(
                {"model": name, "date": d.date().isoformat(), "accuracy": acc, "run_ts": run_ts}
            )
            print(f"  {d.date()}: {acc:.3f}")

        elapsed = time.time() - t0
        if per_day_acc:
            mean_acc = sum(per_day_acc) / len(per_day_acc)
            std_acc = pd.Series(per_day_acc).std()
            print(f"  → mean={mean_acc:.3f}  std={std_acc:.3f}  ({len(per_day_acc)} days, {elapsed:.1f}s)")
            summary_records.append(
                {
                    "model": name,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "n_days": len(per_day_acc),
                    "elapsed_s": elapsed,
                    "run_ts": run_ts,
                }
            )

    summary = pd.DataFrame(summary_records).sort_values("mean_acc", ascending=False)
    per_day = pd.DataFrame(per_day_records)

    RESULTS_DIR.mkdir(exist_ok=True)
    if SUMMARY_CSV.exists():
        prev = pd.read_csv(SUMMARY_CSV)
        pd.concat([prev, summary], ignore_index=True).to_csv(SUMMARY_CSV, index=False)
    else:
        summary.to_csv(SUMMARY_CSV, index=False)
    if PER_DAY_CSV.exists():
        prev = pd.read_csv(PER_DAY_CSV)
        pd.concat([prev, per_day], ignore_index=True).to_csv(PER_DAY_CSV, index=False)
    else:
        per_day.to_csv(PER_DAY_CSV, index=False)

    print("\nLeaderboard (this run):")
    print(summary.to_string(index=False))
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--cutoff-hours", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(models=args.models, days=args.days, cutoff_offset_hours=args.cutoff_hours)
