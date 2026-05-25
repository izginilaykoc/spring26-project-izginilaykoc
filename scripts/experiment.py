"""Experimentation harness — try variants and pick the best.

Backtests several classifier variants on the leaderboard window (May 4-17, 2026).
Uses cutoff_offset_hours=12 to mirror the noon-of-d submission timing.

Run with: .venv/bin/python -m scripts.experiment
"""

from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import (
    DEFAULT_COORDS,
    DEFAULT_WEATHER_VARS,
    add_calendar_features,
    load_imbalance_grid,
    load_weather,
    net_to_label,
    predict_date_window,
)

warnings.filterwarnings("ignore")

TZ = "Europe/Istanbul"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_frame(
    predict_date: pd.Timestamp,
    as_of_dt: pd.Timestamp | None,
    train_days: int = 180,
    extra_weather: bool = False,
    extra_lags: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (train_df, target_df, feature_cols).

    target_df has the 24 hours of `predict_date` with all features filled.
    """
    start_hour, end_hour = predict_date_window(predict_date)
    grid = load_imbalance_grid(end_dt=end_hour, as_of_dt=as_of_dt)

    # Lag features. Submission runs ~noon of day d for predictions of day d+1.
    # Target hour 23 needs lag >= 35h to be safe; use 48h as conservative floor.
    base_lags = (168, 336)
    if extra_lags:
        more = (48, 72, 504, 672)
    else:
        more = ()
    for lag_h in base_lags + more:
        grid[f"net_lag_{lag_h}"] = grid["net"].shift(lag_h)
    # Direction (categorical) — one-hot of lag 168 and 336 (weekly cycle)
    for lag_h in (168, 336):
        col = f"dir_lag_{lag_h}"
        grid[col] = grid["system_direction"].shift(lag_h)

    if extra_lags:
        # Rolling means: compute from data ending 48h ago (safe).
        safe_net = grid["net"].shift(48)
        grid["net_roll_24h"] = safe_net.rolling(24).mean()
        grid["net_roll_168h"] = safe_net.rolling(168).mean()

    last_obs_dt = grid.dropna(subset=["net"])["dt"].max()
    if pd.isna(last_obs_dt):
        raise RuntimeError("No imbalance observations available")

    train_start = (last_obs_dt - pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")

    weather_vars = list(DEFAULT_WEATHER_VARS)
    if extra_weather:
        weather_vars += [
            "precipitation",
            "surface_pressure",
            "shortwave_radiation",
            "wind_gusts_10m",
        ]
    weather = load_weather(
        start=train_start,
        variables=tuple(weather_vars),
    )
    weather = add_calendar_features(weather)
    # Cyclical encodings
    weather["hour_sin"] = np.sin(2 * np.pi * weather["hour"] / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["hour"] / 24)
    weather["dow_sin"] = np.sin(2 * np.pi * weather["dayofweek"] / 7)
    weather["dow_cos"] = np.cos(2 * np.pi * weather["dayofweek"] / 7)
    weather["hour_of_week"] = weather["dayofweek"] * 24 + weather["hour"]

    merged = pd.merge(weather, grid, on="dt", how="inner")

    # One-hot the direction lags
    for col in ("dir_lag_168", "dir_lag_336"):
        for lab in ("Positive", "Negative", "Neutral"):
            merged[f"{col}_{lab}"] = (merged[col] == lab).astype(float)
        merged.drop(columns=[col], inplace=True)

    # Define feature columns: everything that's not dt, net, or direction
    feature_cols = [
        c for c in merged.columns
        if c not in ("dt", "net", "system_direction")
    ]

    lag_cols = [c for c in feature_cols if c.startswith("net_lag_") or c.startswith("net_roll_")]
    train = merged.dropna(subset=["net"] + lag_cols).copy()

    target = (
        merged[(merged["dt"] >= start_hour) & (merged["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
        .copy()
    )

    return train, target, feature_cols


def label_from_net(net: float) -> str:
    if pd.isna(net):
        return "Neutral"
    if net >= 50:
        return "Positive"
    if net <= -50:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

ModelFn = Callable[[pd.Timestamp, pd.Timestamp | None], list[str]]


def make_regress_then_threshold(model_cls, extra_weather=False, extra_lags=False, **kw) -> ModelFn:
    def _predict(predict_date, as_of_dt):
        train, target, feat = build_feature_frame(
            predict_date, as_of_dt, train_days=180,
            extra_weather=extra_weather, extra_lags=extra_lags,
        )
        if train.empty or len(target) < 24 or target[feat].isna().any().any():
            return ["Neutral"] * 24
        m = model_cls(**kw)
        m.fit(train[feat], train["net"])
        preds = m.predict(target[feat])
        return [label_from_net(v) for v in preds]
    return _predict


def make_classifier(model_cls, extra_weather=False, extra_lags=False, **kw) -> ModelFn:
    def _predict(predict_date, as_of_dt):
        train, target, feat = build_feature_frame(
            predict_date, as_of_dt, train_days=180,
            extra_weather=extra_weather, extra_lags=extra_lags,
        )
        if train.empty or len(target) < 24 or target[feat].isna().any().any():
            return ["Neutral"] * 24
        y_train = train["net"].apply(label_from_net)
        m = model_cls(**kw)
        m.fit(train[feat], y_train)
        return list(m.predict(target[feat]))
    return _predict


def make_scaled_classifier(model_cls, extra_weather=False, extra_lags=False, **kw) -> ModelFn:
    def _predict(predict_date, as_of_dt):
        train, target, feat = build_feature_frame(
            predict_date, as_of_dt, train_days=180,
            extra_weather=extra_weather, extra_lags=extra_lags,
        )
        if train.empty or len(target) < 24 or target[feat].isna().any().any():
            return ["Neutral"] * 24
        y_train = train["net"].apply(label_from_net)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feat])
        X_target = scaler.transform(target[feat])
        m = model_cls(**kw)
        m.fit(X_train, y_train)
        return list(m.predict(X_target))
    return _predict


from sklearn.ensemble import HistGradientBoostingRegressor
from collections import Counter


def make_ensemble_vote(member_fns: list[ModelFn]) -> ModelFn:
    """Majority vote across member models, ties broken by RF prediction."""
    def _predict(predict_date, as_of_dt):
        all_preds = [fn(predict_date, as_of_dt) for fn in member_fns]
        out = []
        for h in range(24):
            votes = [p[h] for p in all_preds]
            counts = Counter(votes)
            top = counts.most_common(1)[0]
            # On tie, prefer the first member's vote
            if list(counts.values()).count(top[1]) > 1:
                out.append(all_preds[0][h])
            else:
                out.append(top[0])
        return out
    return _predict


_rf_simple = make_classifier(RandomForestClassifier, n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=0)
_hgb_clf_simple = make_classifier(HistGradientBoostingClassifier, max_iter=300, max_depth=6, learning_rate=0.05, random_state=0)
_logreg_simple = make_scaled_classifier(LogisticRegression, max_iter=3000, C=1.0)
_linreg_simple = make_regress_then_threshold(LinearRegression)

VARIANTS: dict[str, ModelFn] = {
    "linreg_simple": _linreg_simple,
    "logreg_simple": _logreg_simple,
    "rf_simple": _rf_simple,
    "hgb_clf_simple": _hgb_clf_simple,
    "ensemble_rf_hgb_lin": make_ensemble_vote([_rf_simple, _hgb_clf_simple, _linreg_simple]),
    "ensemble_rf_hgb_logreg": make_ensemble_vote([_rf_simple, _hgb_clf_simple, _logreg_simple]),
}


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def _last_full_truth_dates(start: str) -> list[pd.Timestamp]:
    grid = load_imbalance_grid()
    full = (
        grid.dropna(subset=["system_direction"])
        .assign(date=lambda d: d["dt"].dt.normalize())
        .groupby("date").size().loc[lambda s: s == 24].index.sort_values()
    )
    start_ts = pd.Timestamp(start, tz=TZ)
    return [d for d in full if d >= start_ts]


def _truth_for(predict_date: pd.Timestamp) -> list[str] | None:
    start_hour, end_hour = predict_date_window(predict_date)
    grid = load_imbalance_grid(end_dt=end_hour)
    rows = grid[(grid["dt"] >= start_hour) & (grid["dt"] <= end_hour)].sort_values("dt")
    if len(rows) != 24 or rows["system_direction"].isna().any():
        return None
    return rows["system_direction"].tolist()


def backtest(name: str, fn: ModelFn, dates: list[pd.Timestamp], cutoff_hours: int) -> tuple[float, list[float]]:
    accs = []
    t0 = time.time()
    for d in dates:
        as_of = d - pd.Timedelta(hours=cutoff_hours)
        truth = _truth_for(d)
        if truth is None:
            continue
        try:
            preds = fn(d, as_of)
        except Exception as e:
            print(f"  {d.date()}: failed {e!r}")
            accs.append(0.0)
            continue
        acc = sum(p == t for p, t in zip(preds, truth)) / 24
        accs.append(acc)
        print(f"  {d.date()}: {acc:.3f}")
    print(f"  → {name} mean={np.mean(accs):.3f}  ({time.time()-t0:.1f}s)")
    return float(np.mean(accs)), accs


def main():
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-04-01"
    dates = _last_full_truth_dates(start)
    print(f"Backtest window: {dates[0].date()} .. {dates[-1].date()} ({len(dates)} days)\n")

    results = {}
    for name, fn in VARIANTS.items():
        print(f"\n=== {name} ===")
        mean_acc, per_day = backtest(name, fn, dates, cutoff_hours=12)
        results[name] = (mean_acc, per_day)

    print("\n\n=== LEADERBOARD ===")
    for name, (mean_acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:20s} {mean_acc:.3f}")


if __name__ == "__main__":
    main()
