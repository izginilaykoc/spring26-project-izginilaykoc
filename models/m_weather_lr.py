"""Linear regression on `net` with weather + calendar + lag features.

Trains LinearRegression(net ~ weather + calendar + net_lag_168 + net_lag_336)
on the trailing window, predicts net for the 24 hours of `predict_date`,
then thresholds to (Positive / Negative / Neutral).

Lags <= 12 are skipped (EPIAS publish delay), so we lean on lag-168 / lag-336
which are always safe.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import (
    add_calendar_features,
    coerce_predictions,
    load_imbalance_grid,
    load_weather,
    net_to_label,
    predict_date_window,
)

TRAIN_DAYS = 120


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)

    grid = load_imbalance_grid(end_dt=end_hour, as_of_dt=as_of_dt)
    grid["net_lag_168"] = grid["net"].shift(168)
    grid["net_lag_336"] = grid["net"].shift(336)

    last_obs_dt = grid.dropna(subset=["net"])["dt"].max()
    if pd.isna(last_obs_dt):
        return ["Neutral"] * 24

    train_start = (last_obs_dt - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    weather = add_calendar_features(load_weather(start=train_start))

    merged = pd.merge(weather, grid, on="dt", how="inner")

    lag_cols = ["net_lag_168", "net_lag_336"]
    feature_cols = [c for c in merged.columns if c not in ("dt", "net", "system_direction")]

    train = merged.dropna(subset=["net"] + lag_cols).copy()
    if train.empty:
        return ["Neutral"] * 24

    X_train = train[feature_cols]
    y_train = train["net"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    target = (
        merged[(merged["dt"] >= start_hour) & (merged["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
    )

    if len(target) < 24 or target[lag_cols].isna().any().any():
        return ["Neutral"] * 24

    preds_net = model.predict(target[feature_cols])
    raw = [net_to_label(v) for v in preds_net]
    return coerce_predictions(raw)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
