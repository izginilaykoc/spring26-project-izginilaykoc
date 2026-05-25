"""Decision-tree baseline (port of original `classify.py`).

Trains a small DecisionTreeClassifier on (weather + calendar) -> system_direction
using the last `TRAIN_DAYS` of imbalance labels, then predicts the 24 hours of
`predict_date` using the weather covering that day.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.data import (
    add_calendar_features,
    coerce_predictions,
    load_imbalance_grid,
    load_weather,
    predict_date_window,
)

TRAIN_DAYS = 120


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)

    grid = load_imbalance_grid(end_dt=end_hour, as_of_dt=as_of_dt)
    grid = grid.dropna(subset=["system_direction"])

    train_start = (grid["dt"].max() - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    weather = load_weather(start=train_start)

    train = pd.merge(
        add_calendar_features(weather),
        grid[["dt", "system_direction"]],
        on="dt",
        how="inner",
    ).dropna()

    if train.empty:
        return ["Neutral"] * 24

    feature_cols = [c for c in train.columns if c not in ("dt", "system_direction")]
    X_train = train[feature_cols]
    y_train = train["system_direction"]

    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    target_weather = add_calendar_features(weather)
    target_rows = (
        target_weather[(target_weather["dt"] >= start_hour) & (target_weather["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
    )

    if len(target_rows) < 24:
        return ["Neutral"] * 24

    preds = list(model.predict(target_rows[feature_cols]))
    return coerce_predictions(preds)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
