"""LightGBM regressor on `net` — Phase 5 leaderboard-grade workhorse.

Trains LGBMRegressor on the full feature set from src.features (calendar +
lags + weather + interactions), predicts `net`, then thresholds to labels.

Hyperparameters are conservative defaults tuned for tabular hourly data; we'll
sweep them in a follow-up once the harness is wired through. The model handles
missing features (lag NaNs, weather gaps) natively, which makes it robust to
the EPIAS publish delay without hand-rolling fallbacks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data import coerce_predictions, net_to_label, predict_date_window
from src.features import (
    all_feature_cols,
    build_features,
)

TRAIN_DAYS = 365  # LightGBM benefits from more data than GAM/LR

PARAMS = dict(
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)

    train_start = (start_hour - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    feats = build_features(end_dt=end_hour, start=train_start, as_of_dt=as_of_dt)

    train = feats.dropna(subset=["net"]).copy()
    if train.empty:
        return ["Neutral"] * 24

    feature_cols = all_feature_cols(feats)
    X_train = train[feature_cols]
    y_train = train["net"]

    model = LGBMRegressor(**PARAMS)
    model.fit(X_train, y_train)

    target = (
        feats[(feats["dt"] >= start_hour) & (feats["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
    )
    if len(target) < 24:
        return ["Neutral"] * 24

    preds_net = model.predict(target[feature_cols])
    raw = [net_to_label(v) for v in preds_net]
    return coerce_predictions(raw)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
