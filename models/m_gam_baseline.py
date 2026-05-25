"""GAM baseline — Lecture 8a replication, adapted to imbalance series.

LinearGAM(net ~ s(hour, cyclic) + s(doy, cyclic) + s(temp_avg)
                + s(net_lag_48) + s(net_lag_168) + f(dow))

Then threshold predicted net to (Positive / Negative / Neutral).

Notes vs the original lecture formula:
* lag-48 (corr 0.31) + lag-168 (corr 0.30) — lag-24 is risky because the
  EPIAS publish delay (6-12h) can make hours 13-23 of d-1 unobserved at
  prediction time, so half a day's worth of lag-24 values would be NaN.
* `s(time)` (long-run trend) is dropped: training on a sliding 180-day window
  already absorbs slow trends, and pygam fits get noticeably faster without it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s

from src.data import coerce_predictions, net_to_label, predict_date_window
from src.features import build_features

TRAIN_DAYS = 180

# Feature columns the GAM consumes (order matters — pygam uses column indices).
FEATURE_ORDER = [
    "hour",        # 0  cyclic spline
    "doy",         # 1  cyclic spline
    "temp_avg",    # 2  smooth
    "net_lag_48",  # 3  smooth
    "net_lag_168", # 4  smooth
    "dow",         # 5  factor
]

GAM_TERMS = (
    s(0, n_splines=12, basis="cp")
    + s(1, n_splines=10, basis="cp")
    + s(2, n_splines=8)
    + s(3, n_splines=10)
    + s(4, n_splines=10)
    + f(5)
)


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)

    train_start = (start_hour - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    feats = build_features(end_dt=end_hour, start=train_start, as_of_dt=as_of_dt)

    train = feats.dropna(subset=["net"] + FEATURE_ORDER).copy()
    if train.empty:
        return ["Neutral"] * 24

    X_train = train[FEATURE_ORDER].to_numpy(dtype=float)
    y_train = train["net"].to_numpy(dtype=float)

    gam = LinearGAM(GAM_TERMS).fit(X_train, y_train)

    target = (
        feats[(feats["dt"] >= start_hour) & (feats["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
    )
    if len(target) < 24 or target[FEATURE_ORDER].isna().any().any():
        return ["Neutral"] * 24

    X_pred = target[FEATURE_ORDER].to_numpy(dtype=float)
    preds_net = gam.predict(X_pred)

    raw = [net_to_label(v) for v in preds_net]
    return coerce_predictions(raw)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
