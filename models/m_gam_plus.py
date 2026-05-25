"""GAM+ — Lecture 8b improvements over m_gam_baseline.

Three changes vs the baseline:

  1) Tensor product `te(temp_avg, hour)` — temperature behaves differently at
     peak vs off-peak hours; the additive `s(temp) + s(hour)` baseline can't
     capture that interaction.
  2) Predict `delta = net - net_lag_168` (week-over-week change) and reconstruct
     `net_hat = lag168 + delta_hat`. Lecture 8b's trick: regressing on the
     deviation from a strong seasonal anchor often beats regressing on the
     level when the anchor is informative — even though our 168h cycle is
     modest, lag_168 is still our strongest safe predictor.
  3) Stronger smoothing penalty (`lam` ↑) to reduce overfit on the noisier
     imbalance series vs the consumption series in lecture.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s, te

from src.data import coerce_predictions, net_to_label, predict_date_window
from src.features import build_features

TRAIN_DAYS = 180
LAM = 5.0  # smoothing penalty (default 0.6) — stronger to fight noise

FEATURE_ORDER = [
    "hour",        # 0  cyclic
    "doy",         # 1  cyclic
    "temp_avg",    # 2  used in te(2,0) tensor + s(2)
    "humidity_avg",# 3  smooth
    "net_lag_48",  # 4  smooth
    "net_lag_168", # 5  smooth (only used as anchor; we predict delta)
    "dow",         # 6  factor
    "is_weekend",  # 7  factor
]

GAM_TERMS = (
    s(0, n_splines=12, basis="cp", lam=LAM)
    + s(1, n_splines=10, basis="cp", lam=LAM)
    + te(2, 0, n_splines=[6, 8], lam=LAM)   # temp × hour interaction
    + s(3, n_splines=8, lam=LAM)
    + s(4, n_splines=10, lam=LAM)
    + f(6)
    + f(7)
)


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)

    train_start = (start_hour - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    feats = build_features(end_dt=end_hour, start=train_start, as_of_dt=as_of_dt)

    feats["delta"] = feats["net"] - feats["net_lag_168"]

    needed = ["delta"] + FEATURE_ORDER
    train = feats.dropna(subset=needed).copy()
    if train.empty:
        return ["Neutral"] * 24

    X_train = train[FEATURE_ORDER].to_numpy(dtype=float)
    y_train = train["delta"].to_numpy(dtype=float)

    gam = LinearGAM(GAM_TERMS).fit(X_train, y_train)

    target = (
        feats[(feats["dt"] >= start_hour) & (feats["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
    )
    if len(target) < 24 or target[FEATURE_ORDER].isna().any().any() or target["net_lag_168"].isna().any():
        return ["Neutral"] * 24

    X_pred = target[FEATURE_ORDER].to_numpy(dtype=float)
    delta_hat = gam.predict(X_pred)
    net_hat = target["net_lag_168"].to_numpy() + delta_hat

    raw = [net_to_label(v) for v in net_hat]
    return coerce_predictions(raw)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
