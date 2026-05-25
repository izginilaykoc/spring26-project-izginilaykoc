"""Trend-adjusted seasonal naive on net.

predicted_net = net_lag_168 + (net_lag_168 - net_lag_336)

Captures week-over-week drift on top of the weekly cycle. Falls back to plain
seasonal naive (net_lag_168) when the 2-weeks-ago observation is missing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data import (
    coerce_predictions,
    load_imbalance_grid,
    net_to_label,
    predict_date_window,
)


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)
    grid = load_imbalance_grid(end_dt=end_hour, as_of_dt=as_of_dt)

    grid["net_lag_168"] = grid["net"].shift(168)
    grid["net_lag_336"] = grid["net"].shift(336)

    trend = grid["net_lag_168"] - grid["net_lag_336"]
    grid["predicted_net"] = np.where(
        grid["net_lag_336"].notna(),
        grid["net_lag_168"] + trend,    # trend-adjusted
        grid["net_lag_168"],            # fallback: plain seasonal naive
    )

    target_rows = grid[(grid["dt"] >= start_hour) & (grid["dt"] <= end_hour)]
    raw = [net_to_label(v) for v in target_rows["predicted_net"]]
    return coerce_predictions(raw)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
