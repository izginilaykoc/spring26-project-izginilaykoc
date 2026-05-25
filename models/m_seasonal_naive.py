"""Seasonal naive: predict each hour with the system_direction observed exactly
1 week (168 hours) earlier.

Strong baseline because Turkish electricity demand is dominated by a 168-hour cycle
(weekday-of-week structure + 24 h diurnal cycle).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data import (
    VALID_LABELS,
    coerce_predictions,
    load_imbalance_grid,
    predict_date_window,
)


def predict(
    predict_date: pd.Timestamp,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> list[str]:
    start_hour, end_hour = predict_date_window(predict_date)
    grid = load_imbalance_grid(end_dt=end_hour, as_of_dt=as_of_dt)

    grid["direction_lag_168"] = grid["system_direction"].shift(168)
    target_rows = grid[(grid["dt"] >= start_hour) & (grid["dt"] <= end_hour)]
    raw = target_rows["direction_lag_168"].tolist()

    fixed = [p if p in VALID_LABELS else "Neutral" for p in raw]
    return coerce_predictions(fixed)


if __name__ == "__main__":
    tomorrow = (pd.Timestamp.now(tz="Europe/Istanbul") + pd.Timedelta(days=1)).normalize()
    print(predict(tomorrow))
