"""Single source of truth for model features.

`build_features(end_dt, start, as_of_dt)` returns one DataFrame indexed by hour
covering `start..end_dt`, with all the features Phase 4/5 models will consume.

EDA-driven shortlist (see notebooks/01_eda.ipynb takeaways):
  * Calendar:    hour (sin/cos), dow, is_weekend, month, doy (sin/cos)
  * Net lags:    24, 48, 72, 168, 336 (skip <=12 due to EPIAS publish delay)
  * Lag deltas:  Δ168 = lag168 - lag336 (week-over-week trend)
  * Weather:     per-city {temperature, humidity, wind, cloud}; HDD/CDD off 18°C
  * Aggregates:  temp_avg / humidity_avg / wind_avg / cloud_avg across cities
  * Interactions: hour × is_weekend, temp_avg × hour
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data import load_imbalance_grid, load_weather

LAGS = (24, 48, 72, 168, 336)
HDD_BASE = 18.0
CDD_BASE = 18.0


def _cyclic(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def _city_temp_cols(weather: pd.DataFrame) -> list[str]:
    return [c for c in weather.columns if "temperature_2m" in c]


def _city_short(col: str) -> str:
    """Extract 'location_000' from 'location_000 temperature_2m'."""
    return col.split(" ", 1)[0]


def build_features(
    end_dt: pd.Timestamp,
    start: str = "2024-01-01",
    as_of_dt: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Hourly feature frame from `start` through `end_dt`.

    Lag columns are NaN where they reach into pre-`start` data; the caller
    typically wants to drop those rows for training and keep the prediction-day
    rows for inference.

    `as_of_dt` is forwarded to `load_imbalance_grid` so historical net/lag
    features are correctly masked during backtest.
    """
    grid = load_imbalance_grid(start=start, end_dt=end_dt, as_of_dt=as_of_dt)
    weather = load_weather(start=start)

    df = grid.merge(weather, on="dt", how="left")

    df["hour"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["dt"].dt.month
    df["doy"] = df["dt"].dt.dayofyear
    df["hour_sin"], df["hour_cos"] = _cyclic(df["hour"], 24)
    df["doy_sin"], df["doy_cos"] = _cyclic(df["doy"], 365)

    for lag in LAGS:
        df[f"net_lag_{lag}"] = df["net"].shift(lag)
    df["net_diff_168"] = df["net_lag_168"] - df["net_lag_336"]

    temp_cols = _city_temp_cols(df)
    cities = [_city_short(c) for c in temp_cols]

    for c in cities:
        t = df[f"{c} temperature_2m"]
        df[f"{c}_hdd"] = (HDD_BASE - t).clip(lower=0)
        df[f"{c}_cdd"] = (t - CDD_BASE).clip(lower=0)

    for var in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"):
        cols = [f"{c} {var}" for c in cities if f"{c} {var}" in df.columns]
        if cols:
            short = {
                "temperature_2m": "temp",
                "relative_humidity_2m": "humidity",
                "wind_speed_10m": "wind",
                "cloud_cover": "cloud",
            }[var]
            df[f"{short}_avg"] = df[cols].mean(axis=1)
    df["hdd_avg"] = df[[f"{c}_hdd" for c in cities]].mean(axis=1)
    df["cdd_avg"] = df[[f"{c}_cdd" for c in cities]].mean(axis=1)

    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
    df["temp_x_hour"] = df["temp_avg"] * df["hour"]

    return df


# Convenience for downstream models — feature-name groups they typically want.
CALENDAR_COLS = ["hour", "dow", "is_weekend", "month", "doy",
                 "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
LAG_COLS = [f"net_lag_{L}" for L in LAGS] + ["net_diff_168"]
WEATHER_AVG_COLS = ["temp_avg", "humidity_avg", "wind_avg", "cloud_avg",
                    "hdd_avg", "cdd_avg"]
INTERACTION_COLS = ["hour_x_weekend", "temp_x_hour"]


def per_city_weather_cols(df: pd.DataFrame) -> list[str]:
    """Per-city raw weather + HDD/CDD columns present in `df`."""
    cities = [_city_short(c) for c in _city_temp_cols(df)]
    raw = [c for c in df.columns
           if any(c.startswith(f"{city} ") for city in cities)]
    derived = [f"{city}_{kind}" for city in cities for kind in ("hdd", "cdd")]
    return raw + [c for c in derived if c in df.columns]


def all_feature_cols(df: pd.DataFrame) -> list[str]:
    """Full feature list for tree-based models — includes per-city weather."""
    return CALENDAR_COLS + LAG_COLS + WEATHER_AVG_COLS + per_city_weather_cols(df) + INTERACTION_COLS


# Static name used for backwards compatibility (averages only — used by
# linear models that don't want the high-dim per-city block).
ALL_FEATURE_COLS = CALENDAR_COLS + LAG_COLS + WEATHER_AVG_COLS + INTERACTION_COLS


if __name__ == "__main__":
    end = pd.Timestamp.now(tz="Europe/Istanbul").normalize() + pd.Timedelta(hours=23)
    feats = build_features(end_dt=end)
    print(feats.shape)
    print(feats[["dt"] + ALL_FEATURE_COLS].tail(24))
