"""Shared data loading + as-of masking.

`load_imbalance_grid` is the single source of truth for every model:
  * pulls the imbalance series via classify_helper.get_imbalance_data
  * lays it on a gapless hourly grid (so .shift(168) really means "one week ago")
  * blanks out anything after `as_of_dt` so backtests stay honest
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from classify_helper import (
    get_historical_weather,
    get_imbalance_data,
    get_weather_forecast,
)

TZ = "Europe/Istanbul"

# Major Turkish demand centers — Istanbul, Ankara, Izmir.
DEFAULT_COORDS = (
    (41.0082, 28.9784),
    (39.9334, 32.8597),
    (38.4192, 27.1287),
)
DEFAULT_WEATHER_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
)


@lru_cache(maxsize=1)
def _get_imbalance_cached() -> pd.DataFrame:
    """Cached imbalance fetch — the Google Sheet read is slow.

    Cleared by importing `clear_cache` and calling it.
    """
    df = get_imbalance_data().copy()
    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(TZ)
    return df.sort_values("dt").reset_index(drop=True)


def clear_cache() -> None:
    _get_imbalance_cached.cache_clear()


def load_imbalance_grid(
    start: str = "2024-01-01",
    end_dt: Optional[pd.Timestamp] = None,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Hourly imbalance grid masked at `as_of_dt`.

    Parameters
    ----------
    start : str
        First date in the returned grid (inclusive).
    end_dt : Timestamp, optional
        Last hour in the returned grid (inclusive). Defaults to `predict_date 23:00`
        when called via `predict(...)`. If None, uses last imbalance row available.
    as_of_dt : Timestamp, optional
        Pretend data after this timestamp is unknown — used to simulate the EPIAS
        publish delay during backtest. If None, returns whatever is currently in the
        spreadsheet (i.e. real-time).

    Returns
    -------
    DataFrame with columns: dt, net, system_direction (NaN past as_of_dt).
    """
    raw = _get_imbalance_cached()

    if end_dt is None:
        end_dt = raw["dt"].max()
    end_dt = pd.Timestamp(end_dt)
    end_dt = end_dt.tz_convert(TZ) if end_dt.tzinfo else end_dt.tz_localize(TZ)

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(TZ)

    grid = pd.DataFrame(
        {"dt": pd.date_range(start=start_ts, end=end_dt, freq="1h", tz=TZ)}
    )

    keep = raw[["dt", "net", "system_direction"]].copy()
    if as_of_dt is not None:
        as_of_dt = pd.Timestamp(as_of_dt)
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.tz_localize(TZ)
        else:
            as_of_dt = as_of_dt.tz_convert(TZ)
        keep.loc[keep["dt"] > as_of_dt, ["net", "system_direction"]] = pd.NA

    out = grid.merge(keep, on="dt", how="left")
    return out


# ---------------------------------------------------------------------------
# Helpers used by every model
# ---------------------------------------------------------------------------

VALID_LABELS = {"Positive", "Negative", "Neutral"}


def net_to_label(net: float) -> str:
    """Project threshold: net >= 50 -> Positive, <= -50 -> Negative, else Neutral."""
    if pd.isna(net):
        return "Neutral"  # safe fallback
    if net >= 50:
        return "Positive"
    if net <= -50:
        return "Negative"
    return "Neutral"


def coerce_predictions(preds) -> list[str]:
    """Sanitize model output: must be exactly 24 valid labels."""
    out: list[str] = []
    for p in preds:
        out.append(p if p in VALID_LABELS else "Neutral")
    if len(out) != 24:
        raise ValueError(f"Expected 24 predictions, got {len(out)}")
    return out


@lru_cache(maxsize=4)
def _get_weather_cached(
    start_date: str,
    variables: tuple,
    coordinates: tuple,
    get_forecast_data: bool,
) -> pd.DataFrame:
    """Cached historical weather pull. Coordinates / variables are tuples to be hashable."""
    df = get_historical_weather(
        start_date=start_date,
        variables=list(variables),
        coordinates=[list(c) for c in coordinates],
        get_forecast_data=get_forecast_data,
    ).copy()
    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(TZ)
    return df.sort_values("dt").reset_index(drop=True)


@lru_cache(maxsize=4)
def _get_forecast_cached(
    forecast_days: int,
    past_days: int,
    variables: tuple,
    coordinates: tuple,
) -> pd.DataFrame:
    """Cached live-forecast pull (covers today−past_days .. today+forecast_days)."""
    df = get_weather_forecast(
        forecast_days=forecast_days,
        past_days=past_days,
        variables=list(variables),
        coordinates=[list(c) for c in coordinates],
    ).copy()
    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(TZ)
    return df.sort_values("dt").reset_index(drop=True)


def load_weather(
    start: str = "2024-01-01",
    variables: tuple = DEFAULT_WEATHER_VARS,
    coordinates: tuple = DEFAULT_COORDS,
    get_forecast_data: bool = True,
    splice_forecast: bool = True,
) -> pd.DataFrame:
    """Hourly weather over `start..now+2d`.

    `get_historical_weather` only returns data up to today−2d. When
    `splice_forecast=True` (default), we splice `get_weather_forecast(past_days=7,
    forecast_days=2)` on top so models can see the last 2 days and tomorrow.
    """
    archive = _get_weather_cached(start, variables, coordinates, get_forecast_data).copy()
    if not splice_forecast:
        return archive

    forecast = _get_forecast_cached(2, 7, variables, coordinates).copy()
    archive_max = archive["dt"].max()
    forecast_tail = forecast[forecast["dt"] > archive_max]
    return pd.concat([archive, forecast_tail], ignore_index=True).sort_values("dt").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame, dt_col: str = "dt") -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[dt_col].dt.hour
    out["dayofweek"] = out[dt_col].dt.dayofweek
    out["month"] = out[dt_col].dt.month
    return out


def predict_date_window(
    predict_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_hour, end_hour) for the 24 hours of `predict_date`."""
    if predict_date.tzinfo is None:
        predict_date = predict_date.tz_localize(TZ)
    else:
        predict_date = predict_date.tz_convert(TZ)
    day = predict_date.normalize()
    return day, day + pd.Timedelta(hours=23)
