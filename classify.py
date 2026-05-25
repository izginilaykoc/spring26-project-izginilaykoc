"""classify.py — System Imbalance Direction Classifier for Turkish Electricity Markets.
IE 48B Spring 2026.

Self-contained submission: the grader only copies `classify.py` and the
`classify_helper` files into its workspace, so all model logic must live
inline. Importing from `models/` or `src/` would crash on the grader and
trigger the all-Neutral fallback (4.17% / day on the leaderboard).

Pipeline:
  1. Pull imbalance from the shared Google Sheet via classify_helper.
  2. Pull weather (archive + 2-day forecast splice) for ~3 Turkish demand
     centers via Open-Meteo.
  3. Train a RandomForestClassifier directly on label
     (Positive/Negative/Neutral) using weather + calendar + cyclical
     encodings + net_lag_{168,336} + dir_lag_{168,336} one-hots, over the
     trailing 180 days.
  4. Safety chain: RF fails -> seasonal-naive (168 h lag) -> all-Neutral.

Lags are kept >= 168h so the model is robust to the variable 6-12h EPIAS
publish delay (target hour 23 needs lag >= ~35h to be safe; 168h is far
beyond that).
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classify_helper import (
    get_historical_weather,
    get_imbalance_data,
    get_weather_forecast,
)

TZ = "Europe/Istanbul"
VALID_LABELS = {"Positive", "Negative", "Neutral"}

DEFAULT_COORDS = (
    (41.0082, 28.9784),   # Istanbul
    (39.9334, 32.8597),   # Ankara
    (38.4192, 27.1287),   # Izmir
)
DEFAULT_WEATHER_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
)
TRAIN_DAYS = 180


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_imbalance() -> pd.DataFrame:
    df = get_imbalance_data().copy()
    df["dt"] = pd.to_datetime(df["dt"])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(TZ)
    return df.sort_values("dt").reset_index(drop=True)


def load_imbalance_grid(start: str, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Hourly imbalance grid from `start` to `end_dt`, gapless so .shift(168)
    means 'exactly one week ago' even when the source has holes."""
    raw = _load_imbalance()
    end_dt = pd.Timestamp(end_dt)
    end_dt = end_dt.tz_convert(TZ) if end_dt.tzinfo else end_dt.tz_localize(TZ)
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(TZ)
    grid = pd.DataFrame({"dt": pd.date_range(start=start_ts, end=end_dt, freq="1h", tz=TZ)})
    keep = raw[["dt", "net", "system_direction"]].copy()
    return grid.merge(keep, on="dt", how="left")


def load_weather(start: str) -> pd.DataFrame:
    """Hourly weather over `start..tomorrow`. Splices a 2-day forecast on top
    of the historical archive so tomorrow is covered."""
    archive = get_historical_weather(
        start_date=start,
        variables=list(DEFAULT_WEATHER_VARS),
        coordinates=[list(c) for c in DEFAULT_COORDS],
        get_forecast_data=True,
    ).copy()
    archive["dt"] = pd.to_datetime(archive["dt"])
    if archive["dt"].dt.tz is None:
        archive["dt"] = archive["dt"].dt.tz_localize(TZ)

    forecast = get_weather_forecast(
        forecast_days=2,
        past_days=7,
        variables=list(DEFAULT_WEATHER_VARS),
        coordinates=[list(c) for c in DEFAULT_COORDS],
    ).copy()
    forecast["dt"] = pd.to_datetime(forecast["dt"])
    if forecast["dt"].dt.tz is None:
        forecast["dt"] = forecast["dt"].dt.tz_localize(TZ)

    archive_max = archive["dt"].max()
    forecast_tail = forecast[forecast["dt"] > archive_max]
    return (
        pd.concat([archive, forecast_tail], ignore_index=True)
        .sort_values("dt")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["dt"].dt.hour
    out["dayofweek"] = out["dt"].dt.dayofweek
    out["month"] = out["dt"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["hour_of_week"] = out["dayofweek"] * 24 + out["hour"]
    return out


def _net_to_label(net: float) -> str:
    if pd.isna(net):
        return "Neutral"
    if net >= 50:
        return "Positive"
    if net <= -50:
        return "Negative"
    return "Neutral"


def _validate(preds) -> list[str]:
    """Coerce to exactly 24 valid labels."""
    if not isinstance(preds, (list, tuple)):
        raise ValueError("predict() must return list/tuple")
    out = [p if p in VALID_LABELS else "Neutral" for p in preds]
    if len(out) < 24:
        out += ["Neutral"] * (24 - len(out))
    elif len(out) > 24:
        out = out[:24]
    return out


def _predict_window(predict_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if predict_date.tzinfo is None:
        predict_date = predict_date.tz_localize(TZ)
    else:
        predict_date = predict_date.tz_convert(TZ)
    day = predict_date.normalize()
    return day, day + pd.Timedelta(hours=23)


def _build_features(predict_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (train, target, feature_cols)."""
    start_hour, end_hour = _predict_window(predict_date)
    grid = load_imbalance_grid("2024-01-01", end_hour)

    # Safe lags (>> EPIAS publish delay)
    grid["net_lag_168"] = grid["net"].shift(168)
    grid["net_lag_336"] = grid["net"].shift(336)
    grid["dir_lag_168"] = grid["system_direction"].shift(168)
    grid["dir_lag_336"] = grid["system_direction"].shift(336)

    last_obs_dt = grid.dropna(subset=["net"])["dt"].max()
    if pd.isna(last_obs_dt):
        raise RuntimeError("No imbalance observations available")

    train_start = (last_obs_dt - pd.Timedelta(days=TRAIN_DAYS)).strftime("%Y-%m-%d")
    weather = _add_calendar(load_weather(start=train_start))
    merged = pd.merge(weather, grid, on="dt", how="inner")

    # One-hot the direction lags
    for col in ("dir_lag_168", "dir_lag_336"):
        for lab in ("Positive", "Negative", "Neutral"):
            merged[f"{col}_{lab}"] = (merged[col] == lab).astype(float)
        merged.drop(columns=[col], inplace=True)

    feature_cols = [
        c for c in merged.columns
        if c not in ("dt", "net", "system_direction")
    ]

    lag_cols = ["net_lag_168", "net_lag_336"]
    train = merged.dropna(subset=["net"] + lag_cols).copy()

    target = (
        merged[(merged["dt"] >= start_hour) & (merged["dt"] <= end_hour)]
        .sort_values("dt")
        .head(24)
        .copy()
    )
    return train, target, feature_cols


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def predict_rf(predict_date: pd.Timestamp) -> list[str]:
    """RandomForestClassifier on direct labels (Positive/Negative/Neutral).
    Backtested best on April 1 – May 17 window (62.8% mean accuracy)."""
    train, target, feat = _build_features(predict_date)
    if train.empty or len(target) < 24 or target[feat].isna().any().any():
        return ["Neutral"] * 24

    y_train = train["net"].apply(_net_to_label)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=0,
    )
    model.fit(train[feat], y_train)
    return list(model.predict(target[feat]))


def predict_seasonal_naive(predict_date: pd.Timestamp) -> list[str]:
    """Each hour := system_direction from exactly 168h (one week) earlier."""
    start_hour, end_hour = _predict_window(predict_date)
    grid = load_imbalance_grid("2024-01-01", end_hour)
    grid["direction_lag_168"] = grid["system_direction"].shift(168)
    target_rows = grid[(grid["dt"] >= start_hour) & (grid["dt"] <= end_hour)]
    raw = target_rows["direction_lag_168"].tolist()
    return [p if p in VALID_LABELS else "Neutral" for p in raw]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tomorrow = (pd.Timestamp.now(tz=TZ) + pd.Timedelta(days=1)).normalize()

    try:
        predictions = _validate(predict_rf(tomorrow))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("[classify] random_forest failed; falling back to seasonal_naive", file=sys.stderr)
        try:
            predictions = _validate(predict_seasonal_naive(tomorrow))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print("[classify] seasonal_naive also failed; emitting Neutral baseline", file=sys.stderr)
            predictions = ["Neutral"] * 24

    # Final required output — must be the last printed line.
    print(predictions)


if __name__ == "__main__":
    main()
