"""Lag, rolling, and calendar features on the hourly demand series."""

import pandas as pd

from config import FEATURE_COLUMNS


def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using only past demand (no leakage from the same hour).
    Rows with NaNs in any model feature are left in the frame for inspection;
    training code should drop them via dropna(subset=FEATURE_COLUMNS).
    """
    df = hourly.sort_values("hour").reset_index(drop=True)
    d = df["demand"]

    df["demand_lag_1"] = d.shift(1)
    df["demand_lag_24"] = d.shift(24)
    df["demand_lag_168"] = d.shift(168)

    past = d.shift(1)
    df["rolling_mean_3"] = past.rolling(window=3, min_periods=3).mean()
    df["rolling_mean_24"] = past.rolling(window=24, min_periods=24).mean()
    df["rolling_mean_168"] = past.rolling(window=168, min_periods=168).mean()

    dt = df["hour"]
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return df


def attach_hour_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with full features, keeping the hour column for evaluation plots."""
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
