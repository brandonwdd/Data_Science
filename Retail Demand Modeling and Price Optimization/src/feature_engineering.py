"""Time-based features and lags on the daily panel."""
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["transaction_date"]
    out["day_of_week"] = dt.dt.dayofweek.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["product_id", "store_location", "transaction_date"])
    g = out.groupby(["product_id", "store_location"], sort=False)["total_quantity"]

    out["lag_1_qty"] = g.shift(1)
    out["roll_7_qty"] = g.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    out["roll_14_qty"] = g.transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).mean()
    )

    for col in ["lag_1_qty", "roll_7_qty", "roll_14_qty"]:
        out[col] = out[col].fillna(0.0)
    return out.reset_index(drop=True)


def add_price_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_avg_price"] = np.log(out["avg_price"].clip(lower=1e-6))
    return out


def build_modeling_frame(daily: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(daily)
    df = add_lag_features(df)
    df = add_price_transforms(df)
    return df
