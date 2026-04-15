"""LightGBM demand model with time-based split."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CATEGORICAL_FEATURES = [
    "product_id",
    "store_location",
    "product_category",
    "product_type",
]
NUMERIC_FEATURES = [
    "avg_price",
    "log_avg_price",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1_qty",
    "roll_7_qty",
    "roll_14_qty",
]
TARGET = "total_quantity"


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns()].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    y = df[TARGET].astype(float)
    return X, y


def time_based_split(
    df: pd.DataFrame, holdout_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = df["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train = df[df["transaction_date"] <= cutoff].copy()
    test = df[df["transaction_date"] > cutoff].copy()
    return train, test, cutoff


def train_lgbm(
    train_df: pd.DataFrame,
    lgbm_params: dict,
) -> lgb.LGBMRegressor:
    X_tr, y_tr = prepare_features(train_df)
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_tr, y_tr)
    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def predict_quantities(model: lgb.LGBMRegressor, df: pd.DataFrame) -> np.ndarray:
    X, _ = prepare_features(df)
    pred = model.predict(X)
    return np.clip(pred, 0.0, None)


def save_model(model: lgb.LGBMRegressor, path: str | Path) -> None:
    joblib.dump(model, path)


def load_model(path: str | Path) -> lgb.LGBMRegressor:
    return joblib.load(path)
