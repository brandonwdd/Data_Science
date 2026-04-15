"""Train LightGBM with time-based validation slice for early stopping."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import config


def time_validation_mask(
    last_purchase_date: pd.Series,
    cutoff_date: pd.Timestamp,
    observation_days: int,
    last_fraction: float = 0.2,
) -> np.ndarray:
    """
    Customers whose last purchase (calendar day) falls in the last `last_fraction`
    of the observation window are used for early-stopping validation.
    """
    cutoff_norm = pd.Timestamp(cutoff_date).normalize()
    obs_start = cutoff_norm - pd.Timedelta(days=observation_days - 1)
    window_len = (cutoff_norm - obs_start).days + 1
    val_start = obs_start + pd.Timedelta(days=int((1.0 - last_fraction) * window_len))
    lp = pd.to_datetime(last_purchase_date).dt.normalize()
    return (lp >= val_start).to_numpy()


def train_with_time_valid(
    X: pd.DataFrame,
    y: pd.Series,
    last_purchase_date: pd.Series,
    cutoff_date: pd.Timestamp,
    observation_days: int,
    model_dir: Path,
) -> tuple[LGBMClassifier, list[str]]:
    valid_mask = time_validation_mask(
        last_purchase_date, cutoff_date, observation_days, last_fraction=0.2
    )
    X_tr = X.loc[~valid_mask]
    y_tr = y.loc[~valid_mask]
    X_va = X.loc[valid_mask]
    y_va = y.loc[valid_mask]

    if len(X_va) < 100 or len(X_tr) < 100:
        # Fallback: still keep temporal idea by using last 20% of rows order (deterministic)
        split = int(len(X) * 0.8)
        X_tr, X_va = X.iloc[:split], X.iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]

    model = LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=config.ES_ROUNDS),
            log_evaluation(period=0),
        ],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "lgbm_propensity.joblib")
    joblib.dump(list(X.columns), model_dir / "feature_columns.joblib")
    return model, list(X.columns)


def feature_importance_df(model: LGBMClassifier, feature_names: list[str]) -> pd.DataFrame:
    imp = model.booster_.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": feature_names, "gain": imp})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
