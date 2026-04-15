from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import config


def _categorical_features(X: pd.DataFrame) -> list[str] | None:
    cats = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c])]
    return cats if cats else None


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> LGBMClassifier:
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    params = {**config.LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = LGBMClassifier(**params)
    cat_feats = _categorical_features(X_train)
    fit_kw: dict = {}
    if cat_feats:
        fit_kw["categorical_feature"] = cat_feats
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
            log_evaluation(period=0),
        ],
        **fit_kw,
    )
    return model


def predict_proba_positive(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1]
