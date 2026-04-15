"""Inference on a feature matrix."""

import pandas as pd
from sklearn.base import BaseEstimator


def predict_demand(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    pred = model.predict(X)
    return pd.Series(pred, index=X.index, name="predicted_demand")
