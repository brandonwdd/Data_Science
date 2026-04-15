from __future__ import annotations

import re

import numpy as np
import pandas as pd


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename MultiIndex columns to single strings: prefix_COL_agg."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{prefix}_{a}_{b}" if b else f"{prefix}_{a}" for a, b in df.columns]
    else:
        df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


def sanitize_feature_names(cols: list[str]) -> list[str]:
    """LightGBM / sklearn: avoid special characters in feature names."""
    out = []
    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        if s and s[0].isdigit():
            s = "f_" + s
        out.append(s)
    return out


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce float/int memory where safe."""
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov–Smirnov separation between score distributions of class 0 vs 1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    s0 = np.sort(y_score[y_true == 0])
    s1 = np.sort(y_score[y_true == 1])
    if len(s0) == 0 or len(s1) == 0:
        return float("nan")
    # Empirical CDFs on a combined grid of unique scores
    scores = np.sort(np.unique(np.concatenate([s0, s1])))
    cdf0 = np.searchsorted(s0, scores, side="right") / len(s0)
    cdf1 = np.searchsorted(s1, scores, side="right") / len(s1)
    return float(np.max(np.abs(cdf1 - cdf0)))
