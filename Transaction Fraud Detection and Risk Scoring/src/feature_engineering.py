from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.utils import add_missingness_features


def add_simple_behavioral_features(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    Add train-fitted frequency count features for a few key identifiers.
    This avoids leakage by fitting counts on train only, then mapping to val/test.
    """
    out = other.copy()
    keys: list[str] = []
    for c in ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceType"]:
        if c in train.columns and c in out.columns:
            keys.append(c)
    for c in keys:
        counts = train[c].value_counts(dropna=False)
        out[f"{c}_freq"] = out[c].map(counts).fillna(0).astype(np.float32)
    # Email domain match signal
    if "P_emaildomain" in out.columns and "R_emaildomain" in out.columns:
        p = out["P_emaildomain"].fillna("").astype(str)
        r = out["R_emaildomain"].fillna("").astype(str)
        out["email_domain_match"] = ((p == r) & (p != "") & (r != "")).astype(np.int8)
        out["email_missing_any"] = ((out["P_emaildomain"].isna()) | (out["R_emaildomain"].isna())).astype(
            np.int8
        )
    return out


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness summary features; leave raw columns for LightGBM."""
    out = add_missingness_features(df)
    return out


def encode_objects_train_val_test(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    max_cardinality: int = config.MAX_CATEGORY_CARDINALITY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Low-cardinality object -> pandas category using train-only categories.
    High-cardinality object -> train-only factorize, unseen -> -1 codes.
    """
    tr = train.copy()
    va = val.copy()
    te = test.copy()

    obj_cols = [c for c in tr.columns if tr[c].dtype == object and c in va.columns and c in te.columns]
    for c in obj_cols:
        a = tr[c].fillna("missing").astype(str)
        b = va[c].fillna("missing").astype(str)
        d = te[c].fillna("missing").astype(str)
        nuni = a.nunique(dropna=False)
        if nuni <= max_cardinality:
            cats = pd.unique(a)
            tr[c] = pd.Categorical(a, categories=cats)
            va[c] = pd.Categorical(b, categories=cats)
            te[c] = pd.Categorical(d, categories=cats)
        else:
            codes, uniques = pd.factorize(a, sort=False)
            mapping = {u: i for i, u in enumerate(uniques)}
            tr[c] = codes.astype(np.int32)
            va[c] = b.map(mapping).fillna(-1).astype(np.int32)
            te[c] = d.map(mapping).fillna(-1).astype(np.int32)
    return tr, va, te


def split_train_val_by_time(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by TransactionDT (chronology). If input is already sorted, this is a stable tail split.
    """
    df = df.sort_values(config.TIME_COL, kind="mergesort").reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * val_fraction))
    return df.iloc[:-n_val].reset_index(drop=True), df.iloc[-n_val:].reset_index(drop=True)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[config.LABEL_COL].astype(np.int32)
    X = df.drop(columns=[config.LABEL_COL])
    return X, y

