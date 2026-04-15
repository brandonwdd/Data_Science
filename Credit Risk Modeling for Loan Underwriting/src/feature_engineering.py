from __future__ import annotations

import numpy as np
import pandas as pd

from src.aggregation import build_all_aggregates
from src.utils import sanitize_feature_names


def merge_application_with_aggregates(
    application: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> pd.DataFrame:
    out = application.merge(aggregates, on="SK_ID_CURR", how="left")
    agg_cols = [c for c in aggregates.columns if c != "SK_ID_CURR"]
    for c in agg_cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0)
    return out


def encode_object_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_category_codes: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode object/string columns: low-cardinality -> pandas category (for LightGBM);
    high-cardinality -> integer codes fit on train only (unseen in test -> -1).
    """
    train = train.copy()
    test = test.copy()
    obj_cols = [
        c
        for c in train.columns
        if c != "TARGET" and train[c].dtype == object and c in test.columns
    ]
    for c in obj_cols:
        tr = train[c].fillna("missing").astype(str)
        te = test[c].fillna("missing").astype(str)
        nuni = tr.nunique(dropna=False)
        if nuni <= max_category_codes:
            cats = pd.unique(pd.concat([tr, te], ignore_index=True))
            train[c] = pd.Categorical(tr, categories=cats)
            test[c] = pd.Categorical(te, categories=cats)
        else:
            codes_tr, uniques = pd.factorize(tr, sort=False)
            mapping = {u: i for i, u in enumerate(uniques)}
            train[c] = codes_tr.astype(np.int32)
            test[c] = te.map(mapping).fillna(-1).astype(np.int32)
    return train, test


def ordered_train_val_split(
    df: pd.DataFrame,
    order_cols: str | list[str],
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last val_fraction of rows after sorting by order_cols (non-random holdout)."""
    cols = [order_cols] if isinstance(order_cols, str) else list(order_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"order columns missing from training frame: {missing}")
    coerced = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in cols},
        index=df.index,
    )
    order_idx = coerced.sort_values(list(coerced.columns), kind="mergesort").index
    sorted_df = df.loc[order_idx].reset_index(drop=True)
    n = len(sorted_df)
    n_val = max(1, int(n * val_fraction))
    val_df = sorted_df.iloc[-n_val:].reset_index(drop=True)
    train_df = sorted_df.iloc[:-n_val].reset_index(drop=True)
    if len(train_df) < 1000:
        raise ValueError("Training split too small; lower val_fraction or use more data.")
    return train_df, val_df


def prepare_xy(
    df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
    target_col: str = "TARGET",
) -> tuple[pd.DataFrame, pd.Series | None]:
    y = None
    if target_col in df.columns:
        y = df[target_col].astype(np.int32)
    drop_cols = [id_col]
    if target_col in df.columns:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols)
    return X, y


def rename_features_sanitized(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return X with safe column names and mapping sanitized -> original."""
    orig = list(X.columns)
    new = sanitize_feature_names(orig)
    mapping = dict(zip(new, orig))
    X = X.copy()
    X.columns = new
    return X, mapping


def build_feature_tables(
    app_train: pd.DataFrame,
    app_test: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous_application: pd.DataFrame,
    pos_cash: pd.DataFrame,
    credit_card: pd.DataFrame,
    installments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregates = build_all_aggregates(
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        credit_card,
        installments,
    )
    train_full = merge_application_with_aggregates(app_train, aggregates)
    test_full = merge_application_with_aggregates(app_test, aggregates)
    train_full, test_full = encode_object_columns(train_full, test_full)
    return train_full, test_full
