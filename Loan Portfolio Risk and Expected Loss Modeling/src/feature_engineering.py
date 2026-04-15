from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.leakage_filters import is_leakage_column
from src.utils import sanitize_feature_names


def _drop_leakage_and_ids(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    drop = {c for c in df.columns if is_leakage_column(c)}
    drop.update({"id", "member_id", "url", "desc", "emp_title", "title"})
    if drop_target and config.TARGET_COL in df.columns:
        drop.add(config.TARGET_COL)
    cols = [c for c in df.columns if c not in drop]
    return df[cols].copy()


def add_calendar_features_from_issue(df: pd.DataFrame) -> pd.DataFrame:
    """issue_d must exist as datetime."""
    out = df.copy()
    d = out[config.ISSUE_DATE_COL]
    out["issue_year"] = d.dt.year
    out["issue_month"] = d.dt.month
    return out


def encode_objects(df: pd.DataFrame, max_category_codes: int = 2000) -> pd.DataFrame:
    out = df.copy()
    obj_cols = [c for c in out.columns if out[c].dtype == object]
    for c in obj_cols:
        tr = out[c].fillna("missing").astype(str)
        nuni = tr.nunique(dropna=False)
        if nuni <= max_category_codes:
            out[c] = pd.Categorical(tr)
        else:
            codes, uniques = pd.factorize(tr, sort=False)
            out[c] = codes.astype(np.int32)
    return out


def _impute_numeric(
    df: pd.DataFrame,
    medians: dict[str, float] | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    num_cols = [c for c in out.columns if c != config.TARGET_COL and pd.api.types.is_numeric_dtype(out[c])]
    new_medians: dict[str, float] = {}
    for c in num_cols:
        if medians is None:
            s = out[c]
            if s.notna().sum() == 0:
                med = 0.0
            else:
                m = s.median()
                med = float(m) if pd.notna(m) else 0.0
            new_medians[c] = med
        else:
            med = medians.get(c, 0.0)
        out[c] = out[c].fillna(med)
    if medians is None:
        return out, new_medians
    return out, medians


def build_features(
    df: pd.DataFrame,
    numeric_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calendar features, drop leakage + ids (keep target column until prepare_xy)."""
    df = add_calendar_features_from_issue(df)
    df = _drop_leakage_and_ids(df, drop_target=False)
    df, medians = _impute_numeric(df, numeric_medians)
    return df, medians


def encode_objects_train_val(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align object / string columns across train and validation (no target column)."""
    tr = train.copy()
    va = val.copy()
    obj_cols = [c for c in tr.columns if tr[c].dtype == object]
    for c in obj_cols:
        if c not in va.columns:
            continue
        a = tr[c].fillna("missing").astype(str)
        b = va[c].fillna("missing").astype(str)
        nuni = a.nunique(dropna=False)
        if nuni <= 2000:
            cats = pd.unique(pd.concat([a, b], ignore_index=True))
            tr[c] = pd.Categorical(a, categories=cats)
            va[c] = pd.Categorical(b, categories=cats)
        else:
            codes, uniques = pd.factorize(a, sort=False)
            mapping = {u: i for i, u in enumerate(uniques)}
            tr[c] = codes.astype(np.int32)
            va[c] = b.map(mapping).fillna(-1).astype(np.int32)
    return tr, va


def materialize_train_val_xy(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tr_feat, medians = build_features(train_raw, None)
    va_feat, _ = build_features(val_raw, medians)
    y_tr = tr_feat[config.TARGET_COL].astype(np.int32)
    y_va = va_feat[config.TARGET_COL].astype(np.int32)
    X_tr = tr_feat.drop(columns=[config.TARGET_COL])
    X_va = va_feat.drop(columns=[config.TARGET_COL])
    X_tr, X_va = encode_objects_train_val(X_tr, X_va)
    return X_tr, X_va, y_tr, y_va


def ordered_time_split(
    df: pd.DataFrame,
    issue_col: str,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier issue_d -> train; later -> validation."""
    if issue_col not in df.columns:
        raise KeyError(issue_col)
    sorted_df = df.sort_values(issue_col, kind="mergesort").reset_index(drop=True)
    n = len(sorted_df)
    n_val = max(1, int(n * val_fraction))
    val_df = sorted_df.iloc[-n_val:].reset_index(drop=True)
    train_df = sorted_df.iloc[:-n_val].reset_index(drop=True)
    min_train = 500 if config.ROW_CAP is not None and config.ROW_CAP < 50000 else 5000
    if len(train_df) < min_train:
        raise ValueError("Training set too small; increase data or lower val_fraction.")
    return train_df, val_df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[config.TARGET_COL].astype(np.int32)
    X = df.drop(columns=[config.TARGET_COL])
    return X, y


def rename_sanitized(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    orig = list(X.columns)
    new = sanitize_feature_names(orig)
    X = X.copy()
    X.columns = new
    return X, dict(zip(new, orig))
