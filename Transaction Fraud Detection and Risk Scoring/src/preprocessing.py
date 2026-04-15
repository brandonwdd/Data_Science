from __future__ import annotations

import numpy as np
import pandas as pd

import config


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    """test_identity uses id-01 style; normalize to id_01."""
    out = identity.copy()
    out.columns = [c.replace("-", "_") for c in out.columns]
    return out


def merge_transaction_identity(tx: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left join identity features onto transactions."""
    if identity is None or len(identity) == 0:
        return tx.copy()
    merged = tx.merge(identity, on=config.ID_COL, how="left")
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is seconds from a reference. Create coarse time features."""
    out = df.copy()
    t = pd.to_numeric(out[config.TIME_COL], errors="coerce").fillna(0).astype(np.int64)
    out["time_day"] = (t // 86400).astype(np.int32)
    out["time_week"] = (t // (86400 * 7)).astype(np.int32)
    out["time_hour"] = ((t // 3600) % 24).astype(np.int32)
    out["time_dow"] = ((t // 86400) % 7).astype(np.int32)
    return out


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amt = pd.to_numeric(out[config.AMOUNT_COL], errors="coerce")
    out["amt_log1p"] = np.log1p(amt.clip(lower=0).fillna(0)).astype(np.float32)
    return out


def basic_sanity_checks_train(df: pd.DataFrame) -> None:
    if config.LABEL_COL not in df.columns:
        raise KeyError(f"Missing label column {config.LABEL_COL}")
    if df[config.LABEL_COL].isna().any():
        raise ValueError("Label contains missing values.")
    if not df[config.ID_COL].is_unique:
        raise ValueError("TransactionID is not unique in training transaction table.")


def assert_identity_uniqueness(df: pd.DataFrame) -> None:
    if not df[config.ID_COL].is_unique:
        raise ValueError("TransactionID is not unique in identity table.")


def assert_schema_alignment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    if config.LABEL_COL not in train_cols:
        raise ValueError("Training data missing label column.")
    if config.LABEL_COL in test_cols:
        raise ValueError("Test data should not contain label column.")
    # For features, train should equal test + label.
    diff_train = train_cols - test_cols
    diff_test = test_cols - train_cols
    if diff_train != {config.LABEL_COL} or len(diff_test) != 0:
        raise ValueError(
            f"Train/test schema mismatch: train-test={sorted(diff_train)}, test-train={sorted(diff_test)}"
        )

