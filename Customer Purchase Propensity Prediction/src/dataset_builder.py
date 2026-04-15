"""Merge features + labels and encode countries (Top-N one-hot, fit on train only)."""

from __future__ import annotations

import pandas as pd

import config


def encode_countries_top_n(
    train_df: pd.DataFrame, test_df: pd.DataFrame, country_col: str = "country"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s_tr = train_df[country_col].astype(str)
    top = set(s_tr.value_counts().head(config.TOP_N_COUNTRIES).index)

    def bucket(series: pd.Series) -> pd.Series:
        return series.astype(str).map(lambda x: x if x in top else config.OTHER_COUNTRY_LABEL)

    tr = train_df.copy()
    te = test_df.copy()
    tr[country_col] = bucket(tr[country_col])
    te[country_col] = bucket(te[country_col])

    tr_d = pd.get_dummies(tr[country_col], prefix="country")
    te_d = pd.get_dummies(te[country_col], prefix="country")
    for c in tr_d.columns:
        if c not in te_d.columns:
            te_d[c] = 0
    for c in te_d.columns:
        if c not in tr_d.columns:
            tr_d[c] = 0
    te_d = te_d[tr_d.columns]

    tr_out = pd.concat([tr.drop(columns=[country_col]), tr_d], axis=1)
    te_out = pd.concat([te.drop(columns=[country_col]), te_d], axis=1)
    return tr_out, te_out


def build_modeling_matrices(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_features: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns X_train, y_train, X_test, y_test, last_purchase_train
    (last_purchase_train aligned row-wise with X_train for time-based validation).
    """
    tr = train_features.merge(train_labels, on="customer_id", how="inner", validate="one_to_one")
    te = test_features.merge(test_labels, on="customer_id", how="inner", validate="one_to_one")

    if config.COUNTRY_ENCODING != "top_n_one_hot":
        raise ValueError(f"Unsupported COUNTRY_ENCODING: {config.COUNTRY_ENCODING}")

    tr_enc, te_enc = encode_countries_top_n(tr, te, country_col="country")
    tr_enc = tr_enc.reset_index(drop=True)
    te_enc = te_enc.reset_index(drop=True)

    last_purchase_train = tr_enc["last_purchase_date"].copy()

    drop_cols = {"customer_id", "target", "last_purchase_date"}
    feature_cols = [c for c in tr_enc.columns if c not in drop_cols]

    X_train = tr_enc[feature_cols].copy()
    y_train = tr_enc["target"].astype(int)
    X_test = te_enc[feature_cols].copy()
    y_test = te_enc["target"].astype(int)

    return X_train, y_train, X_test, y_test, last_purchase_train
