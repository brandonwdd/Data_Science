from __future__ import annotations

import numpy as np
import pandas as pd

import config


def clean_term_months(term: pd.Series) -> pd.Series:
    """' 36 months' -> 36."""
    return (
        term.astype(str)
        .str.replace("months", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def parse_issue_d(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def build_target(loan_status: pd.Series) -> pd.Series:
    """1 = Charged Off or Default; 0 = Fully Paid."""
    pos = loan_status.isin(["Charged Off", "Default"])
    neg = loan_status == "Fully Paid"
    y = pd.Series(np.nan, index=loan_status.index)
    y.loc[pos] = 1
    y.loc[neg] = 0
    return y


def filter_labeled_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with a clear terminal label for training."""
    y = build_target(df["loan_status"])
    out = df.loc[y.notna()].copy()
    out[config.TARGET_COL] = y.loc[out.index].astype(np.int32)
    return out


def build_lgd_target(df: pd.DataFrame) -> pd.Series:
    """
    LGD defined for defaulted loans only. Uses available columns:
    - Preferred: (loan_amnt - total_rec_prncp) / loan_amnt
    - Fallback: 1 - (recoveries / loan_amnt)
    Clipped to [0, 1]. Returns NaN when undefined.
    """
    if config.EXPOSURE_COL not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ead = pd.to_numeric(df[config.EXPOSURE_COL], errors="coerce")
    ead = ead.where(ead > 0)

    lgd = pd.Series(np.nan, index=df.index, dtype="float64")
    if "total_rec_prncp" in df.columns:
        trp = pd.to_numeric(df["total_rec_prncp"], errors="coerce")
        lgd1 = (ead - trp) / ead
        lgd = lgd1
    if lgd.isna().all() and "recoveries" in df.columns:
        rec = pd.to_numeric(df["recoveries"], errors="coerce")
        lgd2 = 1.0 - (rec / ead)
        lgd = lgd2

    lgd = lgd.clip(lower=0.0, upper=1.0)
    return lgd


def load_accepted(path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df[config.ISSUE_DATE_COL] = parse_issue_d(df[config.ISSUE_DATE_COL])
    if "term" in df.columns:
        df["term"] = clean_term_months(df["term"])
    df = filter_labeled_loans(df)
    df = df.dropna(subset=[config.ISSUE_DATE_COL])
    return df
