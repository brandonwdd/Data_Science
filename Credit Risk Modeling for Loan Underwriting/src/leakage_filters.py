"""Row-level filters so auxiliary history is restricted to pre-application timelines."""

from __future__ import annotations

import pandas as pd


def filter_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Keep prior applications decided strictly before the current application
    (Home Credit: DAYS_DECISION < 0 relative to current app).
    """
    if "DAYS_DECISION" not in prev.columns:
        return prev
    mask = prev["DAYS_DECISION"].notna() & (prev["DAYS_DECISION"] < 0)
    return prev.loc[mask].copy()


def filter_installments_payments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Keep installment events logged before the reference date (DAYS_ENTRY_PAYMENT < 0).
    """
    if "DAYS_ENTRY_PAYMENT" not in inst.columns:
        return inst
    mask = inst["DAYS_ENTRY_PAYMENT"].notna() & (inst["DAYS_ENTRY_PAYMENT"] < 0)
    return inst.loc[mask].copy()


def filter_bureau_for_leakage(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: bureau credits opened before / relative to application (DAYS_CREDIT < 0).
    Not always used in all baselines; aligns with 'past-only' bureau lines.
    """
    if "DAYS_CREDIT" not in bureau.columns:
        return bureau
    mask = bureau["DAYS_CREDIT"].notna() & (bureau["DAYS_CREDIT"] < 0)
    return bureau.loc[mask].copy()
