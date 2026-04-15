"""Construct binary purchase-in-future-window labels per customer."""

from __future__ import annotations

import pandas as pd


def _cutoff_norm(cutoff_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(cutoff_date).normalize()


def customers_with_future_purchase(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    gap_days: int,
    prediction_days: int,
) -> pd.Series:
    """
    Return boolean Series indexed by customer_id: True if at least one purchase
    in (cutoff_norm + gap, cutoff_norm + gap + prediction_days] on invoice dates
    normalized to calendar days.
    """
    cnorm = _cutoff_norm(cutoff_date)
    d = transactions["invoice_date"].dt.normalize()
    start = cnorm + pd.Timedelta(days=gap_days)
    end = cnorm + pd.Timedelta(days=gap_days + prediction_days)
    mask = (d > start) & (d <= end)
    sub = transactions.loc[mask, ["customer_id"]].drop_duplicates()
    pos = pd.Series(True, index=sub["customer_id"].unique())
    return pos


def build_label_table(
    transactions: pd.DataFrame,
    customer_ids: pd.Series,
    cutoff_date: pd.Timestamp,
    gap_days: int,
    prediction_days: int,
) -> pd.DataFrame:
    """Per customer in customer_ids, target 1 if future purchase else 0."""
    pos = customers_with_future_purchase(
        transactions, cutoff_date, gap_days, prediction_days
    )
    base = pd.DataFrame({"customer_id": customer_ids.unique()})
    base["target"] = base["customer_id"].isin(pos.index).astype(int)
    return base
