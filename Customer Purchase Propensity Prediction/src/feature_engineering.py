"""Customer-level RFM and behavioral features for a fixed observation cutoff."""

from __future__ import annotations

import pandas as pd


def _obs_window_mask(
    transactions: pd.DataFrame, cutoff_date: pd.Timestamp, observation_days: int
) -> pd.Series:
    cutoff_norm = pd.Timestamp(cutoff_date).normalize()
    obs_start = cutoff_norm - pd.Timedelta(days=observation_days - 1)
    d = transactions["invoice_date"].dt.normalize()
    return (d >= obs_start) & (d <= cutoff_norm)


def build_customer_features(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    observation_days: int,
) -> pd.DataFrame:
    """
    One row per customer active in the observation window ending at cutoff_date (inclusive).

    Observation window: last `observation_days` calendar days ending on cutoff_date.
    """
    cutoff_norm = pd.Timestamp(cutoff_date).normalize()
    m = _obs_window_mask(transactions, cutoff_date, observation_days)
    tx = transactions.loc[m].copy()
    if tx.empty:
        return pd.DataFrame()

    tx["inv_date_norm"] = tx["invoice_date"].dt.normalize()

    inv_value = tx.groupby(["customer_id", "invoice"], as_index=False)["total_amount"].sum()
    inv_value = inv_value.rename(columns={"total_amount": "order_value"})

    last_purchase = tx.groupby("customer_id")["inv_date_norm"].max().rename("last_purchase_date")
    first_purchase = tx.groupby("customer_id")["inv_date_norm"].min().rename("first_purchase_date")

    recency_days = ((cutoff_norm - last_purchase).dt.days).rename("recency_days")

    frequency = tx.groupby("customer_id")["invoice"].nunique().rename("frequency")
    monetary = tx.groupby("customer_id")["total_amount"].sum().rename("monetary")
    total_items = tx.groupby("customer_id")["quantity"].sum().rename("total_items")
    unique_products = tx.groupby("customer_id")["stock_code"].nunique().rename("unique_products")
    active_days = tx.groupby("customer_id")["inv_date_norm"].nunique().rename("active_days")

    # Country: mode by spend
    country_pick = (
        tx.groupby(["customer_id", "country"], as_index=False)["total_amount"]
        .sum()
        .sort_values(["customer_id", "total_amount"], ascending=[True, False])
        .drop_duplicates("customer_id")[["customer_id", "country"]]
        .set_index("customer_id")["country"]
        .rename("country")
    )

    avg_order_value = (monetary / frequency).rename("avg_order_value")

    od = tx.groupby(["customer_id", "invoice"], as_index=False)["inv_date_norm"].min()
    od = od.sort_values(["customer_id", "inv_date_norm"])
    od["prev_inv_date"] = od.groupby("customer_id")["inv_date_norm"].shift(1)
    od["gap_days"] = (od["inv_date_norm"] - od["prev_inv_date"]).dt.days
    avg_days_between_orders = od.groupby("customer_id")["gap_days"].mean().rename(
        "avg_days_between_orders"
    )

    ov = inv_value.groupby("customer_id")["order_value"]
    max_order_value = ov.max().rename("max_order_value")
    min_order_value = ov.min().rename("min_order_value")
    std_order_value = ov.std().rename("std_order_value")

    days_since_first = (cutoff_norm - first_purchase).dt.days.rename("days_since_first_purchase")
    customer_lifespan_days = (last_purchase - first_purchase).dt.days.rename(
        "customer_lifespan_days"
    )

    feats = pd.concat(
        [
            recency_days,
            frequency,
            monetary,
            avg_order_value,
            total_items,
            (total_items / frequency).rename("avg_items_per_order"),
            unique_products,
            active_days,
            country_pick,
            days_since_first,
            customer_lifespan_days,
            avg_days_between_orders,
            max_order_value,
            min_order_value,
            std_order_value,
            last_purchase,
        ],
        axis=1,
    ).reset_index()

    feats["avg_days_between_orders"] = feats["avg_days_between_orders"].fillna(0.0)
    feats["std_order_value"] = feats["std_order_value"].fillna(0.0)

    return feats
