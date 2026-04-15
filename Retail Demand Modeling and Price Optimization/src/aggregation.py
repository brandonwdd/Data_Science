"""Aggregate transactions to daily product-store panels."""
import pandas as pd


def aggregate_daily_product_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (date, product_id, store_location).
    avg_price is quantity-weighted average unit price for that day.
    """
    gcols = ["transaction_date", "product_id", "store_location"]
    agg = df.groupby(gcols, as_index=False).agg(
        total_quantity=("transaction_qty", "sum"),
        revenue=("total_amount", "sum"),
        product_category=("product_category", "first"),
        product_type=("product_type", "first"),
        product_detail=("product_detail", "first"),
        store_id=("store_id", "first"),
    )
    agg["avg_price"] = agg["revenue"] / agg["total_quantity"]
    agg = agg.drop(columns=["revenue"])
    agg = agg.sort_values(["product_id", "store_location", "transaction_date"]).reset_index(
        drop=True
    )
    return agg
