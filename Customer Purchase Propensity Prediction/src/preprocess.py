"""Clean transactions and derive line-level monetary amount."""

from __future__ import annotations

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid rows, parse dates, compute total_amount.

    Rules (project spec):
    - drop missing customer_id
    - quantity > 0
    - price > 0
    - parse invoice_date (UK-style day first in Online Retail II)
    - drop duplicate rows
    """
    out = df.copy()

    out = out.dropna(subset=["customer_id"])
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["customer_id"])
    out["customer_id"] = out["customer_id"].astype(int)

    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["quantity", "price"])
    out = out[(out["quantity"] > 0) & (out["price"] > 0)]

    out["invoice_date"] = pd.to_datetime(out["invoice_date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["invoice_date"])

    out["total_amount"] = out["quantity"] * out["price"]

    out["country"] = out["country"].fillna("Unknown").astype(str)

    out = out.drop_duplicates()

    out = out.sort_values(["customer_id", "invoice_date", "invoice", "stock_code"]).reset_index(
        drop=True
    )
    return out
