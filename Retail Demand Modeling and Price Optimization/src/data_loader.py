"""Load raw transaction data."""
from pathlib import Path

import pandas as pd


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    expected = {
        "transaction_id",
        "transaction_date",
        "transaction_time",
        "transaction_qty",
        "store_id",
        "store_location",
        "product_id",
        "unit_price",
        "product_category",
        "product_type",
        "product_detail",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {sorted(missing)}")
    return df
