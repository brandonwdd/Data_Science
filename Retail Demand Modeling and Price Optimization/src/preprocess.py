"""Clean and enrich transaction-level rows."""
import pandas as pd


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["transaction_date"] = pd.to_datetime(out["transaction_date"]).dt.normalize()
    out = out[out["transaction_qty"] > 0]
    out = out[out["unit_price"] > 0]
    out["total_amount"] = out["transaction_qty"] * out["unit_price"]
    return out.reset_index(drop=True)
