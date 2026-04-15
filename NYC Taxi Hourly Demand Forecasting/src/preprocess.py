"""Clean trip records before aggregation."""

import pandas as pd


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["pickup_datetime"]).copy()
    return out.reset_index(drop=True)
