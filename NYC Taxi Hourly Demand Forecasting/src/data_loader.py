"""Load trip-level training data."""

from pathlib import Path

import pandas as pd


def load_train_csv(path: Path) -> pd.DataFrame:
    """Load only columns needed for hourly demand (Version 1)."""
    return pd.read_csv(
        path,
        usecols=["pickup_datetime"],
        parse_dates=["pickup_datetime"],
    )
