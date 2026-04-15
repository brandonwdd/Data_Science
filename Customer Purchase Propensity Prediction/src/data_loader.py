"""Load raw Online Retail CSV and standardize column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_raw_transactions(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    missing = [c for c in config.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.rename(columns=config.COLUMN_RENAME_MAP)
    return df
