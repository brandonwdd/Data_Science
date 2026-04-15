from __future__ import annotations

import pandas as pd

import config


def load_train_transaction(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_TRANSACTION, nrows=nrows, low_memory=False)


def load_test_transaction(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(config.TEST_TRANSACTION, nrows=nrows, low_memory=False)


def load_train_identity(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_IDENTITY, nrows=nrows, low_memory=False)


def load_test_identity(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(config.TEST_IDENTITY, nrows=nrows, low_memory=False)

