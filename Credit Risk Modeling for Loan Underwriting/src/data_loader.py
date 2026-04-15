from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_application_train(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.APPLICATION_TRAIN, nrows=nrows)


def load_application_test(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.APPLICATION_TEST, nrows=nrows)


def load_bureau(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.BUREAU, nrows=nrows)


def load_bureau_balance(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.BUREAU_BALANCE, nrows=nrows)


def load_previous_application(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.PREVIOUS_APPLICATION, nrows=nrows)


def load_pos_cash_balance(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.POS_CASH_BALANCE, nrows=nrows)


def load_credit_card_balance(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.CREDIT_CARD_BALANCE, nrows=nrows)


def load_installments_payments(nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.INSTALLMENTS_PAYMENTS, nrows=nrows)
