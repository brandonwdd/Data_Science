from __future__ import annotations

import config

FORBIDDEN_COLUMNS = {
    # Never use explicit identifiers as model features.
    config.ID_COL,
}


def drop_forbidden_columns(df):
    """
    This dataset is transaction-time oriented; main guardrail is not to include label/id.
    Keep TransactionDT (time) and TransactionAmt (exposure proxy) as features.
    """
    drop = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    return df.drop(columns=drop, errors="ignore")

