"""
Columns that reflect payments, outcomes, or post-origination updates — excluded from features.
Origination-time credit attributes (FICO, tradelines, etc.) stay in.
"""

from __future__ import annotations

# Explicit post-outcome / payment fields (README + Lending Club practice).
LEAKAGE_COLUMN_PREFIXES: tuple[str, ...] = (
    "hardship_",
    "debt_settlement",
    "settlement_",
)

LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "hardship_flag",
        # IDs / free text (handled separately; listed for clarity)
        "id",
        "member_id",
        "url",
        "desc",
        "emp_title",
        "title",
        # Target derivation
        "loan_status",
        # Dates used only for splitting / audit
        "issue_d",
        # Payment & balance outcomes
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        # Post-modification program fields
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "deferral_term",
        "payment_plan_start_date",
    }
)


def is_leakage_column(name: str) -> bool:
    if name in LEAKAGE_COLUMNS:
        return True
    lower = name.lower()
    return any(lower.startswith(p) for p in LEAKAGE_COLUMN_PREFIXES)


def leakage_columns_present(columns: list[str]) -> list[str]:
    return sorted({c for c in columns if is_leakage_column(c)})
