from __future__ import annotations

import numpy as np
import pandas as pd

def _flat_agg_columns(cols) -> list[str]:
    out: list[str] = []
    for col in cols:
        if isinstance(col, tuple):
            a, b = col[0], col[1]
            out.append(str(a) if b in ("", None) else f"{a}_{b}")
        else:
            out.append(str(col))
    return out


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR from external bureau lines."""
    num_cols = [
        c
        for c in bureau.columns
        if c not in ("SK_ID_CURR", "SK_ID_BUREAU") and pd.api.types.is_numeric_dtype(bureau[c])
    ]
    agg_spec: dict = {"SK_ID_BUREAU": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "min", "sum"]
    g = bureau.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    return g.add_prefix("bureau_").rename(columns={"bureau_SK_ID_CURR": "SK_ID_CURR"})


def aggregate_bureau_balance(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Map bureau_balance SK_ID_BUREAU -> SK_ID_CURR, then aggregate."""
    bb = bureau_balance.merge(
        bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU"),
        on="SK_ID_BUREAU",
        how="left",
    )
    if bb["SK_ID_CURR"].isna().any():
        bb = bb.dropna(subset=["SK_ID_CURR"])
    bb["SK_ID_CURR"] = bb["SK_ID_CURR"].astype(np.int64)
    num_cols = [c for c in bb.columns if c not in ("SK_ID_BUREAU", "SK_ID_CURR", "STATUS") and pd.api.types.is_numeric_dtype(bb[c])]
    agg_spec: dict = {"SK_ID_BUREAU": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "min"]
    g = bb.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    out = g.add_prefix("bb_").rename(columns={"bb_SK_ID_CURR": "SK_ID_CURR"})
    return out


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        c
        for c in prev.columns
        if c not in ("SK_ID_PREV", "SK_ID_CURR") and pd.api.types.is_numeric_dtype(prev[c])
    ]
    agg_spec: dict = {"SK_ID_PREV": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "min", "sum"]
    g = prev.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    return g.add_prefix("prev_").rename(columns={"prev_SK_ID_CURR": "SK_ID_CURR"})


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        c
        for c in pos.columns
        if c not in ("SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS") and pd.api.types.is_numeric_dtype(pos[c])
    ]
    agg_spec: dict = {"SK_ID_PREV": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "sum"]
    g = pos.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    return g.add_prefix("pos_").rename(columns={"pos_SK_ID_CURR": "SK_ID_CURR"})


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        c
        for c in cc.columns
        if c not in ("SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS") and pd.api.types.is_numeric_dtype(cc[c])
    ]
    agg_spec: dict = {"SK_ID_PREV": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "sum"]
    g = cc.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    return g.add_prefix("cc_").rename(columns={"cc_SK_ID_CURR": "SK_ID_CURR"})


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    x = inst.copy()
    x["pay_delay"] = x["DAYS_ENTRY_PAYMENT"] - x["DAYS_INSTALMENT"]
    with np.errstate(divide="ignore", invalid="ignore"):
        x["pay_ratio"] = np.where(
            (x["AMT_INSTALMENT"].notna()) & (x["AMT_INSTALMENT"] != 0),
            x["AMT_PAYMENT"] / x["AMT_INSTALMENT"],
            np.nan,
        )
    num_cols = ["pay_delay", "pay_ratio", "AMT_PAYMENT", "AMT_INSTALMENT", "NUM_INSTALMENT_NUMBER"]
    num_cols = [c for c in num_cols if c in x.columns]
    agg_spec: dict = {"SK_ID_PREV": ["count"]}
    for c in num_cols:
        agg_spec[c] = ["mean", "max", "min", "sum"]
    g = x.groupby("SK_ID_CURR", as_index=False).agg(agg_spec)
    g.columns = _flat_agg_columns(g.columns)
    return g.add_prefix("ins_").rename(columns={"ins_SK_ID_CURR": "SK_ID_CURR"})


def build_all_aggregates(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous_application: pd.DataFrame,
    pos_cash: pd.DataFrame,
    credit_card: pd.DataFrame,
    installments: pd.DataFrame,
) -> pd.DataFrame:
    """Single wide table keyed by SK_ID_CURR."""
    b = aggregate_bureau(bureau)
    bb = aggregate_bureau_balance(bureau, bureau_balance)
    p = aggregate_previous_application(previous_application)
    pc = aggregate_pos_cash(pos_cash)
    cc = aggregate_credit_card(credit_card)
    ins = aggregate_installments(installments)
    out = b.merge(bb, on="SK_ID_CURR", how="outer")
    out = out.merge(p, on="SK_ID_CURR", how="outer")
    out = out.merge(pc, on="SK_ID_CURR", how="outer")
    out = out.merge(cc, on="SK_ID_CURR", how="outer")
    out = out.merge(ins, on="SK_ID_CURR", how="outer")
    return out
