"""Simulate revenue over a price grid and pick per-product optimum."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import predict_quantities


def volume_weighted_current_price(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Per product_id: volume-weighted mean of avg_price over last `lookback_days`
    of the training window (dates in (train_end - lookback, train_end]).
    """
    start = train_end - pd.Timedelta(days=lookback_days)
    w = panel[
        (panel["transaction_date"] > start) & (panel["transaction_date"] <= train_end)
    ].copy()
    if w.empty:
        w = panel[panel["transaction_date"] <= train_end].copy()
    w["_pxq"] = w["avg_price"] * w["total_quantity"]
    g = w.groupby("product_id", as_index=False).agg(
        num=("_pxq", "sum"),
        den=("total_quantity", "sum"),
    )
    g["current_price"] = g["num"] / g["den"].replace(0, np.nan)
    g = g.drop(columns=["num", "den"])
    g = g[g["current_price"].notna() & (g["current_price"] > 0)].reset_index(drop=True)
    return g


def _last_train_snapshot_per_store(
    panel: pd.DataFrame, train_end: pd.Timestamp
) -> pd.DataFrame:
    """One row per (product_id, store_location): last train-day row <= train_end."""
    tr = panel[panel["transaction_date"] <= train_end].copy()
    tr = tr.sort_values(["product_id", "store_location", "transaction_date"])
    last = tr.groupby(["product_id", "store_location"], as_index=False).tail(1)
    return last.reset_index(drop=True)


def simulate_revenue_grid(
    model,
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    current_prices: pd.DataFrame,
    price_low: float,
    price_high: float,
    n_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each product, sum predicted chain-wide daily quantity across stores
    for each candidate price (same price at all stores in simulation).

    Returns:
      summary: one row per product with optimal price and revenues
      detail_long: optional long form for plotting (product_id, price, q, revenue)
    """
    snap = _last_train_snapshot_per_store(panel, train_end)
    if snap.empty:
        raise ValueError("No training rows available for optimization snapshot.")

    prices_grid = {}
    for _, r in current_prices.iterrows():
        pid = int(r["product_id"])
        cp = float(r["current_price"])
        if not np.isfinite(cp) or cp <= 0:
            continue
        lo = max(cp * price_low, 1e-6)
        hi = max(cp * price_high, lo + 1e-6)
        prices_grid[pid] = np.linspace(lo, hi, n_points)

    detail_rows = []
    summary_rows = []

    for pid, grid in prices_grid.items():
        sub = snap[snap["product_id"] == pid]
        if sub.empty:
            continue
        product_name = sub["product_detail"].iloc[0]
        best_rev = -1.0
        best_p = grid[0]
        cur_p = float(current_prices.loc[current_prices["product_id"] == pid, "current_price"].iloc[0])
        q_at_cur = 0.0
        q_at_opt = 0.0

        for p in grid:
            block = sub.copy()
            p = float(p)
            block["avg_price"] = p
            block["log_avg_price"] = np.log(max(p, 1e-6))
            q_sum = float(predict_quantities(model, block).sum())
            rev = float(p) * q_sum
            detail_rows.append(
                {
                    "product_id": pid,
                    "price": float(p),
                    "predicted_quantity_total": q_sum,
                    "revenue": rev,
                }
            )
            if rev > best_rev:
                best_rev = rev
                best_p = float(p)

        block_cur = sub.copy()
        block_cur["avg_price"] = cur_p
        block_cur["log_avg_price"] = np.log(max(cur_p, 1e-6))
        q_at_cur = float(predict_quantities(model, block_cur).sum())

        block_opt = sub.copy()
        bp = float(best_p)
        block_opt["avg_price"] = bp
        block_opt["log_avg_price"] = np.log(max(bp, 1e-6))
        q_at_opt = float(predict_quantities(model, block_opt).sum())

        cur_rev = float(cur_p * q_at_cur)
        opt_rev = float(best_p * q_at_opt)
        uplift = (opt_rev - cur_rev) / cur_rev if cur_rev > 0 else np.nan

        summary_rows.append(
            {
                "product_id": pid,
                "product_name": product_name,
                "current_price": cur_p,
                "optimal_price": float(best_p),
                "predicted_quantity_at_current_price": q_at_cur,
                "predicted_quantity_at_optimal_price": q_at_opt,
                "current_revenue": cur_rev,
                "optimal_revenue": opt_rev,
                "revenue_uplift_pct": 100.0 * uplift if np.isfinite(uplift) else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    return summary, detail


def merge_elasticity(summary: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(elasticity, on="product_id", how="left")
    return out
