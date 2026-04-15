"""Per-product log-log elasticity estimates with confidence flags."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _daily_product_national(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum quantity per (date, product); quantity-weighted avg price."""
    tmp = panel.copy()
    tmp["_pxq"] = tmp["avg_price"] * tmp["total_quantity"]
    g = tmp.groupby(["transaction_date", "product_id"], as_index=False).agg(
        total_quantity=("total_quantity", "sum"),
        revenue=("_pxq", "sum"),
    )
    g["avg_price"] = g["revenue"] / g["total_quantity"]
    g = g.drop(columns=["revenue"])
    return g


def estimate_elasticity_per_product(
    panel: pd.DataFrame,
    min_distinct_prices_high: int,
    min_rows: int,
) -> pd.DataFrame:
    """
    Fit log(total_quantity) ~ log(avg_price) on national daily series per product.
    Elasticity ≈ coefficient of log(price). Rows with zero quantity dropped.
    """
    daily = _daily_product_national(panel)
    rows = []
    for pid, sub in daily.groupby("product_id"):
        sub = sub[sub["total_quantity"] > 0].copy()
        n = len(sub)
        n_prices = sub["avg_price"].round(6).nunique()
        if n < 2:
            rows.append(
                {
                    "product_id": pid,
                    "elasticity": np.nan,
                    "elasticity_confidence_flag": "insufficient_data",
                    "elasticity_n_obs": n,
                    "elasticity_n_distinct_prices": n_prices,
                }
            )
            continue
        y = np.log(sub["total_quantity"].to_numpy(dtype=float))
        x = np.log(sub["avg_price"].to_numpy(dtype=float)).reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(x, y)
        beta = float(reg.coef_[0])
        if n < min_rows:
            flag = "low_confidence"
        elif n_prices < min_distinct_prices_high:
            flag = "low_confidence"
        else:
            flag = "high_confidence"
        rows.append(
            {
                "product_id": pid,
                "elasticity": beta,
                "elasticity_confidence_flag": flag,
                "elasticity_n_obs": n,
                "elasticity_n_distinct_prices": n_prices,
            }
        )
    return pd.DataFrame(rows)
