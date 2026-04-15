"""Metrics, plots, and uplift summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_regression_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def pricing_uplift_summary(recommendations: pd.DataFrame) -> dict:
    s = recommendations["revenue_uplift_pct"].dropna()
    return {
        "n_products": int(len(recommendations)),
        "mean_revenue_uplift_pct": float(s.mean()) if len(s) else float("nan"),
        "median_revenue_uplift_pct": float(s.median()) if len(s) else float("nan"),
        "share_positive_uplift_pct": float((s > 0).mean()) if len(s) else float("nan"),
    }


def save_uplift_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_feature_importance(
    importances: np.ndarray,
    names: list[str],
    path: Path,
    top_k: int = 20,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(importances)[::-1][:top_k]
    x = importances[order]
    y = [names[i] for i in order]
    plt.figure(figsize=(8, max(4, top_k * 0.25)))
    sns.barplot(x=x, y=y, color="#2c5282")
    plt.xlabel("Importance (gain)")
    plt.title("LightGBM feature importance (top {})".format(top_k))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_product_price_curves(
    detail: pd.DataFrame,
    product_ids: list[int],
    out_dir: Path,
) -> None:
    """Revenue and demand vs candidate price for selected products."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in product_ids:
        sub = detail[detail["product_id"] == pid]
        if sub.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(sub["price"], sub["revenue"], color="#2c5282", lw=2, label="Revenue")
        ax1.set_xlabel("Simulated price")
        ax1.set_ylabel("Predicted revenue", color="#2c5282")
        ax1.tick_params(axis="y", labelcolor="#2c5282")

        ax2 = ax1.twinx()
        ax2.plot(
            sub["price"],
            sub["predicted_quantity_total"],
            color="#c05621",
            lw=2,
            linestyle="--",
            label="Predicted qty (chain)",
        )
        ax2.set_ylabel("Predicted total quantity", color="#c05621")
        ax2.tick_params(axis="y", labelcolor="#c05621")

        plt.title(f"Product {pid}: revenue & demand vs price")
        fig.tight_layout()
        fig.savefig(out_dir / f"product_{pid}_price_curves.png", dpi=150)
        plt.close(fig)
