"""Classification and business-style ranking metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(y_true)))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(y_true[top].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(y_true)))
    pos = int(y_true.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    top = order[:k]
    return float(y_true[top].sum() / pos)


def lift_top_decile(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Ratio: precision in top decile / baseline positive rate."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, n // 10)
    order = np.argsort(-y_score)
    top = order[:k]
    base = float(y_true.mean()) if y_true.mean() > 0 else np.nan
    prec = float(y_true[top].mean())
    return prec / base if base and base > 0 else float("nan")


def evaluate_and_save(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics_dir: Path,
    plots_dir: Path,
    feature_importance: pd.DataFrame | None = None,
) -> dict:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_hat = (y_score >= 0.5).astype(int)
    out: dict = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision_0p5": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall_0p5": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1_0p5": float(f1_score(y_true, y_hat, zero_division=0)),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }

    k10 = int(max(1, round(0.10 * len(y_true))))
    k5 = int(max(1, round(0.05 * len(y_true))))
    out["precision_at_10pct"] = precision_at_k(y_true, y_score, k10)
    out["recall_at_10pct"] = recall_at_k(y_true, y_score, k10)
    out["precision_at_5pct"] = precision_at_k(y_true, y_score, k5)
    out["recall_at_5pct"] = recall_at_k(y_true, y_score, k5)
    out["lift_top_decile"] = lift_top_decile(y_true, y_score)

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(y_score, bins, right=True)
    cal = (
        pd.DataFrame({"bin": idx, "y": y_true})
        .groupby("bin", as_index=False)["y"]
        .mean()
        .rename(columns={"y": "empirical_rate"})
    )
    out["calibration_preview"] = cal.head(5).to_dict(orient="records")

    with open(config.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if feature_importance is not None and not feature_importance.empty:
        top = feature_importance.head(25)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top, x="gain", y="feature", color="#4C72B0")
        plt.title("LightGBM feature importance (gain)")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance_top.png", dpi=150)
        plt.close()

        feature_importance.to_csv(metrics_dir / "feature_importance.csv", index=False)

    pred_df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    pred_df.to_csv(metrics_dir / "test_predictions.csv", index=False)

    return out
