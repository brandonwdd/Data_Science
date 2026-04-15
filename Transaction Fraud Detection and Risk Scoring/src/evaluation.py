from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.utils import ks_statistic


def decile_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    df["decile"] = np.minimum((np.arange(n) * 10 // n), 9)
    overall = float(df["y"].mean())
    rows = []
    for d in range(10):
        sub = df[df["decile"] == d]
        if len(sub) == 0:
            continue
        rate = float(sub["y"].mean())
        lift = rate / overall if overall > 0 else float("nan")
        rows.append({"decile": d + 1, "n": int(len(sub)), "fraud_rate": rate, "lift": float(lift) if lift == lift else None})
    return pd.DataFrame(rows)


def topk_precision_recall(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = max(1, int(len(y_true) * frac))
    order = np.argsort(-y_score)
    top = order[:n]
    tp = int(y_true[top].sum())
    prec = float(tp / n) if n else 0.0
    rec = float(tp / max(int(y_true.sum()), 1))
    return {"fraction": frac, "n": int(n), "precision": prec, "recall": rec}


def compute_metrics_bundle(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    ks = float(ks_statistic(y_true, y_score))
    deciles = decile_table(y_true, y_score)
    top_fracs = [0.01, 0.03, 0.05, 0.10]
    topk = [topk_precision_recall(y_true, y_score, f) for f in top_fracs]
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks_statistic": ks,
        "deciles": deciles.to_dict(orient="records"),
        "topk": topk,
    }


def validate_prediction_range(y_score: np.ndarray) -> None:
    s = np.asarray(y_score, dtype=float)
    if np.any(~np.isfinite(s)):
        raise ValueError("Predictions contain non-finite values.")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Predictions are outside [0, 1].")


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(metrics), f, indent=2, ensure_ascii=False)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(data), f, indent=2, ensure_ascii=False)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (validation)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"PR (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (validation)")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ks(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    s0 = np.sort(y_score[y_true == 0])
    s1 = np.sort(y_score[y_true == 1])
    scores = np.sort(np.unique(np.concatenate([s0, s1])))
    cdf0 = np.searchsorted(s0, scores, side="right") / max(len(s0), 1)
    cdf1 = np.searchsorted(s1, scores, side="right") / max(len(s1), 1)
    diff = cdf1 - cdf0
    ks = float(np.max(np.abs(diff)))
    plt.figure(figsize=(6, 5))
    plt.plot(scores, cdf0, label="CDF | non-fraud")
    plt.plot(scores, cdf1, label="CDF | fraud")
    plt.plot(scores, diff, label=f"Difference (KS={ks:.4f})", alpha=0.8)
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Cumulative proportion")
    plt.title("KS curve (validation)")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], out_path: Path, top_n: int = 30) -> None:
    imp = model.feature_importances_
    order = np.argsort(-imp)[:top_n]
    plt.figure(figsize=(9, max(4, top_n * 0.22)))
    plt.barh(np.array(feature_names)[order][::-1], imp[order][::-1])
    plt.xlabel("Gain-based importance")
    plt.title(f"Top {top_n} features")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap_summary(model, X_sample: pd.DataFrame, feature_names: list[str], out_path: Path, max_samples: int = 1500) -> None:
    import shap

    n = min(len(X_sample), max_samples)
    if n < 50:
        return
    Xs = X_sample.iloc[:n]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv,
        Xs,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=min(30, Xs.shape[1]),
    )
    plt.title("SHAP mean |impact| (validation sample)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

