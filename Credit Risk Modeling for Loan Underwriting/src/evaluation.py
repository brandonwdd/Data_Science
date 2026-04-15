from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from src.utils import ks_statistic


def decile_lift_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Ten equal-count bins by descending score; lift = default rate / overall default rate."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    df = pd.DataFrame({"y": y_true, "s": y_score})
    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    decile = np.minimum((np.arange(n) * 10 // n), 9)
    df["decile"] = decile
    overall = float(df["y"].mean())
    rows = []
    for d in range(10):
        sub = df[df["decile"] == d]
        if len(sub) == 0:
            continue
        rate = float(sub["y"].mean())
        lift = rate / overall if overall > 0 else float("nan")
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(sub)),
                "default_rate": rate,
                "lift": float(lift) if lift == lift else None,
            }
        )
    return pd.DataFrame(rows)


def precision_recall_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = max(1, int(len(y_true) * frac))
    order = np.argsort(-y_score)
    top = order[:n]
    tp = y_true[top].sum()
    fp = n - tp
    prec = tp / n if n else 0.0
    rec = tp / y_true.sum() if y_true.sum() else 0.0
    return {"fraction": frac, "precision": float(prec), "recall": float(rec), "n": n}


def business_simulation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
    exposure: np.ndarray | None = None,
) -> list[dict]:
    """
    Approve loans when predicted default probability <= threshold.
    Report approval rate, default rate among approved, and optional expected-loss proxy
    mean(pred * exposure) on approved (exposure e.g. AMT_CREDIT).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    exp = None
    if exposure is not None:
        exp = np.asarray(exposure, dtype=float)
        exp = np.where(np.isfinite(exp), exp, 0.0)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    out = []
    for t in thresholds:
        approved = y_score <= t
        ar = float(approved.mean())
        if approved.sum() == 0:
            dr = None
            el_mean = None
            el_sum = None
        else:
            dr = float(y_true[approved].mean())
            if exp is not None:
                el = y_score[approved] * exp[approved]
                el_mean = float(np.mean(el))
                el_sum = float(np.sum(el))
            else:
                el_mean = None
                el_sum = None
        row = {
            "threshold_prob": float(t),
            "approval_rate": ar,
            "default_rate_among_approved": dr,
            "expected_loss_proxy_mean_among_approved": el_mean,
            "expected_loss_proxy_sum_among_approved": el_sum,
        }
        out.append(row)
    return out


def compute_metrics_bundle(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    roc_auc = float(roc_auc_score(y_true, y_score))
    ks = float(ks_statistic(y_true, y_score))
    top5 = precision_recall_at_top_fraction(y_true, y_score, 0.05)
    top10 = precision_recall_at_top_fraction(y_true, y_score, 0.10)
    deciles = decile_lift_table(y_true, y_score)
    return {
        "roc_auc": roc_auc,
        "ks_statistic": ks,
        "precision_recall_top_5pct": top5,
        "precision_recall_top_10pct": top10,
        "decile_lift": deciles.to_dict(orient="records"),
    }


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(metrics), f, indent=2, ensure_ascii=False)


def plot_shap_summary(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    out_path: Path,
    max_samples: int = 2000,
) -> None:
    """SHAP bar plot of mean |value| (TreeExplainer). Requires `shap` package."""
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
        max_display=min(25, Xs.shape[1]),
    )
    plt.title("SHAP mean |impact| (validation sample)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (validation)")
    plt.legend(loc="lower right")
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
    plt.plot(scores, cdf0, label="CDF score | TARGET=0")
    plt.plot(scores, cdf1, label="CDF score | TARGET=1")
    plt.plot(scores, diff, label=f"Difference (KS={ks:.4f})", alpha=0.8)
    plt.xlabel("Predicted default probability")
    plt.ylabel("Cumulative proportion")
    plt.title("KS curve (validation)")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], out_path: Path, top_n: int = 25) -> None:
    imp = model.feature_importances_
    order = np.argsort(-imp)[:top_n]
    plt.figure(figsize=(8, max(4, top_n * 0.22)))
    plt.barh(np.array(feature_names)[order][::-1], imp[order][::-1])
    plt.xlabel("Gain-based importance")
    plt.title(f"Top {top_n} features")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
