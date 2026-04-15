from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import ks_statistic


def decile_lift_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
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
    prec = float(tp / n) if n else 0.0
    rec = float(tp / y_true.sum()) if y_true.sum() else 0.0
    return {"fraction": frac, "precision": prec, "recall": rec, "n": n}


def top_decile_default_capture(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Share of all defaults captured in the riskiest decile (collections prioritization)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return {"decile": 1, "default_capture_rate": None, "n_defaults": int(y_true.sum())}
    order = np.argsort(-y_score)
    k = max(1, n // 10)
    top_idx = order[:k]
    captured = y_true[top_idx].sum()
    rate = float(captured / y_true.sum())
    return {"decile": 1, "top_n": int(k), "default_capture_rate": rate, "n_defaults": int(y_true.sum())}


def compute_metrics_bundle(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    roc_auc = float(roc_auc_score(y_true, y_score))
    ks = float(ks_statistic(y_true, y_score))
    top5 = precision_recall_at_top_fraction(y_true, y_score, 0.05)
    top10 = precision_recall_at_top_fraction(y_true, y_score, 0.10)
    deciles = decile_lift_table(y_true, y_score)
    capture = top_decile_default_capture(y_true, y_score)
    return {
        "roc_auc": roc_auc,
        "ks_statistic": ks,
        "precision_recall_top_5pct": top5,
        "precision_recall_top_10pct": top10,
        "decile_lift": deciles.to_dict(orient="records"),
        "collections_top_decile_default_capture": capture,
    }


def compute_lgd_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": None, "mae": None, "n": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return {"rmse": rmse, "mae": mae, "n": int(mask.sum())}


def top_decile_loss_capture(actual_loss: np.ndarray, score: np.ndarray) -> dict:
    """
    Share of total actual loss captured in top (riskiest) decile by `score`.
    `score` should be predicted EL or PD for ranking.
    """
    a = np.asarray(actual_loss, dtype=float)
    s = np.asarray(score, dtype=float)
    a = np.where(np.isfinite(a) & (a >= 0), a, 0.0)
    s = np.where(np.isfinite(s), s, -np.inf)
    total = float(np.sum(a))
    n = len(a)
    if n == 0 or total <= 0:
        return {"top_n": 0, "loss_capture_rate": None, "total_actual_loss": total}
    k = max(1, n // 10)
    top_idx = np.argsort(-s)[:k]
    captured = float(np.sum(a[top_idx]))
    return {"top_n": int(k), "loss_capture_rate": float(captured / total), "total_actual_loss": total}


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(metrics), f, indent=2, ensure_ascii=False)


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
    plt.plot(scores, cdf0, label="CDF score | target=0")
    plt.plot(scores, cdf1, label="CDF score | target=1")
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


def plot_shap_summary(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    out_path: Path,
    max_samples: int = 1500,
) -> None:
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
