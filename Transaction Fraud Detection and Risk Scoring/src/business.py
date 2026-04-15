from __future__ import annotations

import numpy as np


def fraud_loss_proxy(pred_fraud_prob: np.ndarray, amount: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_fraud_prob, dtype=float)
    a = np.asarray(amount, dtype=float)
    a = np.where(np.isfinite(a) & (a >= 0), a, 0.0)
    return p * a


def threshold_simulation_three_way(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    allow_threshold: float,
    block_threshold: float,
) -> dict:
    """
    Allow: score < allow_threshold
    Review: allow_threshold <= score < block_threshold
    Block: score >= block_threshold

    Reports rates and fraud capture; uses amount-weighted loss proxy.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    amt = np.asarray(amount, dtype=float)
    amt = np.where(np.isfinite(amt) & (amt >= 0), amt, 0.0)

    allow = s < allow_threshold
    block = s >= block_threshold
    review = (~allow) & (~block)

    def _rate(mask):
        return float(mask.mean())

    def _fraud_rate(mask):
        return float(y[mask].mean()) if mask.sum() else None

    total_frauds = int(y.sum())
    captured = int(y[block | review].sum())
    capture_rate = float(captured / total_frauds) if total_frauds else None

    loss = fraud_loss_proxy(s, amt)
    remaining_loss_allow = float(loss[allow].sum())
    flagged_loss = float(loss[block | review].sum())

    return {
        "allow_threshold": float(allow_threshold),
        "block_threshold": float(block_threshold),
        "allow_rate": _rate(allow),
        "review_rate": _rate(review),
        "block_rate": _rate(block),
        "fraud_rate_allow": _fraud_rate(allow),
        "fraud_rate_review": _fraud_rate(review),
        "fraud_rate_block": _fraud_rate(block),
        "fraud_capture_rate_flagged": capture_rate,
        "fraud_loss_proxy_flagged": flagged_loss,
        "fraud_loss_proxy_remaining_allow": remaining_loss_allow,
    }


def threshold_grid_simulation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    allow_thresholds: list[float] | None = None,
    block_thresholds: list[float] | None = None,
) -> list[dict]:
    if allow_thresholds is None:
        allow_thresholds = [0.02, 0.05, 0.10]
    if block_thresholds is None:
        block_thresholds = [0.50, 0.70, 0.90]
    out: list[dict] = []
    for a in allow_thresholds:
        for b in block_thresholds:
            if b <= a:
                continue
            out.append(threshold_simulation_three_way(y_true, y_score, amount, a, b))
    return out


def review_budget_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    budgets: list[float],
) -> list[dict]:
    """
    Fixed review-capacity analysis:
    review top X% by score and report precision/recall/fraud-loss-proxy captured.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    loss = fraud_loss_proxy(y_score, amount)
    total_frauds = max(int(y.sum()), 1)
    total_loss = float(loss.sum()) if len(loss) else 0.0
    order = np.argsort(-s)
    out: list[dict] = []
    n = len(y)
    for b in budgets:
        k = max(1, int(n * b))
        idx = order[:k]
        tp = int(y[idx].sum())
        prec = float(tp / k)
        rec = float(tp / total_frauds)
        captured_loss = float(loss[idx].sum())
        loss_capture = float(captured_loss / total_loss) if total_loss > 0 else None
        out.append(
            {
                "budget_fraction": float(b),
                "flagged_n": int(k),
                "precision": prec,
                "recall_fraud_capture": rec,
                "fraud_loss_proxy_captured": captured_loss,
                "fraud_loss_proxy_capture_rate": loss_capture,
            }
        )
    return out

