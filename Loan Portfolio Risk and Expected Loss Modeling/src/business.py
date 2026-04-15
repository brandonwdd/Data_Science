"""Portfolio-level expected loss and threshold simulation."""

from __future__ import annotations

import numpy as np


def portfolio_expected_loss(pred_prob: np.ndarray, exposure: np.ndarray) -> dict:
    """Point-in-time portfolio EL proxy: sum(pred * exposure) / sum(exposure)."""
    p = np.asarray(pred_prob, dtype=float)
    e = np.asarray(exposure, dtype=float)
    e = np.where(np.isfinite(e) & (e >= 0), e, 0.0)
    total_exp = float(np.sum(e))
    if total_exp <= 0:
        return {"portfolio_el_ratio": None, "total_exposure": 0.0, "total_el_proxy": None}
    el = p * e
    return {
        "portfolio_el_ratio": float(np.sum(el) / total_exp),
        "total_exposure": total_exp,
        "total_el_proxy": float(np.sum(el)),
    }


def business_threshold_simulation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray,
    lgd_pred: np.ndarray | None = None,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Approve / hold portfolio when score <= threshold (low predicted risk).
    Report approval rate, default rate among approved, portfolio EL proxy on approved book.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    exp = np.asarray(exposure, dtype=float)
    exp = np.where(np.isfinite(exp), exp, 0.0)
    lgd = None
    if lgd_pred is not None:
        lgd = np.asarray(lgd_pred, dtype=float)
        lgd = np.where(np.isfinite(lgd), lgd, 0.0)
        lgd = np.clip(lgd, 0.0, 1.0)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    rows = []
    for t in thresholds:
        approved = y_score <= t
        n_app = int(approved.sum())
        ar = float(approved.mean())
        dr = float(y_true[approved].mean()) if n_app else None
        if n_app == 0:
            el_port = None
        else:
            if lgd is None:
                el_port = portfolio_expected_loss(y_score[approved], exp[approved])["portfolio_el_ratio"]
            else:
                el_port = portfolio_expected_loss(y_score[approved] * lgd[approved], exp[approved])["portfolio_el_ratio"]
        rows.append(
            {
                "threshold_prob": float(t),
                "approval_rate": ar,
                "default_rate_among_approved": dr,
                "portfolio_el_proxy_on_approved": el_port,
            }
        )
    return rows
