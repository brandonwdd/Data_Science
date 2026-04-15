"""Run the Lending Club EL pipeline"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import joblib
import numpy as np
import pandas as pd

import config
from src.business import business_threshold_simulation, portfolio_expected_loss
from src.data_loader import load_training_frame
from src.evaluation import (
    compute_metrics_bundle,
    compute_lgd_metrics,
    plot_feature_importance,
    plot_ks,
    plot_roc,
    plot_shap_summary,
    save_metrics,
    top_decile_loss_capture,
)
from src.feature_engineering import materialize_train_val_xy, ordered_time_split, rename_sanitized
from src.leakage_filters import leakage_columns_present
from src.modeling import predict_lgd, predict_proba_positive, train_lgbm, train_lgbm_lgd
from src.preprocessing import build_lgd_target


def ensure_dirs() -> None:
    for p in (
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    X_other = X_other.copy()
    for c in X_ref.columns:
        if c not in X_other.columns:
            X_other[c] = 0.0
    return X_other[X_ref.columns]


def main() -> None:
    ensure_dirs()

    df = load_training_frame(config.ROW_CAP)
    train_raw, val_raw = ordered_time_split(df, config.ISSUE_DATE_COL, config.VAL_FRACTION)

    X_tr, X_va, y_tr, y_va = materialize_train_val_xy(train_raw, val_raw)

    X_tr, _ = rename_sanitized(X_tr)
    X_va, _ = rename_sanitized(X_va)
    X_va = align_columns(X_tr, X_va)

    # -----------------------------
    # PD model
    # -----------------------------
    pd_model = train_lgbm(X_tr, y_tr, X_va, y_va)
    joblib.dump(pd_model, config.PD_MODEL_PATH)

    pd_val = predict_proba_positive(pd_model, X_va)
    exposure = val_raw[config.EXPOSURE_COL].to_numpy(dtype=float)

    pd_metrics = compute_metrics_bundle(y_va.to_numpy(), pd_val)
    pd_metrics["n_train"] = int(len(X_tr))
    pd_metrics["n_val"] = int(len(X_va))
    pd_metrics["best_iteration"] = int(getattr(pd_model, "best_iteration_", 0) or 0)

    # -----------------------------
    # LGD model (defaults only)
    # -----------------------------
    lgd_train = build_lgd_target(train_raw)
    lgd_val_true = build_lgd_target(val_raw)
    train_def_mask = y_tr.to_numpy() == 1
    val_def_mask = y_va.to_numpy() == 1

    lgd_metrics = {"rmse": None, "mae": None, "n": 0}
    if train_def_mask.sum() >= 200 and val_def_mask.sum() >= 50:
        lgd_model = train_lgbm_lgd(
            X_tr.loc[train_def_mask],
            pd.Series(lgd_train.loc[train_raw.index[train_def_mask]].to_numpy(dtype=float)),
            X_va.loc[val_def_mask],
            pd.Series(lgd_val_true.loc[val_raw.index[val_def_mask]].to_numpy(dtype=float)),
        )
        joblib.dump(lgd_model, config.LGD_MODEL_PATH)
        lgd_val_pred = predict_lgd(lgd_model, X_va)
        lgd_metrics = compute_lgd_metrics(
            lgd_val_true.loc[val_raw.index[val_def_mask]].to_numpy(dtype=float),
            lgd_val_pred[val_def_mask],
        )
    else:
        train_lgd_vals = lgd_train.loc[train_raw.index[train_def_mask]].to_numpy(dtype=float)
        const = float(np.nanmean(train_lgd_vals)) if np.isfinite(train_lgd_vals).any() else 0.6
        lgd_val_pred = np.clip(np.full_like(pd_val, const, dtype=float), 0.0, 1.0)
        lgd_metrics = compute_lgd_metrics(
            lgd_val_true.loc[val_raw.index[val_def_mask]].to_numpy(dtype=float),
            lgd_val_pred[val_def_mask],
        )
    with config.resolve_accepted_csv().open("r", encoding="utf-8", errors="replace", newline="") as fp:
        header_cols = next(csv.reader(fp))
    metrics: dict = {
        "pd_model": pd_metrics,
        "lgd_model": lgd_metrics,
        "leakage_columns_excluded": leakage_columns_present(header_cols),
        "validation": {
            "portfolio_el_proxy_pd_only": portfolio_expected_loss(pd_val, exposure),
            "portfolio_el_proxy_pd_lgd": portfolio_expected_loss(pd_val * lgd_val_pred, exposure),
        },
    }

    actual_el = np.where(val_def_mask, lgd_val_true.to_numpy(dtype=float) * exposure, 0.0)
    pred_el = pd_val * lgd_val_pred * exposure
    metrics["collections_top_decile_loss_capture_by_el"] = top_decile_loss_capture(actual_el, pred_el)
    save_metrics(metrics, config.METRICS_JSON)

    biz = business_threshold_simulation(y_va.to_numpy(), pd_val, exposure, lgd_pred=lgd_val_pred)
    save_metrics({"business_simulation": biz}, config.BUSINESS_SIM_JSON)

    pred = pd.DataFrame(
        {
            "target": y_va.values,
            "pred_pd": pd_val,
            "pred_lgd": lgd_val_pred,
            config.EXPOSURE_COL: exposure,
            "pred_el": pred_el,
            "true_lgd": lgd_val_true.to_numpy(dtype=float),
            "true_el": actual_el,
        }
    )
    if "id" in val_raw.columns:
        pred.insert(0, "id", val_raw["id"].values)
    pred.to_csv(config.PREDICTIONS_CSV, index=False)

    plot_roc(y_va.to_numpy(), pd_val, config.PLOTS_DIR / "roc_curve.png")
    plot_ks(y_va.to_numpy(), pd_val, config.PLOTS_DIR / "ks_curve.png")
    plot_feature_importance(pd_model, list(X_tr.columns), config.PLOTS_DIR / "feature_importance_top.png")

    if config.SHAP_MAX_SAMPLES and config.SHAP_MAX_SAMPLES > 0:
        try:
            plot_shap_summary(
                pd_model,
                X_va,
                list(X_tr.columns),
                config.PLOTS_DIR / "shap_summary_bar.png",
                max_samples=config.SHAP_MAX_SAMPLES,
            )
        except Exception as exc:
            print(f"SHAP plot skipped: {exc}")

    print("Done.")
    print(f"  PD model: {config.PD_MODEL_PATH}")
    if config.LGD_MODEL_PATH.is_file():
        print(f"  LGD model: {config.LGD_MODEL_PATH}")
    print(f"  Val ROC-AUC: {pd_metrics['roc_auc']:.4f}  KS: {pd_metrics['ks_statistic']:.4f}")
    print(f"  Predictions: {config.PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
