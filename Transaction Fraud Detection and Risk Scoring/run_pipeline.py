"""Run the IEEE-CIS fraud pipeline"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

import config
from src import data_loader
from src.business import fraud_loss_proxy, threshold_grid_simulation
from src.evaluation import (
    compute_metrics_bundle,
    plot_feature_importance,
    plot_ks,
    plot_pr,
    plot_roc,
    plot_shap_summary,
    save_json,
    save_metrics,
    validate_prediction_range,
)
from src.feature_engineering import (
    add_simple_behavioral_features,
    build_xy,
    encode_objects_train_val_test,
    prepare_model_frame,
    split_train_val_by_time,
)
from src.leakage_filters import drop_forbidden_columns
from src.modeling import predict_proba_positive, train_lgbm
from src.preprocessing import (
    add_amount_features,
    add_time_features,
    assert_identity_uniqueness,
    assert_schema_alignment,
    basic_sanity_checks_train,
    merge_transaction_identity,
    normalize_identity_columns,
)

LOGGER = logging.getLogger("fraud_pipeline")


def ensure_dirs() -> None:
    for p in (
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def _align_feature_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    X_other = X_other.copy()
    for c in X_ref.columns:
        if c not in X_other.columns:
            X_other[c] = 0
    return X_other[X_ref.columns]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _save_repro_artifacts(
    *,
    feature_columns: list[str],
    train_fit: pd.DataFrame,
    val_fit: pd.DataFrame,
) -> None:
    if config.SAVE_RUN_CONFIG:
        run_cfg = {
            "random_seed": config.RANDOM_SEED,
            "train_row_cap": config.TRAIN_ROW_CAP,
            "test_row_cap": config.TEST_ROW_CAP,
            "val_fraction": config.VAL_FRACTION,
            "enable_shap": config.ENABLE_SHAP,
            "shap_max_samples": config.SHAP_MAX_SAMPLES,
            "enable_review_budget_analysis": config.ENABLE_REVIEW_BUDGET_ANALYSIS,
            "review_budgets": config.REVIEW_BUDGETS,
            "allow_thresholds": config.ALLOW_THRESHOLDS,
            "block_thresholds": config.BLOCK_THRESHOLDS,
            "lgbm_params": config.LGBM_PARAMS,
        }
        save_json(run_cfg, config.RUN_CONFIG_JSON)
    if config.SAVE_FEATURE_COLUMNS:
        save_json({"feature_columns": feature_columns}, config.FEATURE_COLUMNS_JSON)
    if config.SAVE_DATA_SUMMARY:
        data_summary = {
            "n_train_fit": int(len(train_fit)),
            "n_val": int(len(val_fit)),
            "label_rate_train_fit": float(train_fit[config.LABEL_COL].mean()),
            "label_rate_val": float(val_fit[config.LABEL_COL].mean()),
            "train_time_min": int(train_fit[config.TIME_COL].min()),
            "train_time_max": int(train_fit[config.TIME_COL].max()),
            "val_time_min": int(val_fit[config.TIME_COL].min()),
            "val_time_max": int(val_fit[config.TIME_COL].max()),
            "feature_count": int(len(feature_columns)),
        }
        save_json(data_summary, config.DATA_SUMMARY_JSON)


def main() -> None:
    _setup_logging()
    _set_seed(config.RANDOM_SEED)
    ensure_dirs()

    LOGGER.info("Loading raw tables")
    train_tx = data_loader.load_train_transaction(config.TRAIN_ROW_CAP)
    test_tx = data_loader.load_test_transaction(config.TEST_ROW_CAP)
    train_id = normalize_identity_columns(data_loader.load_train_identity(config.TRAIN_ROW_CAP))
    test_id = normalize_identity_columns(data_loader.load_test_identity(config.TEST_ROW_CAP))

    basic_sanity_checks_train(train_tx)
    assert_identity_uniqueness(train_id)
    assert_identity_uniqueness(test_id)
    assert_schema_alignment(train_tx, test_tx)
    LOGGER.info(
        "Loaded shapes | train_tx=%s train_id=%s test_tx=%s test_id=%s",
        train_tx.shape,
        train_id.shape,
        test_tx.shape,
        test_id.shape,
    )

    train = merge_transaction_identity(train_tx, train_id)
    test = merge_transaction_identity(test_tx, test_id)
    if not train[config.ID_COL].is_unique or not test[config.ID_COL].is_unique:
        raise ValueError("TransactionID not unique after merge.")
    LOGGER.info("Merged shapes | train=%s test=%s", train.shape, test.shape)

    # Enrich with time/amount and missingness summary
    train = add_time_features(add_amount_features(train))
    test = add_time_features(add_amount_features(test))
    train = prepare_model_frame(train)
    test = prepare_model_frame(test)

    # Time split (chronology)
    train_fit, val_fit = split_train_val_by_time(train, config.VAL_FRACTION)
    if int(train_fit[config.TIME_COL].max()) >= int(val_fit[config.TIME_COL].min()):
        raise ValueError("Time split invalid: train max time >= val min time.")
    LOGGER.info(
        "Time split | train_fit=%s val=%s | train_dt=[%s,%s] val_dt=[%s,%s]",
        train_fit.shape,
        val_fit.shape,
        int(train_fit[config.TIME_COL].min()),
        int(train_fit[config.TIME_COL].max()),
        int(val_fit[config.TIME_COL].min()),
        int(val_fit[config.TIME_COL].max()),
    )

    # Leakage / feature drop
    X_tr, y_tr = build_xy(train_fit)
    X_va, y_va = build_xy(val_fit)
    X_te = test.copy()

    # Save IDs for exports
    val_ids = val_fit[[config.ID_COL, config.TIME_COL]].copy()
    test_ids = test[[config.ID_COL]].copy()

    X_tr = drop_forbidden_columns(X_tr)
    X_va = drop_forbidden_columns(X_va)
    X_te = drop_forbidden_columns(X_te)

    # Train-only fitted behavioral features (freq counts etc.)
    X_tr = add_simple_behavioral_features(X_tr, X_tr)
    X_va = add_simple_behavioral_features(X_tr, X_va)
    X_te = add_simple_behavioral_features(X_tr, X_te)

    # Encode object columns consistently
    X_tr, X_va, X_te = encode_objects_train_val_test(X_tr, X_va, X_te)

    # Align columns (in case of rare schema differences)
    X_va = _align_feature_columns(X_tr, X_va)
    X_te = _align_feature_columns(X_tr, X_te)
    LOGGER.info("Feature matrix | n_features=%d", X_tr.shape[1])
    _save_repro_artifacts(feature_columns=list(X_tr.columns), train_fit=train_fit, val_fit=val_fit)

    model = train_lgbm(X_tr, y_tr, X_va, y_va)
    joblib.dump(model, config.MODEL_PATH)
    LOGGER.info("Model trained and saved -> %s", config.MODEL_PATH)

    val_scores = predict_proba_positive(model, X_va)
    validate_prediction_range(val_scores)
    metrics = compute_metrics_bundle(y_va.to_numpy(), val_scores)
    metrics["n_train_fit"] = int(len(X_tr))
    metrics["n_val"] = int(len(X_va))
    metrics["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    save_metrics(metrics, config.METRICS_JSON)
    LOGGER.info(
        "Validation metrics | ROC-AUC=%.4f PR-AUC=%.4f KS=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["ks_statistic"],
    )

    # Business simulation + loss proxy
    val_amount = val_fit[config.AMOUNT_COL].to_numpy(dtype=float)
    sim = threshold_grid_simulation(
        y_va.to_numpy(),
        val_scores,
        val_amount,
        allow_thresholds=config.ALLOW_THRESHOLDS,
        block_thresholds=config.BLOCK_THRESHOLDS,
    )
    save_metrics({"three_way_threshold_grid": sim}, config.BUSINESS_SIM_JSON)
    if config.ENABLE_REVIEW_BUDGET_ANALYSIS:
        from src.business import review_budget_analysis

        budget_report = review_budget_analysis(
            y_va.to_numpy(),
            val_scores,
            val_amount,
            budgets=config.REVIEW_BUDGETS,
        )
        save_json({"review_budget_analysis": budget_report}, config.REVIEW_BUDGET_JSON)

    val_out = val_ids.copy()
    val_out["y_true"] = y_va.values
    val_out["pred_fraud_prob"] = val_scores
    val_out["fraud_loss_proxy"] = fraud_loss_proxy(val_scores, val_amount)
    val_out.to_csv(config.VAL_PREDICTIONS_CSV, index=False)

    test_scores = predict_proba_positive(model, X_te)
    validate_prediction_range(test_scores)
    sub = test_ids.copy()
    sub[config.LABEL_COL] = test_scores
    sub.to_csv(config.TEST_PREDICTIONS_CSV, index=False)

    plot_roc(y_va.to_numpy(), val_scores, config.PLOTS_DIR / "roc_curve.png")
    plot_pr(y_va.to_numpy(), val_scores, config.PLOTS_DIR / "pr_curve.png")
    plot_ks(y_va.to_numpy(), val_scores, config.PLOTS_DIR / "ks_curve.png")
    plot_feature_importance(model, list(X_tr.columns), config.PLOTS_DIR / "feature_importance.png")

    if config.ENABLE_SHAP and config.SHAP_MAX_SAMPLES and config.SHAP_MAX_SAMPLES > 0:
        try:
            plot_shap_summary(
                model,
                X_va,
                list(X_tr.columns),
                config.PLOTS_DIR / "shap_summary_bar.png",
                max_samples=config.SHAP_MAX_SAMPLES,
            )
        except Exception as exc:
            LOGGER.warning("SHAP plot skipped: %s", exc)

    LOGGER.info("Done.")
    LOGGER.info("Model: %s", config.MODEL_PATH)
    LOGGER.info(
        "Val metrics | ROC-AUC=%.4f PR-AUC=%.4f KS=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["ks_statistic"],
    )
    LOGGER.info("Test predictions: %s", config.TEST_PREDICTIONS_CSV)


if __name__ == "__main__":
    main()

