"""Run the Home Credit risk pipeline"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

import config
from src import data_loader
from src.evaluation import (
    business_simulation,
    compute_metrics_bundle,
    plot_feature_importance,
    plot_ks,
    plot_roc,
    plot_shap_summary,
    save_metrics,
)
from src.leakage_filters import (
    filter_bureau_for_leakage,
    filter_installments_payments,
    filter_previous_application,
)
from src.feature_engineering import (
    build_feature_tables,
    ordered_train_val_split,
    prepare_xy,
    rename_features_sanitized,
)
from src.modeling import predict_proba_positive, train_lgbm


def ensure_dirs() -> None:
    for p in (
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def _filter_auxiliary_to_ids(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous_application: pd.DataFrame,
    pos_cash: pd.DataFrame,
    credit_card: pd.DataFrame,
    installments: pd.DataFrame,
    curr_ids: set,
) -> tuple[pd.DataFrame, ...]:
    bureau_f = bureau[bureau["SK_ID_CURR"].isin(curr_ids)].copy()
    bureau_ids = set(bureau_f["SK_ID_BUREAU"])
    bureau_balance_f = bureau_balance[bureau_balance["SK_ID_BUREAU"].isin(bureau_ids)].copy()
    prev_f = previous_application[previous_application["SK_ID_CURR"].isin(curr_ids)].copy()
    pos_f = pos_cash[pos_cash["SK_ID_CURR"].isin(curr_ids)].copy()
    cc_f = credit_card[credit_card["SK_ID_CURR"].isin(curr_ids)].copy()
    ins_f = installments[installments["SK_ID_CURR"].isin(curr_ids)].copy()
    return bureau_f, bureau_balance_f, prev_f, pos_f, cc_f, ins_f


def apply_row_caps(
    app_train: pd.DataFrame,
    app_test: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous_application: pd.DataFrame,
    pos_cash: pd.DataFrame,
    credit_card: pd.DataFrame,
    installments: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    if config.TRAIN_ROW_CAP is not None:
        app_train = app_train.head(config.TRAIN_ROW_CAP).copy()
    if config.TEST_ROW_CAP is not None:
        app_test = app_test.head(config.TEST_ROW_CAP).copy()
    if config.TRAIN_ROW_CAP is None and config.TEST_ROW_CAP is None:
        return (
            app_train,
            app_test,
            bureau,
            bureau_balance,
            previous_application,
            pos_cash,
            credit_card,
            installments,
        )
    curr_ids = set(app_train["SK_ID_CURR"]) | set(app_test["SK_ID_CURR"])
    bureau, bureau_balance, previous_application, pos_cash, credit_card, installments = (
        _filter_auxiliary_to_ids(
            bureau,
            bureau_balance,
            previous_application,
            pos_cash,
            credit_card,
            installments,
            curr_ids,
        )
    )
    return (
        app_train,
        app_test,
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        credit_card,
        installments,
    )


def align_feature_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    X_other = X_other.copy()
    for c in X_ref.columns:
        if c not in X_other.columns:
            X_other[c] = 0 if X_ref[c].dtype != bool else False
    return X_other[X_ref.columns]


def main() -> None:
    ensure_dirs()

    app_train = data_loader.load_application_train()
    app_test = data_loader.load_application_test()
    bureau = data_loader.load_bureau()
    bureau_balance = data_loader.load_bureau_balance()
    previous_application = data_loader.load_previous_application()
    pos_cash = data_loader.load_pos_cash_balance()
    credit_card = data_loader.load_credit_card_balance()
    installments = data_loader.load_installments_payments()

    (
        app_train,
        app_test,
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        credit_card,
        installments,
    ) = apply_row_caps(
        app_train,
        app_test,
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        credit_card,
        installments,
    )

    if config.APPLY_BUREAU_DAYS_CREDIT_FILTER:
        bureau = filter_bureau_for_leakage(bureau)
    previous_application = filter_previous_application(previous_application)
    installments = filter_installments_payments(installments)

    train_full, test_full = build_feature_tables(
        app_train,
        app_test,
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        credit_card,
        installments,
    )

    train_fit, val_fit = ordered_train_val_split(
        train_full,
        config.TIME_ORDER_COLS,
        config.VAL_FRACTION,
    )

    X_tr, y_tr = prepare_xy(train_fit)
    X_va, y_va = prepare_xy(val_fit)
    X_te, _ = prepare_xy(test_full)

    X_te = align_feature_columns(X_tr, X_te)

    X_tr, _map_tr = rename_features_sanitized(X_tr)
    X_va, _ = rename_features_sanitized(X_va)
    X_te, _ = rename_features_sanitized(X_te)
    X_va = align_feature_columns(X_tr, X_va)
    X_te = align_feature_columns(X_tr, X_te)

    model = train_lgbm(X_tr, y_tr, X_va, y_va)
    joblib.dump(model, config.MODEL_PATH)

    val_scores = predict_proba_positive(model, X_va)
    metrics = compute_metrics_bundle(y_va.to_numpy(), val_scores)
    metrics["n_train_fit"] = int(len(X_tr))
    metrics["n_val"] = int(len(X_va))
    metrics["n_test"] = int(len(X_te))
    metrics["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    metrics["leakage_filters"] = {
        "previous_application": "DAYS_DECISION < 0 (notna)",
        "installments_payments": "DAYS_ENTRY_PAYMENT < 0 (notna)",
        "bureau_days_credit_lt_0": bool(config.APPLY_BUREAU_DAYS_CREDIT_FILTER),
    }
    metrics["time_order_cols"] = list(config.TIME_ORDER_COLS)
    metrics["exposure_col"] = config.EXPOSURE_COL
    save_metrics(metrics, config.METRICS_JSON)

    exposure_va = None
    if config.EXPOSURE_COL in val_fit.columns:
        exposure_va = val_fit[config.EXPOSURE_COL].to_numpy(dtype=float)
    sim = business_simulation(
        y_va.to_numpy(),
        val_scores,
        exposure=exposure_va,
    )
    save_metrics({"business_simulation": sim}, config.BUSINESS_SIM_JSON)

    val_out = val_fit[["SK_ID_CURR"]].copy()
    val_out["TARGET"] = y_va.values
    val_out["pred_default_prob"] = val_scores
    val_out.to_csv(config.VAL_PREDICTIONS_CSV, index=False)

    test_scores = predict_proba_positive(model, X_te)
    sub = test_full[["SK_ID_CURR"]].copy()
    sub["TARGET"] = test_scores
    sub.to_csv(config.TEST_PREDICTIONS_CSV, index=False)

    plot_roc(y_va.to_numpy(), val_scores, config.PLOTS_DIR / "roc_curve.png")
    plot_ks(y_va.to_numpy(), val_scores, config.PLOTS_DIR / "ks_curve.png")
    plot_feature_importance(model, list(X_tr.columns), config.PLOTS_DIR / "feature_importance_top.png")

    if config.SHAP_MAX_SAMPLES and config.SHAP_MAX_SAMPLES > 0:
        try:
            plot_shap_summary(
                model,
                X_va,
                list(X_tr.columns),
                config.PLOTS_DIR / "shap_summary_bar.png",
                max_samples=config.SHAP_MAX_SAMPLES,
            )
        except Exception as exc:
            print(f"SHAP plot skipped: {exc}")

    print("Done.")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  Val ROC-AUC: {metrics['roc_auc']:.4f}  KS: {metrics['ks_statistic']:.4f}")
    print(f"  Test predictions: {config.TEST_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
