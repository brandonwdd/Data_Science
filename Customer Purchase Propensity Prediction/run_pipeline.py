"""Run the purchase propensity pipeline"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.data_loader import load_raw_transactions  # noqa: E402
from src.dataset_builder import build_modeling_matrices  # noqa: E402
from src.evaluate import evaluate_and_save  # noqa: E402
from src.feature_engineering import build_customer_features  # noqa: E402
from src.labeling import build_label_table  # noqa: E402
from src.preprocess import clean_transactions  # noqa: E402
from src.train import feature_importance_df, train_with_time_valid  # noqa: E402
from src.utils import ensure_dirs  # noqa: E402


def compute_cutoffs(clean: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    data_max = clean["invoice_date"].max().normalize()
    min_date = clean["invoice_date"].min().normalize()

    test_cutoff = data_max - pd.Timedelta(days=config.TEST_CUTOFF_DAYS_BEFORE_END)
    train_cutoff = data_max - pd.Timedelta(days=config.TRAIN_CUTOFF_DAYS_BEFORE_END)

    train_obs_start = train_cutoff - pd.Timedelta(days=config.OBSERVATION_DAYS - 1)
    if train_obs_start < min_date:
        raise ValueError(
            "Train observation window extends before the first transaction date. "
            "Reduce TRAIN_CUTOFF_DAYS_BEFORE_END or OBSERVATION_DAYS."
        )

    if not (test_cutoff > train_cutoff):
        raise ValueError("test_cutoff must be after train_cutoff; adjust config offsets.")

    return train_cutoff, test_cutoff


def main() -> None:
    ensure_dirs(
        config.RAW_DIR,
        config.INTERIM_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.PLOTS_DIR,
    )

    print("Loading raw transactions…")
    raw = load_raw_transactions()
    print("Cleaning…")
    clean = clean_transactions(raw)
    clean.to_parquet(config.INTERIM_DIR / "transactions_clean.parquet", index=False)

    train_cutoff, test_cutoff = compute_cutoffs(clean)
    print(f"data_max={clean['invoice_date'].max().normalize()} train_cutoff={train_cutoff} test_cutoff={test_cutoff}")

    print("Building train features / labels…")
    train_features = build_customer_features(
        clean, train_cutoff, config.OBSERVATION_DAYS
    )
    train_labels = build_label_table(
        clean,
        train_features["customer_id"],
        train_cutoff,
        config.GAP_DAYS,
        config.PREDICTION_DAYS,
    )

    print("Building hold-out test features / labels…")
    test_features = build_customer_features(clean, test_cutoff, config.OBSERVATION_DAYS)
    test_labels = build_label_table(
        clean,
        test_features["customer_id"],
        test_cutoff,
        config.GAP_DAYS,
        config.PREDICTION_DAYS,
    )

    print("Encoding + matrices…")
    X_train, y_train, X_test, y_test, last_purchase_train = build_modeling_matrices(
        train_features, train_labels, test_features, test_labels
    )
    print(f"Train rows={len(X_train):,}  Test rows={len(X_test):,}  pos_rate_train={y_train.mean():.3f}")

    print("Training LightGBM…")
    model, feature_names = train_with_time_valid(
        X_train,
        y_train,
        last_purchase_train,
        train_cutoff,
        config.OBSERVATION_DAYS,
        config.MODELS_DIR,
    )

    print("Evaluating on hold-out…")
    y_score = model.predict_proba(X_test)[:, 1]
    fi = feature_importance_df(model, feature_names)
    metrics = evaluate_and_save(
        y_test.to_numpy(),
        y_score,
        config.METRICS_DIR,
        config.PLOTS_DIR,
        feature_importance=fi,
    )

    print("Done. Key test metrics:")
    for k in (
        "roc_auc",
        "pr_auc",
        "precision_0p5",
        "recall_0p5",
        "f1_0p5",
        "precision_at_10pct",
        "recall_at_10pct",
        "lift_top_decile",
        "positive_rate",
    ):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
