"""Paths and hyperparameters for the Home Credit default-risk pipeline"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# Kaggle Home Credit CSVs live in data/raw/.
RAW_DATA_DIR = RAW_DIR
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
APPLICATION_TRAIN = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST = RAW_DATA_DIR / "application_test.csv"
BUREAU = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE = RAW_DATA_DIR / "bureau_balance.csv"
PREVIOUS_APPLICATION = RAW_DATA_DIR / "previous_application.csv"
POS_CASH_BALANCE = RAW_DATA_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE = RAW_DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS = RAW_DATA_DIR / "installments_payments.csv"

PROCESSED_TRAIN_FEATURES = PROCESSED_DIR / "train_features.parquet"
PROCESSED_TEST_FEATURES = PROCESSED_DIR / "test_features.parquet"
MODEL_PATH = MODELS_DIR / "lgbm_model.joblib"
METRICS_JSON = METRICS_DIR / "val_metrics.json"
VAL_PREDICTIONS_CSV = METRICS_DIR / "val_predictions.csv"
TEST_PREDICTIONS_CSV = METRICS_DIR / "test_predictions.csv"
BUSINESS_SIM_JSON = METRICS_DIR / "business_simulation.json"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
# Sort training rows by these columns (ascending), then hold out the last VAL_FRACTION
# as validation — proxy for application-time ordering using relative date fields.
TIME_ORDER_COLS: list[str] = ["DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "SK_ID_CURR"]
VAL_FRACTION = 0.2
# Exposure column on application* tables for expected-loss proxy in business simulation.
EXPOSURE_COL = "AMT_CREDIT"
# Optional: limit rows for quick dev runs (None = full data).
TRAIN_ROW_CAP: int | None = None  # e.g. 6000 for a faster smoke test
TEST_ROW_CAP: int | None = None
# SHAP (TreeExplainer) — subsample for speed; set max_samples 0 to skip.
SHAP_MAX_SAMPLES = 2000
# Extra guard: keep only bureau lines with DAYS_CREDIT < 0 (off by default; README focuses on prev/installments).
APPLY_BUREAU_DAYS_CREDIT_FILTER = False

# -----------------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 48,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 100
