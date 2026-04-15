"""Config for IEEE-CIS fraud detection pipeline"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Raw inputs live under data/raw/.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def _resolve(name: str) -> Path:
    p = RAW_DIR / name
    if p.is_file():
        return p
    raise FileNotFoundError(f"Missing {name}. Place it in data/raw/.")


TRAIN_TRANSACTION = _resolve("train_transaction.csv")
TRAIN_IDENTITY = _resolve("train_identity.csv")
TEST_TRANSACTION = _resolve("test_transaction.csv")
TEST_IDENTITY = _resolve("test_identity.csv")
SAMPLE_SUBMISSION = _resolve("sample_submission.csv")

# Outputs
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_JSON = METRICS_DIR / "val_metrics.json"
BUSINESS_SIM_JSON = METRICS_DIR / "business_simulation.json"
VAL_PREDICTIONS_CSV = METRICS_DIR / "val_predictions.csv"
TEST_PREDICTIONS_CSV = METRICS_DIR / "test_predictions.csv"
RUN_CONFIG_JSON = METRICS_DIR / "run_config.json"
FEATURE_COLUMNS_JSON = METRICS_DIR / "feature_columns.json"
DATA_SUMMARY_JSON = METRICS_DIR / "data_summary.json"
REVIEW_BUDGET_JSON = METRICS_DIR / "review_budget_analysis.json"

# Columns
ID_COL = "TransactionID"
LABEL_COL = "isFraud"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"

# Run knobs
RANDOM_SEED = 42
RANDOM_STATE = RANDOM_SEED
VAL_FRACTION = 0.2
TRAIN_ROW_CAP: int | None = None
TEST_ROW_CAP: int | None = None

# Encoding knobs
MAX_CATEGORY_CARDINALITY = 2000

# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 8000,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 200

# Interpretability
ENABLE_SHAP = True
SHAP_MAX_SAMPLES = 1000  # set 0 to skip

# Business simulation
ENABLE_REVIEW_BUDGET_ANALYSIS = True
REVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10]
ALLOW_THRESHOLDS = [0.02, 0.05, 0.10]
BLOCK_THRESHOLDS = [0.50, 0.70, 0.90]

# Reproducibility artifacts
SAVE_RUN_CONFIG = True
SAVE_FEATURE_COLUMNS = True
SAVE_DATA_SUMMARY = True

