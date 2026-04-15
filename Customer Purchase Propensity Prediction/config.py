"""Central configuration for the customer purchase propensity pipeline"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (project root = directory containing this file)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
RAW_DATA_PATH = RAW_DIR / "online_retail_II.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_JSON = METRICS_DIR / "val_metrics.json"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Time windows (days)
# -----------------------------------------------------------------------------
OBSERVATION_DAYS = 180
GAP_DAYS = 0
PREDICTION_DAYS = 30

# Cutoffs derived from max(InvoiceDate) in cleaned data (see run_pipeline / labeling helpers)
# Test: label interval (test_cutoff, max_date] requires max_date >= test_cutoff + PREDICTION_DAYS as dates;
# we set test_cutoff = max_date - PREDICTION_DAYS so the prediction window is the last PREDICTION_DAYS.
TEST_CUTOFF_DAYS_BEFORE_END = PREDICTION_DAYS
# Train cutoff further back so train label window ends before test label window starts.
TRAIN_CUTOFF_DAYS_BEFORE_END = 75

# -----------------------------------------------------------------------------
# Country encoding
# -----------------------------------------------------------------------------
COUNTRY_ENCODING = "top_n_one_hot"
TOP_N_COUNTRIES = 10
OTHER_COUNTRY_LABEL = "Other"

# -----------------------------------------------------------------------------
# Column mapping (raw Online Retail II -> internal snake_case)
# -----------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

REQUIRED_RAW_COLUMNS = list(COLUMN_RENAME_MAP.keys())

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Early stopping on time-based validation slice
ES_ROUNDS = 50
