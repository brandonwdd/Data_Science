"""Paths and hyperparameters — Lending Club accepted-loan default / EL pipeline"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

ACCEPTED_CSV = RAW_DIR / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = RAW_DIR / "rejected_2007_to_2018Q4.csv"

PD_MODEL_PATH = MODELS_DIR / "pd_model.joblib"
LGD_MODEL_PATH = MODELS_DIR / "lgd_model.joblib"
METRICS_JSON = METRICS_DIR / "val_metrics.json"
BUSINESS_SIM_JSON = METRICS_DIR / "business_simulation.json"
PREDICTIONS_CSV = METRICS_DIR / "predictions.csv"

ISSUE_DATE_COL = "issue_d"
TARGET_COL = "target"
EXPOSURE_COL = "loan_amnt"

RANDOM_STATE = 42
VAL_FRACTION = 0.2
# None = full data; set e.g. 80000 for a faster run.
ROW_CAP: int | None = None

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 4000,
    "learning_rate": 0.05,
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
EARLY_STOPPING_ROUNDS = 100

SHAP_MAX_SAMPLES = 1500

# LGD model (regression on defaulted loans only)
LGBM_LGD_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "n_estimators": 4000,
    "learning_rate": 0.05,
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


def resolve_accepted_csv() -> Path:
    if ACCEPTED_CSV.is_file():
        return ACCEPTED_CSV
    raise FileNotFoundError(
        "Place accepted_2007_to_2018Q4.csv in data/raw/."
    )
