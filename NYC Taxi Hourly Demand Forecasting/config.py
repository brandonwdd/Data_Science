"""Paths and hyperparameters for the demand forecasting pipeline"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
TRAIN_CSV = DATA_RAW / "train.csv"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

OUTPUT_MODELS = ROOT / "outputs" / "models"
OUTPUT_METRICS = ROOT / "outputs" / "metrics"
OUTPUT_PLOTS = ROOT / "outputs" / "plots"

TRAIN_RATIO = 0.8
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_168",
    "rolling_mean_3",
    "rolling_mean_24",
    "rolling_mean_168",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
]

PROCESSED_HOURLY_CSV = DATA_PROCESSED / "hourly_demand.csv"
MODEL_PATH = OUTPUT_MODELS / "demand_model.joblib"
METRICS_PATH = OUTPUT_METRICS / "val_metrics.json"
FORECAST_PLOT_PATH = OUTPUT_PLOTS / "demand_forecast.png"
