"""Project configuration for the retail pricing pipeline"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Raw Excel input (place file here or override RAW_SALES_PATH)
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_SALES_PATH = DATA_RAW / "coffee_shop_sales.xlsx"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_PLOTS = PROJECT_ROOT / "outputs" / "plots"

# Honest naming: this is the time-based hold-out evaluation metrics bundle
VAL_METRICS_JSON = OUTPUT_METRICS / "val_metrics.json"

# Time-based holdout: last N calendar days reserved for test
HOLDOUT_DAYS = 30

# Price simulation (per product, around observed "current" price)
PRICE_BAND_LOW = 0.9
PRICE_BAND_HIGH = 1.1
PRICE_GRID_POINTS = 21

# "Current price" = volume-weighted average of daily avg_price over last K train days
CURRENT_PRICE_LOOKBACK_DAYS = 14

# Elasticity: flag as high-confidence only if enough distinct historical prices
ELASTICITY_MIN_DISTINCT_PRICES_HIGH = 3
ELASTICITY_MIN_ROWS = 15

# LightGBM
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 48,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
}

# Plotting: revenue/demand curves for top-N products by train-period volume
PLOT_TOP_N_PRODUCTS = 6

RANDOM_SEED = 42


def ensure_output_dirs() -> None:
    for p in (
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        OUTPUT_MODELS,
        OUTPUT_METRICS,
        OUTPUT_PLOTS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def apply_menu_rounding(price: float) -> float:
    """Hook for future menu rounding (e.g. to nearest $0.05). Version 1: tidy decimals only."""
    return round(float(price), 4)
