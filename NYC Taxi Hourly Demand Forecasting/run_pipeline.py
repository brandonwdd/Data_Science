"""Run the NYC taxi demand pipeline"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

import config
from src.aggregation import hourly_demand
from src.data_loader import load_train_csv
from src.evaluate import compute_metrics, plot_actual_vs_predicted, save_metrics
from src.feature_engineering import add_features, attach_hour_index
from src.forecasting import predict_demand
from src.model import train_demand_model
from src.preprocess import clean_trips


def ensure_dirs() -> None:
    for p in (
        config.DATA_RAW,
        config.DATA_INTERIM,
        config.DATA_PROCESSED,
        config.OUTPUT_MODELS,
        config.OUTPUT_METRICS,
        config.OUTPUT_PLOTS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not config.TRAIN_CSV.is_file():
        raise FileNotFoundError(f"Missing {config.TRAIN_CSV}")

    ensure_dirs()

    trips = load_train_csv(config.TRAIN_CSV)
    trips = clean_trips(trips)
    hourly = hourly_demand(trips)
    hourly.to_csv(config.PROCESSED_HOURLY_CSV, index=False)

    featured = add_features(hourly)
    usable = attach_hour_index(featured)

    n = len(usable)
    split = int(n * config.TRAIN_RATIO)
    if split < 100 or n - split < 24:
        raise ValueError("Not enough hourly rows after feature drop for a stable split.")

    train_df = usable.iloc[:split].reset_index(drop=True)
    val_df = usable.iloc[split:].reset_index(drop=True)

    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df["demand"]
    X_val = val_df[config.FEATURE_COLUMNS]
    y_val = val_df["demand"]

    model = train_demand_model(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=config.RANDOM_STATE,
    )

    pred = predict_demand(model, X_val)
    y_v = y_val.to_numpy()
    p = pred.to_numpy()
    lag1 = val_df["demand_lag_1"].to_numpy()
    lag168 = val_df["demand_lag_168"].to_numpy()
    metrics = {
        "eval_split": "validation",
        "lgbm": compute_metrics(y_v, p),
        "baseline_persistence_lag1": compute_metrics(y_v, lag1),
        "baseline_seasonal_lag168": compute_metrics(y_v, lag168),
    }
    save_metrics(metrics, config.METRICS_PATH)
    joblib.dump(model, config.MODEL_PATH)

    plot_actual_vs_predicted(
        val_df["hour"],
        y_val.to_numpy(),
        pred.to_numpy(),
        config.FORECAST_PLOT_PATH,
    )

    print("Rows (hourly, full grid):", len(hourly))
    print("Rows (with features):", n)
    print("Train hours:", len(train_df), "Validation hours:", len(val_df))
    print("Metrics (lgbm vs baselines on same validation hours):", metrics)
    print("Saved model:", config.MODEL_PATH)
    print("Saved metrics:", config.METRICS_PATH)
    print("Saved plot:", config.FORECAST_PLOT_PATH)


if __name__ == "__main__":
    main()
