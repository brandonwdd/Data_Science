"""Run the retail pricing pipeline"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.aggregation import aggregate_daily_product_store
from src.data_loader import load_transactions
from src.elasticity import estimate_elasticity_per_product
from src.evaluate import (
    plot_feature_importance,
    plot_product_price_curves,
    pricing_uplift_summary,
    save_regression_metrics,
    save_uplift_summary,
)
from src.feature_engineering import build_modeling_frame
from src.model import (
    evaluate_regression,
    feature_columns,
    predict_quantities,
    save_model,
    time_based_split,
    train_lgbm,
)
from src.optimization import merge_elasticity, simulate_revenue_grid, volume_weighted_current_price
from src.preprocess import preprocess_transactions


def main() -> None:
    config.ensure_output_dirs()
    raw_path = config.RAW_SALES_PATH
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    print("Loading & preprocessing…")
    tx = load_transactions(raw_path)
    tx = preprocess_transactions(tx)
    daily = aggregate_daily_product_store(tx)
    panel = build_modeling_frame(daily)

    panel.to_csv(config.DATA_PROCESSED / "daily_product_store_panel.csv", index=False)

    print("Time-based split…")
    train_df, test_df, cutoff = time_based_split(panel, config.HOLDOUT_DAYS)
    train_end = train_df["transaction_date"].max()
    print(f"  Train: through {train_end.date()} | Test: after {cutoff.date()}")

    print("Training LightGBM…")
    model = train_lgbm(train_df, config.LGBM_PARAMS)

    print("Hold-out evaluation…")
    if test_df.empty:
        raise ValueError("Hold-out test set is empty; reduce HOLDOUT_DAYS.")
    y_te = test_df["total_quantity"].to_numpy(dtype=float)
    y_hat = predict_quantities(model, test_df)
    reg_metrics = evaluate_regression(y_te, y_hat)
    reg_metrics["holdout_days"] = config.HOLDOUT_DAYS
    reg_metrics["test_end_date"] = str(panel["transaction_date"].max().date())
    save_regression_metrics(reg_metrics, config.OUTPUT_METRICS / "regression_test.json")
    print(f"  RMSE={reg_metrics['rmse']:.3f} MAE={reg_metrics['mae']:.3f} R²={reg_metrics['r2']:.3f}")

    save_model(model, config.OUTPUT_MODELS / "demand_lgbm.joblib")

    imp = model.booster_.feature_importance(importance_type="gain")
    names = feature_columns()
    plot_feature_importance(
        np.array(imp, dtype=float),
        names,
        config.OUTPUT_PLOTS / "feature_importance.png",
    )

    print("Elasticity (log-log, train window only)…")
    elas = estimate_elasticity_per_product(
        train_df,
        min_distinct_prices_high=config.ELASTICITY_MIN_DISTINCT_PRICES_HIGH,
        min_rows=config.ELASTICITY_MIN_ROWS,
    )

    print("Price simulation & optimization…")
    current_px = volume_weighted_current_price(
        panel,
        train_end=train_end,
        lookback_days=config.CURRENT_PRICE_LOOKBACK_DAYS,
    )
    summary, detail = simulate_revenue_grid(
        model,
        panel,
        train_end=train_end,
        current_prices=current_px,
        price_low=config.PRICE_BAND_LOW,
        price_high=config.PRICE_BAND_HIGH,
        n_points=config.PRICE_GRID_POINTS,
    )

    recs = merge_elasticity(summary, elas)
    recs["optimal_price"] = recs["optimal_price"].map(config.apply_menu_rounding)

    out_cols = [
        "product_id",
        "product_name",
        "current_price",
        "optimal_price",
        "predicted_quantity_at_current_price",
        "predicted_quantity_at_optimal_price",
        "current_revenue",
        "optimal_revenue",
        "revenue_uplift_pct",
        "elasticity",
        "elasticity_confidence_flag",
    ]
    for c in out_cols:
        if c not in recs.columns:
            recs[c] = np.nan
    recs[out_cols].to_csv(config.OUTPUT_METRICS / "pricing_recommendations.csv", index=False)

    uplift = pricing_uplift_summary(recs)
    save_uplift_summary(uplift, config.OUTPUT_METRICS / "pricing_uplift_summary.json")
    print(f"  Mean revenue uplift (simulated): {uplift['mean_revenue_uplift_pct']:.2f}%")

    val_bundle = {
        "eval_split": "holdout_test",
        "regression": reg_metrics,
        "pricing_uplift": uplift,
    }
    with config.VAL_METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(val_bundle, f, indent=2)

    train_vol = (
        train_df.groupby("product_id")["total_quantity"].sum().sort_values(ascending=False)
    )
    top_ids = train_vol.head(config.PLOT_TOP_N_PRODUCTS).index.astype(int).tolist()
    plot_product_price_curves(detail, top_ids, config.OUTPUT_PLOTS)

    print("Done.")
    print(f"  Recommendations: {config.OUTPUT_METRICS / 'pricing_recommendations.csv'}")
    print(f"  Plots: {config.OUTPUT_PLOTS}")


if __name__ == "__main__":
    main()
