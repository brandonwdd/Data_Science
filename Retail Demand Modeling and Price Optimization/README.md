# Retail Demand Modeling and Price Optimization

**This project builds an end to end retail demand modeling and price optimization pipeline that predicts daily product store quantities from historical transactions and simulates revenue impact across a price grid to recommend an approximate optimal menu price per product**

**This is a useful pricing model because it produces a quantitative demand forecast and a consistent uplift signal where the simulated mean revenue uplift is 11.6071 percent and 96.25 percent of products show positive uplift which means the price grid optimization can surface menu price adjustments that are directionally revenue improving while remaining explainable via elasticity estimates**

Dataset source Coffee Shop Sales (Excel). Put `coffee_shop_sales.xlsx` in `data/raw/`

| File | Rows | Columns | Column examples |
|---|---:|---:|---|
| `coffee_shop_sales.xlsx` | varies | 11 | `transaction_id`, `transaction_date`, `transaction_time`, `transaction_qty`, `store_id`, `store_location`, `product_id`, `unit_price`, `product_category`, `product_type`, `product_detail` |

## Pipeline steps

1. Input setup Put `coffee_shop_sales.xlsx` in `data/raw/` and install pinned deps from `requirements.txt`
2. Load and clean Read the Excel file validate schema parse `transaction_date` filter invalid quantity and price and add `total_amount`
3. Daily panel Aggregate to daily product store level targets including `total_quantity` and `avg_price` then build modeling frame with calendar features lags rolling averages and price transforms
4. Time based split Reserve the last `HOLDOUT_DAYS` calendar days as a strict hold out test period
5. Algorithm Train `lightgbm.LGBMRegressor` to predict `total_quantity` using engineered features
6. Hold out evaluation Score the hold out days compute RMSE MAE R² and export `outputs/metrics/regression_test.json`
7. Elasticity estimation Fit a per product log log regression on the training window to estimate price elasticity and confidence flags
8. Price simulation and optimization For each product simulate a price grid around the current price band and pick the price that maximizes predicted revenue then export recommendations and uplift summary

## Outputs and model evidence

| Metric | Value | Evidence file |
|---|---:|---|
| Hold out RMSE | 4.4977 | `outputs/metrics/val_metrics.json` |
| Hold out MAE | 3.2740 | `outputs/metrics/val_metrics.json` |
| Hold out R² | 0.3662 | `outputs/metrics/val_metrics.json` |
| Mean revenue uplift percent simulated | 11.6071 | `outputs/metrics/val_metrics.json` |
| Share of products with positive uplift | 0.9625 | `outputs/metrics/val_metrics.json` |

## Project directory

| Path | Description |
|---|---|
| `.gitignore` | Prevents committing local env files raw data and tabular artifacts |
| `README.md` | Documents objective dataset pipeline evidence and file map |
| `config.py` | Central config for paths hold out window price grid and model params |
| `data/raw/coffee_shop_sales.xlsx` | Raw Excel transaction source file |
| `data/processed/daily_product_store_panel.csv` | Daily product store modeling panel produced by the pipeline |
| `outputs/metrics/val_metrics.json` | Hold out evaluation bundle including regression metrics and pricing uplift summary |
| `outputs/metrics/regression_test.json` | Regression metrics on the hold out days |
| `outputs/metrics/pricing_uplift_summary.json` | Aggregate uplift summary across products |
| `outputs/metrics/pricing_recommendations.csv` | Per product current price optimal price and uplift fields plus elasticity estimates |
| `outputs/models/demand_lgbm.joblib` | Trained LightGBM demand regressor |
| `outputs/plots/feature_importance.png` | Feature importance plot from the trained model |
| `outputs/plots/price_curve_product_*.png` | Example product revenue demand curves from price simulation |
| `requirements.txt` | Exact dependency versions required to reproduce the run |
| `run_pipeline.py` | Orchestrates load clean aggregation feature build training evaluation elasticity optimization and exports |
| `src/aggregation.py` | Builds daily product store targets from transaction rows |
| `src/data_loader.py` | Reader and schema validator for the raw Excel file |
| `src/elasticity.py` | Per product elasticity estimation utilities |
| `src/evaluate.py` | Regression metrics feature importance plots uplift summary and curve plotting |
| `src/feature_engineering.py` | Calendar lag rolling and price transform features |
| `src/model.py` | LightGBM train predict split and serialization helpers |
| `src/optimization.py` | Revenue simulation grid and elasticity merge for recommendations |
| `src/preprocess.py` | Transaction cleaning and enrichment utilities |

