"""
forecasting.py
--------------
Builds, evaluates and uses the sales forecasting models.

Models compared (simple -> advanced):
  1. Seasonal Naive   : "sales = average of the same weekday over the last
                        4 weeks" (the baseline every model must beat)
  2. Linear Regression: straight-line relationship between features & sales
  3. Random Forest    : many decision trees averaged -> captures non-linear
                        effects such as festival x category interactions

Evaluation is done HONESTLY with a time-based split and RECURSIVE
forecasting: during the 56 test days the model never sees real sales - it
feeds its own predictions back in as lag features, exactly like real life.

Metrics
  MAE  : average absolute error in units
  RMSE : like MAE but punishes big misses more
  WAPE : total absolute error / total actual sales  (business-friendly %)
  Bias : are we systematically over (+) or under (-) forecasting?
  Weekly WAPE : WAPE after summing to weekly store-product totals - this is
         the level at which replenishment orders are actually planned
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_generator import is_festival_period
from src.feature_engineering import FEATURE_COLUMNS, build_features

HISTORY_DAYS_NEEDED = config.HISTORY_DAYS
BASE_COLS = ["date", "store_id", "product_id", "category", "unit_price",
             "promotion", "festival_period", config.TARGET]


class SeasonalNaive:
    """Baseline model: prediction = average of the same weekday, last 4 weeks."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["same_weekday_mean_4w"].values


def get_models() -> dict:
    return {
        "Seasonal Naive (baseline)": SeasonalNaive(),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=3,
            n_jobs=-1, random_state=config.RANDOM_SEED),
    }


def evaluate(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    err = predicted - actual
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "WAPE_%": np.abs(err).sum() / actual.sum() * 100,
        "Bias_%": err.sum() / actual.sum() * 100,
        "Accuracy_%": 100 - np.abs(err).sum() / actual.sum() * 100,
    }


def recursive_forecast(model, history: pd.DataFrame,
                       future: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast day by day. After predicting day D, the prediction is written
    into the history so that day D+1 can use it as its 'lag_1' feature.
    `future` must contain the BASE_COLS except the target.
    """
    hist = history[BASE_COLS].copy()
    results = []
    for day in sorted(future["date"].unique()):
        day = pd.Timestamp(day)
        day_rows = future[future["date"] == day].copy()
        day_rows[config.TARGET] = np.nan
        window_start = day - pd.Timedelta(days=HISTORY_DAYS_NEEDED)
        work = pd.concat([hist[hist["date"] >= window_start], day_rows[BASE_COLS]],
                         ignore_index=True)
        feats = build_features(work)
        today = feats[feats["date"] == day].copy()
        today["prediction"] = np.clip(model.predict(today[FEATURE_COLUMNS]), 0, None)

        results.append(today[["date", "store_id", "product_id", "category",
                              "prediction"]])
        filled = today[BASE_COLS].copy()
        filled[config.TARGET] = today["prediction"].values
        hist = pd.concat([hist, filled], ignore_index=True)
    return pd.concat(results, ignore_index=True)


def train_and_evaluate(clean: pd.DataFrame, feats: pd.DataFrame):
    """Train every model on the training period, test on the last TEST_DAYS."""
    cutoff = clean["date"].max() - pd.Timedelta(days=config.TEST_DAYS)
    train = feats[feats["date"] <= cutoff]
    history = clean[clean["date"] <= cutoff]
    test_actual = clean[clean["date"] > cutoff]
    future_frame = test_actual[BASE_COLS].drop(columns=[config.TARGET])

    print(f"[forecasting] Train: up to {cutoff.date()} ({len(train):,} rows) | "
          f"Test: {config.TEST_DAYS} days recursive")

    metrics, test_preds, fitted = [], {}, {}
    for name, model in get_models().items():
        model.fit(train[FEATURE_COLUMNS], train[config.TARGET])
        pred = recursive_forecast(model, history, future_frame)
        merged = pred.merge(test_actual[["date", "store_id", "product_id",
                                         config.TARGET]],
                            on=["date", "store_id", "product_id"])
        m = evaluate(merged[config.TARGET], merged["prediction"])
        weekly = merged.groupby(["store_id", "product_id",
                                 pd.Grouper(key="date", freq="W")])[
            [config.TARGET, "prediction"]].sum()
        m["Weekly_WAPE_%"] = evaluate(weekly[config.TARGET],
                                      weekly["prediction"])["WAPE_%"]
        m["Model"] = name
        metrics.append(m)
        test_preds[name] = merged
        fitted[name] = model
        print(f"    {name:<28} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"WAPE={m['WAPE_%']:.1f}%  Weekly WAPE={m['Weekly_WAPE_%']:.1f}%  "
              f"Bias={m['Bias_%']:+.1f}%")

    metrics_df = pd.DataFrame(metrics).set_index("Model").round(3)
    best_name = metrics_df["WAPE_%"].idxmin()
    print(f"[forecasting] Best model: {best_name}")
    return metrics_df, best_name, test_preds, fitted


def forecast_error_stats(test_pred: pd.DataFrame) -> pd.DataFrame:
    """Std-dev of daily forecast error per store-product (for safety stock)."""
    test_pred = test_pred.assign(error=test_pred[config.TARGET]
                                 - test_pred["prediction"])
    return (test_pred.groupby(["store_id", "product_id"])
            .agg(forecast_error_std=("error", "std"),
                 test_mae=("error", lambda e: e.abs().mean()))
            .reset_index())


def build_future_frame(clean: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Rows for the next FORECAST_HORIZON days with known/planned inputs."""
    last = clean["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1),
                          periods=config.FORECAST_HORIZON, freq="D")
    keys = clean[["store_id", "product_id", "category"]].drop_duplicates()
    future = keys.merge(pd.DataFrame({"date": dates}), how="cross")
    future = future.merge(products[["product_id", "unit_price"]], on="product_id")
    future["promotion"] = 0  # no promotions planned (can be edited)
    fest = pd.Series(is_festival_period(dates), index=dates)
    future["festival_period"] = future["date"].map(fest)
    return future[BASE_COLS[:-1]]


def run_forecasting(clean: pd.DataFrame | None = None,
                    feats: pd.DataFrame | None = None) -> dict:
    config.make_dirs()
    if clean is None:
        clean = pd.read_csv(config.CLEAN_SALES_FILE, parse_dates=["date"])
    if feats is None:
        feats = pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])
    products = pd.read_csv(config.PRODUCTS_FILE)

    metrics_df, best_name, test_preds, fitted = train_and_evaluate(clean, feats)
    metrics_df.to_csv(config.OUTPUTS_DIR / "model_metrics.csv")

    best_test = test_preds[best_name]
    best_test.to_csv(config.OUTPUTS_DIR / "test_predictions.csv", index=False)
    err_stats = forecast_error_stats(best_test)
    err_stats.to_csv(config.OUTPUTS_DIR / "forecast_error_stats.csv", index=False)

    # Feature importance (only tree models have it)
    importance = None
    if hasattr(fitted[best_name], "feature_importances_"):
        importance = (pd.Series(fitted[best_name].feature_importances_,
                                index=FEATURE_COLUMNS)
                      .sort_values(ascending=False))
        importance.to_csv(config.OUTPUTS_DIR / "feature_importance.csv",
                          header=["importance"])

    # Retrain the best model on ALL data, then forecast the real future
    final_model = get_models()[best_name]
    final_model.fit(feats[FEATURE_COLUMNS], feats[config.TARGET])
    joblib.dump(final_model, config.MODELS_DIR / "best_forecasting_model.joblib")
    meta = {"model": best_name, "features": FEATURE_COLUMNS,
            "trained_until": str(clean["date"].max().date()),
            "test_metrics": metrics_df.loc[best_name].to_dict()}
    (config.MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    future = build_future_frame(clean, products)
    forecast = recursive_forecast(final_model, clean, future)
    forecast["prediction"] = forecast["prediction"].round(2)
    forecast = forecast.rename(columns={"prediction": "forecast_units"})
    forecast.to_csv(config.OUTPUTS_DIR / "forecast_next_30_days.csv", index=False)
    print(f"[forecasting] {config.FORECAST_HORIZON}-day forecast saved "
          f"({len(forecast):,} rows) | model saved to models/")

    return {"metrics": metrics_df, "best_name": best_name,
            "test_preds": test_preds, "forecast": forecast,
            "error_stats": err_stats, "importance": importance}


if __name__ == "__main__":
    run_forecasting()
