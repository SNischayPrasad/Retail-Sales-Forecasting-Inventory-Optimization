"""Tests that feature engineering has no data leakage and metrics are correct."""
import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.forecasting import evaluate


def _toy_data(days=40):
    dates = pd.date_range("2024-01-01", periods=days)
    return pd.DataFrame({
        "date": dates, "store_id": "S01", "product_id": "P001",
        "category": "Groceries", "unit_price": 100.0, "promotion": 0,
        "festival_period": 0, "units_sold": np.arange(days, dtype=float),
    })


def test_all_feature_columns_created():
    feats = build_features(_toy_data())
    assert set(FEATURE_COLUMNS).issubset(feats.columns)


def test_lag_features_use_only_past_values():
    feats = build_features(_toy_data())
    row = feats.iloc[35]  # units_sold = 35
    assert row["lag_14"] == 21
    assert row["lag_28"] == 7
    # same weekday, last 4 weeks: (28 + 21 + 14 + 7) / 4
    assert row["same_weekday_mean_4w"] == 17.5
    # mean of the previous 28 days (7..34) - today's value is never included
    assert row["normal_day_mean_28"] == 20.5


def test_festival_days_excluded_from_baseline():
    data = _toy_data()
    data.loc[30:34, "festival_period"] = 1
    data.loc[30:34, "units_sold"] = 1000  # festival spike
    feats = build_features(data)
    assert feats.iloc[35]["normal_day_mean_28"] < 100


def test_evaluate_perfect_forecast():
    m = evaluate([10, 20, 30], [10, 20, 30])
    assert m["MAE"] == 0 and m["WAPE_%"] == 0


def test_evaluate_wape():
    # errors 10 on total actual 100 -> 10%
    m = evaluate([50, 50], [55, 45])
    assert round(m["WAPE_%"], 6) == 10
    assert m["Bias_%"] == 0
