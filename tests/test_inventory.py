"""Unit tests for the inventory formulas. Run with:  pytest -v"""
import numpy as np
import pandas as pd
import pytest

from src.inventory_optimization import (classify_stock, economic_order_quantity,
                                        reorder_point, safety_stock,
                                        simulate_policy, stockout_date)


def test_safety_stock_formula():
    # Z=1.645, sigma=10, lead time=4 -> 1.645*10*2 = 32.9 -> ceil 33
    assert safety_stock(10, 4, z=1.645) == 33


def test_safety_stock_zero_variability():
    assert safety_stock(0, 5) == 0


def test_reorder_point():
    assert reorder_point(50, 33) == 83


def test_eoq_textbook_example():
    # D=1000, S=500, H=0.2*100=20 -> sqrt(2*1000*500/20)=223.6 -> 224
    assert economic_order_quantity(1000, 100, 500, 0.2) == 224


def test_classify_stock():
    current = np.array([5, 40, 500, 60])
    ss = np.array([10, 10, 10, 10])
    rop = np.array([50, 50, 50, 50])
    cover = np.array([1, 4, 90, 20])
    assert list(classify_stock(current, ss, rop, cover)) == [
        "CRITICAL", "REORDER", "OVERSTOCK", "HEALTHY"]


def test_stockout_date():
    fc = pd.Series([10, 10, 10], index=pd.date_range("2025-01-01", periods=3))
    assert str(stockout_date(fc, 25)) == "2025-01-03"
    assert stockout_date(fc, 100) is None


def test_simulation_never_negative_and_conserves_units():
    demand = np.full((1, 30), 10.0)
    stock, sold, orders = simulate_policy(demand, np.array([50]), np.array([3]),
                                          np.array([40]), np.array([100]))
    assert (stock >= 0).all()
    assert (sold <= demand).all()
    assert orders > 0


@pytest.mark.parametrize("sigma,lt", [(5, 1), (5, 9)])
def test_longer_lead_time_needs_more_safety_stock(sigma, lt):
    assert safety_stock(sigma, lt) >= safety_stock(sigma, 1)
