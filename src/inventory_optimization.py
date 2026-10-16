"""
inventory_optimization.py
-------------------------
Converts the sales forecast into STOCK DECISIONS.

Formulas (classic supply-chain textbook formulas used in industry):

  Safety Stock (SS)      = Z x sigma_daily x sqrt(Lead Time)
        Z            -> from the service level (95% -> 1.645)
        sigma_daily  -> std-dev of daily forecast error (uncertainty)
  Lead-Time Demand       = forecast units during the supplier lead time
  Reorder Point (ROP)    = Lead-Time Demand + Safety Stock
        "When stock falls to this level, place an order."
  Economic Order Qty     = sqrt( 2 x Annual Demand x Ordering Cost
                                 / Holding Cost per unit per year )
        "The order size that minimises ordering + holding cost."
  Target (order-up-to)   = avg daily forecast x (Lead Time + Review Period) + SS
  Days of Cover          = Current Stock / avg daily forecast

Decision rules
  stock <= SS                   -> CRITICAL   (stockout very likely)
  stock <= ROP                  -> REORDER    (place an order today)
  days of cover > 60            -> OVERSTOCK  (stop ordering / run promotion)
  otherwise                     -> HEALTHY

This module also contains a VIRTUAL INVENTORY SIMULATION that replays the
last 56 days of real demand under (a) the store's old naive rule and
(b) our forecast-driven policy, and compares cost & service level.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import config

Z_SCORE = NormalDist().inv_cdf(config.SERVICE_LEVEL)


# ---------------------------------------------------------------------------
# Core formulas (kept as small functions so they are easy to unit-test)
# ---------------------------------------------------------------------------
def safety_stock(sigma_daily, lead_time_days, z=Z_SCORE):
    return np.ceil(z * np.asarray(sigma_daily) * np.sqrt(lead_time_days))


def reorder_point(lead_time_demand, safety_stock_units):
    return np.ceil(np.asarray(lead_time_demand) + safety_stock_units)


def economic_order_quantity(annual_demand, unit_cost,
                            ordering_cost=config.ORDERING_COST,
                            holding_rate=config.HOLDING_COST_RATE):
    holding_cost = np.asarray(unit_cost) * holding_rate
    return np.ceil(np.sqrt(2 * np.asarray(annual_demand) * ordering_cost
                           / holding_cost))


def classify_stock(current, ss, rop, days_cover,
                   overstock_days=config.OVERSTOCK_DAYS):
    return np.select(
        [current <= ss, current <= rop, days_cover > overstock_days],
        ["CRITICAL", "REORDER", "OVERSTOCK"], default="HEALTHY")


def stockout_date(daily_forecast: pd.Series, current_stock: float):
    """First date on which cumulative forecast demand exceeds current stock."""
    cum = daily_forecast.cumsum()
    hit = cum[cum > current_stock]
    return hit.index[0].date() if len(hit) else None


# ---------------------------------------------------------------------------
# 1. Reorder recommendations for TODAY (using the 30-day future forecast)
# ---------------------------------------------------------------------------
def build_recommendations(clean: pd.DataFrame, forecast: pd.DataFrame,
                          error_stats: pd.DataFrame) -> pd.DataFrame:
    last_day = clean["date"].max()
    snapshot = clean[clean["date"] == last_day][
        ["store_id", "store_name", "product_id", "product_name", "category",
         "unit_cost", "lead_time_days", "closing_stock"]
    ].rename(columns={"closing_stock": "current_stock"})

    # Fallback uncertainty: std of last 28 days of sales
    recent_std = (clean[clean["date"] > last_day - pd.Timedelta(days=28)]
                  .groupby(["store_id", "product_id"])["units_sold"].std()
                  .rename("recent_sales_std").reset_index())

    rows = []
    for (s, p), fc in forecast.groupby(["store_id", "product_id"]):
        fc = fc.set_index("date")["forecast_units"].sort_index()
        info = snapshot[(snapshot.store_id == s) & (snapshot.product_id == p)].iloc[0]
        lt = int(info.lead_time_days)
        rows.append({
            "store_id": s, "product_id": p,
            "avg_daily_forecast": fc.mean(),
            "forecast_30d_units": fc.sum(),
            "lead_time_demand": fc.iloc[:lt].sum(),
            "expected_stockout_date": stockout_date(fc, info.current_stock),
        })
    rec = snapshot.merge(pd.DataFrame(rows), on=["store_id", "product_id"])
    rec = rec.merge(error_stats, on=["store_id", "product_id"], how="left")
    rec = rec.merge(recent_std, on=["store_id", "product_id"], how="left")
    rec["sigma_daily"] = rec["forecast_error_std"].fillna(rec["recent_sales_std"])

    rec["safety_stock"] = safety_stock(rec["sigma_daily"], rec["lead_time_days"])
    rec["reorder_point"] = reorder_point(rec["lead_time_demand"], rec["safety_stock"])
    rec["eoq"] = economic_order_quantity(rec["avg_daily_forecast"] * 365,
                                         rec["unit_cost"])
    rec["target_stock"] = np.ceil(
        rec["avg_daily_forecast"] * (rec["lead_time_days"]
                                     + config.REVIEW_PERIOD_DAYS)
        + rec["safety_stock"])
    rec["days_of_cover"] = (rec["current_stock"]
                            / rec["avg_daily_forecast"].clip(lower=0.01)).round(1)
    rec["stock_status"] = classify_stock(rec["current_stock"], rec["safety_stock"],
                                         rec["reorder_point"], rec["days_of_cover"])

    needs_order = rec["stock_status"].isin(["CRITICAL", "REORDER"])
    rec["recommended_order_qty"] = np.where(
        needs_order,
        np.maximum(rec["eoq"], rec["target_stock"] - rec["current_stock"]), 0
    ).astype(int)
    rec["order_value_rs"] = (rec["recommended_order_qty"] * rec["unit_cost"]).round(0)
    rec["excess_units"] = np.where(rec["stock_status"] == "OVERSTOCK",
                                   rec["current_stock"] - rec["target_stock"], 0
                                   ).astype(int)
    rec["excess_value_rs"] = (rec["excess_units"] * rec["unit_cost"]).round(0)

    priority = {"CRITICAL": 0, "REORDER": 1, "OVERSTOCK": 2, "HEALTHY": 3}
    rec = rec.sort_values(["stock_status", "days_of_cover"],
                          key=lambda c: c.map(priority) if c.name == "stock_status" else c)

    for col in ["avg_daily_forecast", "forecast_30d_units", "lead_time_demand",
                "sigma_daily"]:
        rec[col] = rec[col].round(1)
    cols = ["store_id", "store_name", "product_id", "product_name", "category",
            "current_stock", "avg_daily_forecast", "forecast_30d_units",
            "lead_time_days", "lead_time_demand", "sigma_daily", "safety_stock",
            "reorder_point", "eoq", "target_stock", "days_of_cover",
            "stock_status", "recommended_order_qty", "order_value_rs",
            "excess_units", "excess_value_rs", "expected_stockout_date"]
    return rec[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Virtual simulation: old naive policy vs forecast-driven policy
# ---------------------------------------------------------------------------
def simulate_policy(demand, start_stock, lead, reorder_level, order_qty,
                    target=None):
    """
    Day-by-day stock simulation for many series at once (numpy arrays).
    demand: (n_series, n_days). If `target` is given the order quantity is
    max(order_qty, target - inventory_position) ("order-up-to" logic).
    """
    n, days = demand.shape
    stock = start_stock.astype(float).copy()
    pipeline = np.zeros((n, days + int(lead.max()) + 1))
    stock_hist, sold_hist, orders = np.zeros((n, days)), np.zeros((n, days)), 0
    for d in range(days):
        stock += pipeline[:, d]
        sold = np.minimum(demand[:, d], stock)
        stock -= sold
        sold_hist[:, d], stock_hist[:, d] = sold, stock
        position = stock + pipeline[:, d + 1:].sum(axis=1)
        for i in np.where(position <= reorder_level)[0]:
            qty = order_qty[i]
            if target is not None:
                qty = max(qty, target[i] - position[i])
            pipeline[i, d + int(lead[i])] += qty
            orders += 1
    return stock_hist, sold_hist, orders


def run_policy_simulation(clean: pd.DataFrame, test_preds: pd.DataFrame,
                          products: pd.DataFrame, stores: pd.DataFrame):
    cutoff = clean["date"].max() - pd.Timedelta(days=config.TEST_DAYS)
    test = clean[clean["date"] > cutoff].sort_values(
        ["store_id", "product_id", "date"])
    keys = test[["store_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    n, days = len(keys), test["date"].nunique()

    # True customer demand = what was sold + what was lost due to stockouts
    demand = (test["units_sold"] + test["lost_sales_units"]).values.reshape(n, days)
    start_stock = test.groupby(["store_id", "product_id"])["opening_stock"].first().values
    info = keys.merge(products, on="product_id").merge(stores, on="store_id")
    lead = info["lead_time_days"].values
    cost = info["unit_cost"].values
    price = info["unit_price"].values

    # (a) OLD naive rule: reorder at 7 days of base sales, order 21 days
    base = (info["base_daily_demand"] * info["size_factor"]).values
    naive = simulate_policy(demand, start_stock, lead, np.ceil(base * 7),
                            np.ceil(base * 21))

    # (b) NEW forecast-driven policy. Uses only information available at the
    #     cutoff: the model's forecast for the period + recent sales variability
    pred = (test_preds.sort_values(["store_id", "product_id", "date"])
            ["prediction"].values.reshape(n, days))
    avg_fc = pred.mean(axis=1)
    hist = clean[(clean["date"] <= cutoff)
                 & (clean["date"] > cutoff - pd.Timedelta(days=28))]
    sigma = (hist.groupby(["store_id", "product_id"])["units_sold"].std()
             .reindex(pd.MultiIndex.from_frame(keys)).values)
    ss = safety_stock(sigma, lead)
    rop = reorder_point(avg_fc * lead, ss)
    eoq = economic_order_quantity(avg_fc * 365, cost)
    target = np.ceil(avg_fc * (lead + config.REVIEW_PERIOD_DAYS) + ss)
    smart = simulate_policy(demand, start_stock, lead, rop, eoq, target)

    def kpis(name, result):
        stock_hist, sold_hist, orders = result
        lost = demand - sold_hist
        avg_inv_value = (stock_hist.mean(axis=1) * cost).sum()
        holding = avg_inv_value * config.HOLDING_COST_RATE * days / 365
        lost_margin = (lost.sum(axis=1) * (price - cost)).sum()
        return {"Policy": name,
                "Fill rate %": sold_hist.sum() / demand.sum() * 100,
                "Stockout days %": (stock_hist <= 0).mean() * 100,
                "Lost sales (units)": lost.sum(),
                "Lost revenue (Rs)": (lost.sum(axis=1) * price).sum(),
                "Avg inventory value (Rs)": avg_inv_value,
                "Purchase orders": orders,
                "Holding cost (Rs)": holding,
                "Ordering cost (Rs)": orders * config.ORDERING_COST,
                "Lost margin (Rs)": lost_margin,
                "Total cost (Rs)": holding + orders * config.ORDERING_COST
                                   + lost_margin}

    comparison = pd.DataFrame([kpis("Old naive rule", naive),
                               kpis("Forecast-driven (ROP + EOQ)", smart)]
                              ).set_index("Policy").round(1)

    # Keep one example series for the stock-level chart (highest revenue item)
    example_idx = int(np.argmax((demand.sum(axis=1) * price)))
    dates = sorted(test["date"].unique())
    example = pd.DataFrame({
        "date": dates,
        "naive_stock": naive[0][example_idx],
        "smart_stock": smart[0][example_idx],
        "demand": demand[example_idx],
    })
    example.attrs = {"label": f"{info.loc[example_idx, 'product_name']} @ "
                              f"{info.loc[example_idx, 'store_name']}",
                     "rop": rop[example_idx], "ss": ss[example_idx]}
    return comparison, example


def run_inventory_optimization(clean, forecast, error_stats, test_preds):
    config.make_dirs()
    products = pd.read_csv(config.PRODUCTS_FILE)
    stores = pd.read_csv(config.STORES_FILE)

    rec = build_recommendations(clean, forecast, error_stats)
    rec.to_csv(config.OUTPUTS_DIR / "inventory_recommendations.csv", index=False)
    alerts = rec[rec["stock_status"].isin(["CRITICAL", "REORDER"])]
    alerts.to_csv(config.OUTPUTS_DIR / "reorder_alerts.csv", index=False)

    comparison, example = run_policy_simulation(clean, test_preds, products, stores)
    comparison.to_csv(config.OUTPUTS_DIR / "policy_comparison.csv")

    counts = rec["stock_status"].value_counts().to_dict()
    print(f"[inventory] Z={Z_SCORE:.3f} for {config.SERVICE_LEVEL:.0%} service level"
          f" | status counts: {counts}")
    print(f"[inventory] {len(alerts)} reorder alerts | total order value "
          f"Rs. {alerts['order_value_rs'].sum():,.0f}")
    print("[simulation] Old vs new policy on last "
          f"{config.TEST_DAYS} days of real demand:")
    print(comparison[["Fill rate %", "Stockout days %", "Avg inventory value (Rs)",
                      "Total cost (Rs)"]].to_string())
    return rec, alerts, comparison, example
