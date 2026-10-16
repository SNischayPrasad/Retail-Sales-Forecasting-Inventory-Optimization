"""
streamlit_app.py
----------------
Interactive dashboard for store / category managers.

Run from the project root (after `python main.py`):
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

st.set_page_config(page_title="Retail Forecast & Inventory", layout="wide")


@st.cache_data
def load():
    clean = pd.read_csv(config.CLEAN_SALES_FILE, parse_dates=["date"])
    forecast = pd.read_csv(config.OUTPUTS_DIR / "forecast_next_30_days.csv",
                           parse_dates=["date"])
    rec = pd.read_csv(config.OUTPUTS_DIR / "inventory_recommendations.csv")
    metrics = pd.read_csv(config.OUTPUTS_DIR / "model_metrics.csv", index_col=0)
    policy = pd.read_csv(config.OUTPUTS_DIR / "policy_comparison.csv", index_col=0)
    return clean, forecast, rec, metrics, policy


try:
    clean, forecast, rec, metrics, policy = load()
except FileNotFoundError:
    st.error("Output files not found. Run `python main.py` first.")
    st.stop()

# ----------------------------- Sidebar filters -----------------------------
st.sidebar.header("Filters")
stores = st.sidebar.multiselect("Store", sorted(rec["store_name"].unique()),
                                default=sorted(rec["store_name"].unique()))
cats = st.sidebar.multiselect("Category", sorted(rec["category"].unique()),
                              default=sorted(rec["category"].unique()))
store_ids = rec.loc[rec["store_name"].isin(stores), "store_id"].unique()

c = clean[clean["store_id"].isin(store_ids) & clean["category"].isin(cats)]
f = forecast[forecast["store_id"].isin(store_ids) & forecast["category"].isin(cats)]
r = rec[rec["store_id"].isin(store_ids) & rec["category"].isin(cats)]

# ----------------------------- KPI header ----------------------------------
st.title("Retail Sales Forecasting & Inventory Optimization")
st.caption(f"Data up to {clean['date'].max().date()} | virtual simulation of "
           f"{clean['store_id'].nunique()} stores x {clean['product_id'].nunique()} "
           "products")

best = metrics["WAPE_%"].idxmin()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue (2 yrs)", f"Rs. {c['revenue'].sum()/1e7:,.2f} Cr")
k2.metric("Forecast accuracy", f"{metrics.loc[best, 'Accuracy_%']:.1f}%", best)
k3.metric("Next 30 days demand", f"{f['forecast_units'].sum():,.0f} units")
k4.metric("Items to reorder", int(r["stock_status"].isin(["CRITICAL", "REORDER"]).sum()))
k5.metric("Overstock value", f"Rs. {r['excess_value_rs'].sum():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Sales trend", "Forecast", "Inventory actions",
                                  "Model & policy"])

with tab1:
    weekly = c.set_index("date").groupby("category")["units_sold"] \
              .resample("W").sum().unstack(0)
    st.subheader("Weekly units sold by category")
    st.line_chart(weekly)
    col1, col2 = st.columns(2)
    col1.subheader("Revenue by category")
    col1.bar_chart(c.groupby("category")["revenue"].sum())
    col2.subheader("Revenue by store")
    col2.bar_chart(c.groupby("store_name")["revenue"].sum())

with tab2:
    st.subheader("Last 60 days + next 30 days forecast")
    product = st.selectbox("Product", sorted(r["product_name"].unique()))
    pid = rec.loc[rec["product_name"] == product, "product_id"].iloc[0]
    hist = (c[(c["product_id"] == pid)
              & (c["date"] > c["date"].max() - pd.Timedelta(days=60))]
            .groupby("date")["units_sold"].sum().rename("Actual"))
    fut = f[f["product_id"] == pid].groupby("date")["forecast_units"].sum() \
            .rename("Forecast")
    st.line_chart(pd.concat([hist, fut], axis=1))
    st.dataframe(f[f["product_id"] == pid].pivot_table(
        index="date", columns="store_id", values="forecast_units").round(1),
        width="stretch")

with tab3:
    st.subheader("Reorder alerts")
    alerts = r[r["stock_status"].isin(["CRITICAL", "REORDER"])]
    if alerts.empty:
        st.success("No items need reordering for this selection.")
    else:
        st.dataframe(alerts[["store_name", "product_name", "current_stock",
                             "safety_stock", "reorder_point",
                             "recommended_order_qty", "order_value_rs",
                             "expected_stockout_date", "stock_status"]],
                     width="stretch", hide_index=True)
    st.subheader("All recommendations")
    status = st.multiselect("Status", ["CRITICAL", "REORDER", "OVERSTOCK", "HEALTHY"],
                            default=["CRITICAL", "REORDER", "OVERSTOCK", "HEALTHY"])
    st.dataframe(r[r["stock_status"].isin(status)], width="stretch",
                 hide_index=True)
    st.download_button("Download recommendations CSV",
                       r.to_csv(index=False), "inventory_recommendations.csv")

with tab4:
    st.subheader("Forecast model comparison (56-day recursive test)")
    st.dataframe(metrics, width="stretch")
    st.subheader("Old naive rule vs forecast-driven policy (simulation)")
    st.dataframe(policy.T, width="stretch")
    img = config.IMAGES_DIR / "15_inventory_simulation.png"
    if img.exists():
        st.image(str(img))
