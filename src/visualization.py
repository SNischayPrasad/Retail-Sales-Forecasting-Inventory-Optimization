"""
visualization.py
----------------
Charts for the modelling + inventory part of the project (images 09-16)
and an auto-generated business insights report (reports/business_insights.md).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="deep")
STATUS_COLORS = {"CRITICAL": "#d62828", "REORDER": "#f77f00",
                 "OVERSTOCK": "#457b9d", "HEALTHY": "#2a9d8f"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / name, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics["WAPE_%"].sort_values().plot.barh(ax=axes[0], color="#264653")
    axes[0].set(title="WAPE % (lower is better)", xlabel="%", ylabel="")
    metrics["MAE"].sort_values().plot.barh(ax=axes[1], color="#e9c46a")
    axes[1].set(title="MAE in units (lower is better)", xlabel="units", ylabel="")
    fig.suptitle("Forecast Model Comparison - 56-day Recursive Test",
                 fontweight="bold")
    _save(fig, "09_model_comparison.png")


def plot_actual_vs_predicted(test_pred: pd.DataFrame, model_name: str):
    daily = test_pred.groupby("date")[[config.TARGET, "prediction"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(daily.index, daily[config.TARGET], marker="o", ms=3,
                 label="Actual")
    axes[0].plot(daily.index, daily["prediction"], marker="o", ms=3,
                 label=f"Predicted ({model_name})")
    axes[0].set(title="Actual vs Predicted - Total Daily Units (Test Period)",
                xlabel="Date", ylabel="Units")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    by_cat = test_pred.groupby("category")[[config.TARGET, "prediction"]].sum()
    by_cat.columns = ["Actual", "Predicted"]
    by_cat.plot.bar(ax=axes[1], color=["#264653", "#e76f51"])
    axes[1].set(title="Test Period Units by Category", xlabel="", ylabel="Units")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, "10_actual_vs_predicted.png")


def plot_future_forecast(clean: pd.DataFrame, forecast: pd.DataFrame):
    hist = clean[clean["date"] > clean["date"].max() - pd.Timedelta(days=90)]
    hist_cat = hist.groupby(["date", "category"])["units_sold"].sum().unstack()
    fc_cat = forecast.groupby(["date", "category"])["forecast_units"].sum().unstack()
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cat in enumerate(hist_cat.columns):
        color = sns.color_palette("deep")[i]
        ax.plot(hist_cat.index, hist_cat[cat], color=color, lw=1.2, label=cat)
        ax.plot(fc_cat.index, fc_cat[cat], color=color, lw=2, ls="--")
    ax.axvline(clean["date"].max(), color="black", ls=":", lw=1)
    ax.text(clean["date"].max(), ax.get_ylim()[1] * 0.95, "  forecast ->",
            fontsize=10)
    ax.set(title="Next 30 Days Forecast by Category (solid = history, "
                 "dashed = forecast)", xlabel="Date", ylabel="Units per day")
    ax.legend(ncol=5, loc="upper left", fontsize=8)
    _save(fig, "11_forecast_next_30_days.png")


def plot_feature_importance(importance: pd.Series | None):
    if importance is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.head(15).sort_values().plot.barh(ax=ax, color="#2a9d8f")
    ax.set(title="Top 15 Features - Random Forest Importance",
           xlabel="Importance", ylabel="")
    _save(fig, "12_feature_importance.png")


def plot_inventory_status(rec: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    counts = rec["stock_status"].value_counts().reindex(STATUS_COLORS).fillna(0)
    axes[0].bar(counts.index, counts.values,
                color=[STATUS_COLORS[s] for s in counts.index])
    axes[0].set(title="Store-Product Stock Status Today", ylabel="Count")
    pivot = (rec.pivot_table(index="category", columns="stock_status",
                             values="product_id", aggfunc="count", fill_value=0)
             .reindex(columns=[c for c in STATUS_COLORS if c in
                               rec["stock_status"].unique()]))
    pivot.plot.barh(stacked=True, ax=axes[1],
                    color=[STATUS_COLORS[c] for c in pivot.columns])
    axes[1].set(title="Stock Status by Category", xlabel="Count", ylabel="")
    _save(fig, "13_inventory_status.png")


def plot_reorder_table(alerts: pd.DataFrame):
    """Render the top reorder alerts as an image (easy to share on LinkedIn)."""
    cols = ["store_id", "product_name", "current_stock", "reorder_point",
            "safety_stock", "recommended_order_qty", "stock_status"]
    top = alerts[cols].head(12)
    top = top.assign(reorder_point=top["reorder_point"].astype(int),
                     safety_stock=top["safety_stock"].astype(int))
    fig, ax = plt.subplots(figsize=(13, 0.4 * len(top) + 0.6))
    ax.axis("off")
    table = ax.table(cellText=top.values,
                     colLabels=["Store", "Product", "Stock", "ROP", "Safety",
                                "Order Qty", "Status"],
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif c == len(cols) - 1:
            cell.set_facecolor(STATUS_COLORS.get(top.iloc[r - 1, -1], "white"))
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Reorder Alerts - Action Required Today", fontweight="bold")
    _save(fig, "14_reorder_alerts_table.png")


def plot_stock_simulation(example: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(example["date"], example["naive_stock"], label="Old naive rule",
            color="#adb5bd", lw=2)
    ax.plot(example["date"], example["smart_stock"],
            label="Forecast-driven policy", color="#2a9d8f", lw=2)
    ax.bar(example["date"], example["demand"], alpha=0.25, color="#e76f51",
           label="Daily demand")
    ax.axhline(example.attrs["rop"], color="#f77f00", ls="--",
               label=f"Reorder point ({example.attrs['rop']:.0f})")
    ax.axhline(example.attrs["ss"], color="#d62828", ls=":",
               label=f"Safety stock ({example.attrs['ss']:.0f})")
    ax.set(title=f"Inventory Simulation - {example.attrs['label']}",
           xlabel="Date", ylabel="Units on hand")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, "15_inventory_simulation.png")


def plot_policy_comparison(comparison: pd.DataFrame):
    metrics = ["Fill rate %", "Stockout days %", "Avg inventory value (Rs)",
               "Total cost (Rs)"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, m in zip(axes, metrics):
        ax.bar(["Old", "New"], comparison[m].values, color=["#adb5bd", "#2a9d8f"])
        ax.set_title(m, fontsize=10)
        for i, v in enumerate(comparison[m].values):
            ax.text(i, v, f"{v:,.1f}" if v < 1000 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Old Naive Rule vs Forecast-Driven Policy (simulated on "
                 f"last {config.TEST_DAYS} days)", fontweight="bold")
    _save(fig, "16_policy_comparison.png")


def write_business_report(eda: dict, metrics: pd.DataFrame, best: str,
                          rec: pd.DataFrame, comparison: pd.DataFrame):
    old, new = comparison.iloc[0], comparison.iloc[1]
    m = metrics.loc[best]
    base = metrics.loc["Seasonal Naive (baseline)"]
    status = rec["stock_status"].value_counts()
    top_alerts = rec[rec["stock_status"].isin(["CRITICAL", "REORDER"])].head(5)
    over = rec[rec["stock_status"] == "OVERSTOCK"]

    lines = [
        "# Business Insights Report",
        "",
        "_Auto-generated by `main.py` from the simulated retail dataset._",
        "",
        "## 1. Sales performance",
        f"- Total revenue (2 years): **Rs. {eda['total_revenue']/1e7:,.2f} crore**,"
        f" gross profit **Rs. {eda['total_profit']/1e7:,.2f} crore**",
        f"- Year-on-year revenue growth: **{eda['yoy_growth_pct']:.1f}%**",
        f"- Top category: **{eda['top_category']}** | top store: "
        f"**{eda['top_store']}** | busiest day: **{eda['busiest_weekday']}**",
        f"- Promotions lift unit sales by about **{eda['promo_uplift_pct']:.0f}%**",
        "",
        "## 2. The problem with the current stock policy",
        f"- Stockouts on **{eda['stockout_rate_pct']:.1f}%** of store-product-days",
        f"- Estimated revenue lost to stockouts: **Rs. "
        f"{eda['lost_sales_value']/1e5:,.1f} lakh**",
        "",
        "## 3. Forecast accuracy",
        f"- Best model: **{best}** with WAPE **{m['WAPE_%']:.1f}%** "
        f"(forecast accuracy {m['Accuracy_%']:.1f}%) at daily store-product level",
        f"- At weekly store-product level (how orders are planned) WAPE is "
        f"**{m['Weekly_WAPE_%']:.1f}%**",
        f"- Baseline (same-weekday average of last 4 weeks) WAPE: {base['WAPE_%']:.1f}% -> "
        f"error reduced by **{(1 - m['WAPE_%']/base['WAPE_%'])*100:.0f}%**",
        f"- Bias: {m['Bias_%']:+.1f}% (close to 0 = no systematic over/under "
        "forecasting)",
        "",
        "## 4. Inventory decisions for today",
        f"- CRITICAL: {status.get('CRITICAL', 0)} | REORDER: "
        f"{status.get('REORDER', 0)} | OVERSTOCK: {status.get('OVERSTOCK', 0)} | "
        f"HEALTHY: {status.get('HEALTHY', 0)}",
        f"- Purchase orders recommended worth **Rs. "
        f"{rec['order_value_rs'].sum():,.0f}**",
        f"- Capital locked in excess stock: **Rs. "
        f"{over['excess_value_rs'].sum():,.0f}** across {len(over)} items",
        "",
        "Top urgent items:",
        "",
        "| Store | Product | Stock | Reorder point | Order qty | Status |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in top_alerts.iterrows():
        lines.append(f"| {r.store_name} | {r.product_name} | {r.current_stock} | "
                     f"{r.reorder_point:.0f} | {r.recommended_order_qty} | "
                     f"{r.stock_status} |")
    lines += [
        "",
        f"## 5. Simulated impact (last {config.TEST_DAYS} days replayed)",
        "",
        "| KPI | Old naive rule | Forecast-driven |",
        "|---|---|---|",
        f"| Fill rate | {old['Fill rate %']:.1f}% | {new['Fill rate %']:.1f}% |",
        f"| Stockout days | {old['Stockout days %']:.1f}% | "
        f"{new['Stockout days %']:.1f}% |",
        f"| Lost revenue | Rs. {old['Lost revenue (Rs)']:,.0f} | "
        f"Rs. {new['Lost revenue (Rs)']:,.0f} |",
        f"| Avg inventory value | Rs. {old['Avg inventory value (Rs)']:,.0f} | "
        f"Rs. {new['Avg inventory value (Rs)']:,.0f} |",
        f"| Total cost (holding + ordering + lost margin) | "
        f"Rs. {old['Total cost (Rs)']:,.0f} | Rs. {new['Total cost (Rs)']:,.0f} |",
        "",
        f"Trade-off: the new policy holds "
        f"{(new['Avg inventory value (Rs)']/old['Avg inventory value (Rs)']-1)*100:.0f}% "
        "more average inventory, but cuts lost sales so much that total cost "
        f"falls by **{(1-new['Total cost (Rs)']/old['Total cost (Rs)'])*100:.0f}%**.",
        "",
        "## 6. Recommendations",
        "1. Replace the fixed rule-of-thumb with forecast-based reorder points "
        "reviewed weekly.",
        "2. Pre-build stock 2-3 weeks before Diwali for Electronics and Home Care.",
        "3. Clear OVERSTOCK items with targeted promotions instead of new orders.",
        "4. Track forecast accuracy (WAPE) weekly and retrain the model monthly.",
    ]
    path = config.REPORTS_DIR / "business_insights.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Business insights written to reports/{path.name}")


def run_visualization(clean, eda, fc_result, rec, alerts, comparison, example):
    config.make_dirs()
    best = fc_result["best_name"]
    plot_model_comparison(fc_result["metrics"])
    plot_actual_vs_predicted(fc_result["test_preds"][best], best)
    plot_future_forecast(clean, fc_result["forecast"])
    plot_feature_importance(fc_result["importance"])
    plot_inventory_status(rec)
    if len(alerts):
        plot_reorder_table(alerts)
    plot_stock_simulation(example)
    plot_policy_comparison(comparison)
    write_business_report(eda, fc_result["metrics"], best, rec, comparison)
    print("[visualization] 8 result charts saved to images/")
