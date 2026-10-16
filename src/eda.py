"""
eda.py
------
Exploratory Data Analysis: understand the business BEFORE building models.

Questions answered (each saves one chart to images/):
  01 How do total daily sales move over time?         (trend + festivals)
  02 How does monthly revenue grow?                   (growth)
  03 Which categories make the most money?            (category mix)
  04 Which weekday is busiest?                        (weekly seasonality)
  05 How do stores / regions compare?                 (store performance)
  06 Do promotions really increase sales?             (promo impact)
  07 Which categories peak in which month?            (seasonality heatmap)
  08 How often do we run out of stock?                (stockout analysis)
"""
import matplotlib

matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_daily_sales_trend(df):
    daily = df.groupby("date")["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily.values, lw=0.7, alpha=0.6, label="Daily units")
    ax.plot(daily.index, daily.rolling(28).mean(), lw=2.2, color="crimson",
            label="28-day moving average")
    ax.set(title="Total Daily Units Sold - All Stores",
           xlabel="Date", ylabel="Units sold")
    ax.legend()
    _save(fig, "01_daily_sales_trend.png")


def plot_monthly_revenue(df):
    monthly = df.set_index("date")["revenue"].resample("MS").sum() / 1e5
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(monthly.index.strftime("%b-%y"), monthly.values, color="#2a9d8f")
    ax.set(title="Monthly Revenue (Rs. Lakh)", xlabel="Month",
           ylabel="Revenue (Rs. Lakh)")
    ax.tick_params(axis="x", rotation=60)
    _save(fig, "02_monthly_revenue.png")


def plot_category_sales(df):
    cat = (df.groupby("category")[["revenue", "gross_profit"]].sum() / 1e5
           ).sort_values("revenue")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cat.plot.barh(ax=ax, color=["#264653", "#e9c46a"])
    ax.set(title="Revenue & Gross Profit by Category (Rs. Lakh)",
           xlabel="Rs. Lakh", ylabel="")
    _save(fig, "03_category_sales.png")


def plot_weekday_pattern(df):
    daily = df.groupby("date")["units_sold"].sum().reset_index()
    daily["weekday"] = daily["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=daily, x="weekday", y="units_sold", order=order,
                ax=ax, color="#457b9d", errorbar=None)
    ax.set(title="Average Daily Units by Weekday", xlabel="", ylabel="Units")
    _save(fig, "04_weekday_pattern.png")


def plot_store_comparison(df):
    store = df.groupby(["store_name", "region"])["revenue"].sum().reset_index()
    store["label"] = store["store_name"] + "\n(" + store["region"] + ")"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(store["label"], store["revenue"] / 1e5, color="#e76f51")
    ax.set(title="Total Revenue by Store (Rs. Lakh)", ylabel="Rs. Lakh")
    _save(fig, "05_store_region_sales.png")


def plot_promo_impact(df):
    promo = (df.groupby(["category", "promotion"])["units_sold"].mean()
               .unstack().rename(columns={0: "No promotion", 1: "Promotion"}))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    promo.plot.bar(ax=ax, color=["#adb5bd", "#2a9d8f"])
    ax.set(title="Average Daily Units: Promotion vs No Promotion",
           xlabel="", ylabel="Avg units / product / store / day")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "06_promo_impact.png")
    return promo


def plot_seasonality_heatmap(df):
    tmp = df.assign(month=df["date"].dt.month)
    pivot = tmp.groupby(["category", "month"])["units_sold"].sum().unstack()
    pivot = pivot.div(pivot.mean(axis=1), axis=0)  # index vs category average
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, cmap="RdYlGn", center=1, annot=True, fmt=".2f", ax=ax)
    ax.set(title="Seasonality Index (1.0 = average month)", xlabel="Month",
           ylabel="")
    _save(fig, "07_seasonality_heatmap.png")


def plot_stockouts(df):
    so = (df.groupby("category")
            .agg(stockout_rate=("stockout_flag", "mean"),
                 lost_units=("lost_sales_units", "sum")))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (so["stockout_rate"] * 100).sort_values().plot.barh(ax=ax, color="#d62828")
    ax.set(title="Stockout Days (%) Under the Current Naive Policy",
           xlabel="% of store-product-days out of stock", ylabel="")
    _save(fig, "08_stockout_rate_by_category.png")
    return so


def run_eda(df: pd.DataFrame | None = None) -> dict:
    config.make_dirs()
    if df is None:
        df = pd.read_csv(config.CLEAN_SALES_FILE, parse_dates=["date"])

    plot_daily_sales_trend(df)
    plot_monthly_revenue(df)
    plot_category_sales(df)
    plot_weekday_pattern(df)
    plot_store_comparison(df)
    promo = plot_promo_impact(df)
    plot_seasonality_heatmap(df)
    so = plot_stockouts(df)

    # Key numbers used later in the business insights report
    yearly = df.groupby(df["date"].dt.year)["revenue"].sum()
    lost_value = (df["lost_sales_units"] * df["unit_price"]).sum()
    summary = {
        "total_revenue": df["revenue"].sum(),
        "total_profit": df["gross_profit"].sum(),
        "total_units": int(df["units_sold"].sum()),
        "yoy_growth_pct": (yearly.iloc[-1] / yearly.iloc[0] - 1) * 100,
        "top_category": df.groupby("category")["revenue"].sum().idxmax(),
        "top_store": df.groupby("store_name")["revenue"].sum().idxmax(),
        "busiest_weekday": df.groupby(df["date"].dt.day_name())["units_sold"]
                             .sum().idxmax(),
        "promo_uplift_pct": ((promo["Promotion"] / promo["No promotion"]).mean()
                             - 1) * 100,
        "stockout_rate_pct": df["stockout_flag"].mean() * 100,
        "lost_sales_value": lost_value,
    }
    pd.Series(summary).to_csv(config.OUTPUTS_DIR / "eda_summary.csv",
                              header=["value"])
    print(f"[eda] 8 charts saved to {config.IMAGES_DIR.name}/ | "
          f"YoY growth {summary['yoy_growth_pct']:.1f}% | "
          f"stockout rate {summary['stockout_rate_pct']:.1f}%")
    return summary


if __name__ == "__main__":
    run_eda()
