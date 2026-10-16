"""
data_preprocessing.py
---------------------
Turns the messy raw POS export into a clean, analysis-ready table.

Steps:
 1. Load raw sales + master data
 2. Parse dates and fix data types
 3. Standardise text (category names)
 4. Remove exact duplicate rows
 5. Fix negative quantities (returns booked as sales)
 6. Cap extreme outliers (IQR-style rule per store-product)
 7. Fill missing values (sales -> interpolation, price -> product price)
 8. Join product/store details and compute revenue & profit
A data-quality report is printed and saved so you can prove what was fixed.
"""
import numpy as np
import pandas as pd

from src import config


def load_raw_data():
    sales = pd.read_csv(config.RAW_SALES_FILE)
    products = pd.read_csv(config.PRODUCTS_FILE)
    stores = pd.read_csv(config.STORES_FILE)
    return sales, products, stores


def clean_sales(sales: pd.DataFrame, products: pd.DataFrame,
                stores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": len(sales),
              "missing_units_sold": int(sales["units_sold"].isna().sum()),
              "missing_unit_price": int(sales["unit_price"].isna().sum())}

    df = sales.copy()

    # 1. Dates: string -> datetime. errors="coerce" turns bad dates into NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["bad_dates_dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # 2. Text cleanup: " GROCERIES " -> "Groceries" (use master data as truth)
    df["category"] = df["category"].str.strip()
    df = df.drop(columns=["category"]).merge(
        products[["product_id", "category"]], on="product_id", how="left")

    # 3. Duplicates: the same store/product/date must appear only once
    before = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"])
    report["duplicates_removed"] = before - len(df)

    # 4. Negative sales are data-entry errors -> treat as missing
    neg = df["units_sold"] < 0
    report["negative_values_fixed"] = int(neg.sum())
    df.loc[neg, "units_sold"] = np.nan

    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    grp = df.groupby(["store_id", "product_id"])["units_sold"]

    # 5. Outliers: value above median + 6 * IQR of its own series -> missing.
    #    (Festival peaks are ~2x normal, typos are ~10x, so this is safe.)
    q1, q3 = grp.transform(lambda s: s.quantile(0.25)), grp.transform(
        lambda s: s.quantile(0.75))
    med = grp.transform("median")
    upper = med + 6 * (q3 - q1).clip(lower=1)
    out = df["units_sold"] > upper
    report["outliers_fixed"] = int(out.sum())
    df.loc[out, "units_sold"] = np.nan

    # 6. Missing sales: linear interpolation within each series (time-aware)
    report["units_sold_imputed"] = int(df["units_sold"].isna().sum())
    df["units_sold"] = (df.groupby(["store_id", "product_id"])["units_sold"]
                          .transform(lambda s: s.interpolate(limit_direction="both"))
                          .round().astype(int))

    # 7. Missing price: use the product's list price from master data
    df = df.merge(products[["product_id", "product_name", "unit_price",
                            "unit_cost", "lead_time_days"]]
                  .rename(columns={"unit_price": "list_price"}),
                  on="product_id", how="left")
    expected_price = df["list_price"].where(df["promotion"] == 0,
                                            df["list_price"] * 0.9)
    df["unit_price"] = df["unit_price"].fillna(expected_price)
    df = df.drop(columns=["list_price"])

    # 8. Store details + business metrics
    df = df.merge(stores[["store_id", "store_name", "city", "region"]],
                  on="store_id", how="left")
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    df["gross_profit"] = (df["units_sold"]
                          * (df["unit_price"] - df["unit_cost"])).round(2)

    report["clean_rows"] = len(df)
    report["remaining_missing_values"] = int(df.isna().sum().sum())

    cols = ["date", "store_id", "store_name", "city", "region", "product_id",
            "product_name", "category", "unit_price", "unit_cost", "promotion",
            "festival_period", "units_sold", "revenue", "gross_profit",
            "opening_stock", "units_received", "closing_stock",
            "stockout_flag", "lost_sales_units", "lead_time_days"]
    return df[cols], report


def run_preprocessing() -> pd.DataFrame:
    config.make_dirs()
    sales, products, stores = load_raw_data()
    clean, report = clean_sales(sales, products, stores)
    clean.to_csv(config.CLEAN_SALES_FILE, index=False)

    report_df = pd.DataFrame(list(report.items()), columns=["check", "value"])
    report_df.to_csv(config.OUTPUTS_DIR / "data_quality_report.csv", index=False)

    print("[preprocessing] Data quality report:")
    for k, v in report.items():
        print(f"    {k:<28}: {v:,}")
    return clean


if __name__ == "__main__":
    run_preprocessing()
