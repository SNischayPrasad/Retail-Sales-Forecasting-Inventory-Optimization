"""
data_generator.py
-----------------
VIRTUAL SIMULATION of a retail chain.

We do not have access to a real retailer's systems, so we create a realistic
dataset ourselves. Every number is produced by explicit business rules, which
means we KNOW the patterns the model should discover:

    demand = base demand
             x store size
             x long-term growth trend
             x weekly pattern (weekends are busier)
             x yearly seasonality (summer / winter products)
             x festival boost (Diwali, Holi, Independence Day sale ...)
             x salary-week effect (first days of month)
             x promotion uplift
             + random noise (Poisson)

On top of demand we also simulate the store's CURRENT (naive) stock policy:
"when stock looks low, re-order 3 weeks of average sales". Because this rule
ignores trend and festivals, it creates realistic stockouts and overstock -
exactly the problem our system will fix.

Finally we inject typical data quality problems (missing values, duplicates,
negative quantities, outliers) so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Master data: stores and products
# ---------------------------------------------------------------------------
STORES = [
    # store_id, store_name, city, region, size multiplier
    ("S01", "SuperMart Andheri", "Mumbai", "West", 1.30),
    ("S02", "SuperMart Koramangala", "Bengaluru", "South", 1.00),
    ("S03", "SuperMart Rohini", "Delhi", "North", 0.80),
]

PRODUCTS = [
    # product_id, product_name, category, unit_price (Rs), base daily demand
    ("P001", "Basmati Rice 5kg", "Groceries", 650, 18),
    ("P002", "Sunflower Oil 1L", "Groceries", 160, 30),
    ("P003", "Toor Dal 1kg", "Groceries", 170, 22),
    ("P004", "Wheat Atta 10kg", "Groceries", 450, 15),
    ("P005", "Toned Milk 1L", "Dairy & Bakery", 60, 80),
    ("P006", "Brown Bread 400g", "Dairy & Bakery", 50, 45),
    ("P007", "Paneer 200g", "Dairy & Bakery", 90, 25),
    ("P008", "Curd 400g", "Dairy & Bakery", 45, 35),
    ("P009", "Shampoo 340ml", "Personal Care", 280, 10),
    ("P010", "Toothpaste 150g", "Personal Care", 110, 20),
    ("P011", "Bath Soap Pack of 4", "Personal Care", 180, 16),
    ("P012", "Sunscreen SPF50", "Personal Care", 450, 5),
    ("P013", "Detergent Powder 2kg", "Home Care", 380, 12),
    ("P014", "Dishwash Liquid 750ml", "Home Care", 190, 14),
    ("P015", "Floor Cleaner 1L", "Home Care", 210, 9),
    ("P016", "Room Freshener", "Home Care", 230, 6),
    ("P017", "LED Bulb 9W", "Electronics", 120, 8),
    ("P018", "Bluetooth Earbuds", "Electronics", 1500, 2),
    ("P019", "Electric Kettle", "Electronics", 1200, 2),
    ("P020", "Table Fan", "Electronics", 2200, 1.5),
]

# Yearly growth trend per category (e.g. 0.15 = +15% per year)
CATEGORY_GROWTH = {"Groceries": 0.08, "Dairy & Bakery": 0.06,
                   "Personal Care": 0.12, "Home Care": 0.10,
                   "Electronics": 0.25}

# Products with strong summer seasonality (peak around May)
SUMMER_PRODUCTS = {"P012": 0.6, "P020": 0.9, "P008": 0.25, "P005": 0.05}
# Products with winter seasonality (peak around December)
WINTER_PRODUCTS = {"P019": 0.5, "P006": 0.1}

# Festivals / big sale days: (date, days of effect before, days after, boost)
FESTIVALS = [
    ("2023-01-26", 2, 0, 1.25),  # Republic Day sale
    ("2023-03-08", 4, 0, 1.35),  # Holi
    ("2023-08-15", 2, 0, 1.30),  # Independence Day sale
    ("2023-11-12", 10, 1, 1.80), # Diwali
    ("2023-12-31", 3, 0, 1.25),  # New Year
    ("2024-01-26", 2, 0, 1.25),
    ("2024-03-25", 4, 0, 1.35),
    ("2024-08-15", 2, 0, 1.30),
    ("2024-11-01", 10, 1, 1.80),
    ("2024-12-31", 3, 0, 1.25),
    # 2025 dates are used only for the future forecast calendar
    ("2025-01-26", 2, 0, 1.25),
]
# Diwali boosts electronics and home care more than other categories
FESTIVAL_CATEGORY_EXTRA = {"Electronics": 1.4, "Home Care": 1.2}


def festival_multiplier(dates: pd.DatetimeIndex) -> pd.Series:
    """Return a multiplier per date (1.0 on normal days, >1 near festivals)."""
    mult = pd.Series(1.0, index=dates)
    for day, before, after, boost in FESTIVALS:
        day = pd.Timestamp(day)
        window = pd.date_range(day - pd.Timedelta(days=before),
                               day + pd.Timedelta(days=after))
        window = window.intersection(dates)
        mult.loc[window] = np.maximum(mult.loc[window], boost)
    return mult


def is_festival_period(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 if the date falls inside a festival window, else 0."""
    return (festival_multiplier(dates).values > 1.0).astype(int)


def generate_master_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    stores = pd.DataFrame(STORES, columns=["store_id", "store_name", "city",
                                           "region", "size_factor"])
    products = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name",
                                               "category", "unit_price",
                                               "base_daily_demand"])
    # Cost price = 70-85% of selling price (gross margin 15-30%)
    rng = np.random.default_rng(config.RANDOM_SEED)
    products["unit_cost"] = (products["unit_price"]
                             * rng.uniform(0.70, 0.85, len(products))).round(2)
    products["lead_time_days"] = products["category"].map(
        config.LEAD_TIME_BY_CATEGORY)
    return stores, products


def simulate_demand(stores, products, dates, rng) -> pd.DataFrame:
    """Create the TRUE customer demand for every store x product x day."""
    t_years = (dates - dates[0]).days.values / 365.0
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values  # Monday=0 ... Sunday=6

    weekly = np.select([dow == 5, dow == 6, dow == 4], [1.30, 1.25, 1.10], 1.0)
    salary_week = np.where(dates.day.values <= 5, 1.10, 1.0)
    fest = festival_multiplier(dates).values

    frames = []
    for _, s in stores.iterrows():
        for _, p in products.iterrows():
            trend = 1 + CATEGORY_GROWTH[p.category] * t_years
            seasonal = np.ones(len(dates))
            if p.product_id in SUMMER_PRODUCTS:   # peak ~ day 135 (mid-May)
                amp = SUMMER_PRODUCTS[p.product_id]
                seasonal = 1 + amp * np.cos(2 * np.pi * (doy - 135) / 365)
            elif p.product_id in WINTER_PRODUCTS:  # peak ~ day 350 (mid-Dec)
                amp = WINTER_PRODUCTS[p.product_id]
                seasonal = 1 + amp * np.cos(2 * np.pi * (doy - 350) / 365)
            seasonal = np.clip(seasonal, 0.15, None)

            fest_p = np.where(fest > 1,
                              1 + (fest - 1) * FESTIVAL_CATEGORY_EXTRA.get(
                                  p.category, 1.0), 1.0)

            # Promotions: ~6 random 3-day promo blocks per year
            promo = np.zeros(len(dates), dtype=int)
            n_blocks = int(len(dates) / 365 * 6)
            for start in rng.choice(len(dates) - 3, n_blocks, replace=False):
                promo[start:start + 3] = 1
            promo_uplift = np.where(promo == 1, rng.uniform(1.3, 1.6), 1.0)
            price = np.where(promo == 1, p.unit_price * 0.90, p.unit_price)

            expected = (p.base_daily_demand * s.size_factor * trend * weekly
                        * seasonal * fest_p * salary_week * promo_uplift)
            # Gamma-Poisson noise = realistic "over-dispersed" retail demand
            noisy_rate = expected * rng.gamma(shape=8, scale=1 / 8,
                                              size=len(dates))
            demand = rng.poisson(noisy_rate)

            frames.append(pd.DataFrame({
                "date": dates, "store_id": s.store_id,
                "product_id": p.product_id, "category": p.category,
                "unit_price": price.round(2), "promotion": promo,
                "festival_period": (fest > 1).astype(int),
                "demand": demand,
                "lead_time_days": p.lead_time_days,
                "base_rate": p.base_daily_demand * s.size_factor,
            }))
    return pd.concat(frames, ignore_index=True)


def simulate_naive_inventory(df: pd.DataFrame, rng) -> pd.DataFrame:
    """
    Simulate the store's CURRENT rule-of-thumb stock policy, day by day:
      - reorder when (stock + on order) < 7 days of average sales
      - order quantity = 21 days of average sales
    'Average sales' is based on the initial base rate only, so it ignores
    growth and festivals -> realistic stockouts and overstock.
    """
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    series_keys = df[["store_id", "product_id"]].drop_duplicates()
    n_series, n_days = len(series_keys), df["date"].nunique()

    demand = df["demand"].values.reshape(n_series, n_days)
    lead = df["lead_time_days"].values.reshape(n_series, n_days)[:, 0]
    base = df["base_rate"].values.reshape(n_series, n_days)[:, 0]

    reorder_level = np.ceil(base * 7)
    order_qty = np.ceil(base * 21)

    stock = np.ceil(base * 14)            # start with 2 weeks of stock
    pipeline = np.zeros((n_series, n_days + 30))  # arrivals scheduled per day

    opening = np.zeros((n_series, n_days))
    received = np.zeros((n_series, n_days))
    sold = np.zeros((n_series, n_days))
    closing = np.zeros((n_series, n_days))

    for d in range(n_days):
        received[:, d] = pipeline[:, d]
        stock = stock + pipeline[:, d]
        opening[:, d] = stock
        sold[:, d] = np.minimum(demand[:, d], stock)
        stock = stock - sold[:, d]
        closing[:, d] = stock

        on_order = pipeline[:, d + 1:].sum(axis=1)
        need = (stock + on_order) < reorder_level
        # Supplier sometimes delivers 1-2 days late
        delay = rng.integers(0, 3, n_series)
        for i in np.where(need)[0]:
            pipeline[i, d + lead[i] + delay[i]] += order_qty[i]

    df["opening_stock"] = opening.ravel().astype(int)
    df["units_received"] = received.ravel().astype(int)
    df["units_sold"] = sold.ravel().astype(int)
    df["closing_stock"] = closing.ravel().astype(int)
    df["stockout_flag"] = (df["units_sold"] < df["demand"]).astype(int)
    df["lost_sales_units"] = df["demand"] - df["units_sold"]
    return df


def inject_data_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Add the kind of mess found in real POS (point-of-sale) exports."""
    df = df.copy()
    df["units_sold"] = df["units_sold"].astype(float)
    n = len(df)

    # 1% missing sales values (POS sync failure)
    df.loc[rng.choice(n, int(n * 0.01), replace=False), "units_sold"] = np.nan
    # 0.5% missing price values
    df.loc[rng.choice(n, int(n * 0.005), replace=False), "unit_price"] = np.nan
    # 0.1% negative quantities (returns wrongly booked as sales)
    neg_idx = rng.choice(n, int(n * 0.001), replace=False)
    df.loc[neg_idx, "units_sold"] = -df.loc[neg_idx, "units_sold"].abs() - 1
    # 0.05% extreme outliers (typo: extra zero typed)
    out_idx = rng.choice(n, int(n * 0.0005), replace=False)
    df.loc[out_idx, "units_sold"] = df.loc[out_idx, "units_sold"] * 10 + 50
    # Inconsistent text formatting in category names
    txt_idx = rng.choice(n, int(n * 0.01), replace=False)
    df.loc[txt_idx, "category"] = df.loc[txt_idx, "category"].str.upper() + " "
    # 0.3% duplicated rows (same record exported twice)
    dupes = df.sample(frac=0.003, random_state=config.RANDOM_SEED)
    df = pd.concat([df, dupes], ignore_index=True)
    # Shuffle so the file is not perfectly ordered
    return df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)


def generate_dataset(save: bool = True) -> pd.DataFrame:
    """Run the full simulation and save raw CSV files."""
    config.make_dirs()
    rng = np.random.default_rng(config.RANDOM_SEED)
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="D")

    stores, products = generate_master_data()
    df = simulate_demand(stores, products, dates, rng)
    df = simulate_naive_inventory(df, rng)

    # The raw file looks like a POS export: true demand is NOT visible to the
    # business, but we keep 'lost_sales_units' as a simulation-only column so
    # we can later measure how much money stockouts cost.
    raw = df[["date", "store_id", "product_id", "category", "unit_price",
              "promotion", "festival_period", "units_sold", "opening_stock",
              "units_received", "closing_stock", "stockout_flag",
              "lost_sales_units"]].copy()
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    raw = inject_data_quality_issues(raw, rng)

    if save:
        raw.to_csv(config.RAW_SALES_FILE, index=False)
        stores.to_csv(config.STORES_FILE, index=False)
        products.to_csv(config.PRODUCTS_FILE, index=False)
        print(f"[simulation] {len(raw):,} rows -> {config.RAW_SALES_FILE.name}")
        print(f"[simulation] {len(stores)} stores, {len(products)} products, "
              f"{len(dates)} days ({config.START_DATE} to {config.END_DATE})")
    return raw


if __name__ == "__main__":
    generate_dataset()
