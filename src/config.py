"""
config.py
---------
Central place for every setting used in the project.
Change values here instead of hunting through the code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folder paths (all relative to the project root, so the code works anywhere)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGES_DIR = ROOT_DIR / "images"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_SALES_FILE = DATA_RAW_DIR / "retail_sales_raw.csv"
PRODUCTS_FILE = DATA_RAW_DIR / "products_master.csv"
STORES_FILE = DATA_RAW_DIR / "stores_master.csv"
CLEAN_SALES_FILE = DATA_PROCESSED_DIR / "retail_sales_clean.csv"
FEATURES_FILE = DATA_PROCESSED_DIR / "retail_sales_features.csv"

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# ---------------------------------------------------------------------------
# Forecasting settings
# ---------------------------------------------------------------------------
TEST_DAYS = 56            # last 8 weeks are held out to test the model
FORECAST_HORIZON = 30     # how many days into the future we forecast
TARGET = "units_sold"
LAGS = [7, 14, 21, 28]    # "sales N days ago" (same weekday) history
HISTORY_DAYS = 35         # days of history needed to build all features

# ---------------------------------------------------------------------------
# Inventory policy settings
# ---------------------------------------------------------------------------
SERVICE_LEVEL = 0.95      # we want to avoid stockouts 95% of the time
ORDERING_COST = 500.0     # Rs. per purchase order (transport, admin, etc.)
HOLDING_COST_RATE = 0.20  # 20% of unit cost per year to hold one unit
REVIEW_PERIOD_DAYS = 7    # we review stock once a week
OVERSTOCK_DAYS = 60       # more than 60 days of cover = overstock

# Supplier lead time (days) by category - fresh goods arrive faster
LEAD_TIME_BY_CATEGORY = {
    "Groceries": 3,
    "Dairy & Bakery": 2,
    "Personal Care": 5,
    "Home Care": 5,
    "Electronics": 10,
}


def make_dirs() -> None:
    """Create all output folders if they do not exist yet."""
    for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
                   OUTPUTS_DIR, IMAGES_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
