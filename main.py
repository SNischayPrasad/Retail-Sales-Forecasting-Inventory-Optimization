"""
main.py
-------
Runs the COMPLETE pipeline in one command:

    python main.py                 # full run (simulate data -> report)
    python main.py --skip-simulation   # reuse existing data/raw files

Pipeline:
  1. Virtual simulation  -> data/raw/*.csv
  2. Data cleaning       -> data/processed/retail_sales_clean.csv
  3. EDA                 -> images/01-08
  4. Feature engineering -> data/processed/retail_sales_features.csv
  5. Forecasting         -> models/, outputs/forecast_next_30_days.csv
  6. Inventory optimisation + policy simulation -> outputs/*.csv
  7. Visualisation & business report -> images/09-16, reports/
"""
import argparse
import time

from src import config
from src.data_generator import generate_dataset
from src.data_preprocessing import run_preprocessing
from src.eda import run_eda
from src.feature_engineering import run_feature_engineering
from src.forecasting import run_forecasting
from src.inventory_optimization import run_inventory_optimization
from src.visualization import run_visualization


def banner(step: int, text: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {step}: {text}\n{'=' * 70}")


def main(skip_simulation: bool = False) -> None:
    start = time.time()
    config.make_dirs()

    banner(1, "Virtual retail simulation (synthetic dataset)")
    if skip_simulation and config.RAW_SALES_FILE.exists():
        print("[simulation] skipped - using existing raw data")
    else:
        generate_dataset()

    banner(2, "Data cleaning & preprocessing")
    clean = run_preprocessing()

    banner(3, "Exploratory data analysis")
    eda = run_eda(clean)

    banner(4, "Feature engineering")
    feats = run_feature_engineering(clean)

    banner(5, "Sales forecasting (train, test, forecast)")
    fc = run_forecasting(clean, feats)

    banner(6, "Inventory optimisation & policy simulation")
    rec, alerts, comparison, example = run_inventory_optimization(
        clean, fc["forecast"], fc["error_stats"], fc["test_preds"][fc["best_name"]])

    banner(7, "Visualisation & business report")
    run_visualization(clean, eda, fc, rec, alerts, comparison, example)

    print(f"\nPipeline finished in {time.time() - start:.0f} seconds.")
    print("Next: open reports/business_insights.md or run "
          "`streamlit run app/streamlit_app.py`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail forecasting pipeline")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="reuse data/raw instead of regenerating it")
    main(parser.parse_args().skip_simulation)
