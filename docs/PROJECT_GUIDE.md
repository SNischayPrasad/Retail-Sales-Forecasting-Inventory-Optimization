# Project Guide: Retail Sales Forecasting & Inventory Optimization

A beginner-friendly but complete guide. Read it in order the first time, then use it as a reference.

| Section | Covers |
|---|---|
| [A. Project explanation](#a-project-explanation) | What the system is and why retailers need it |
| [B. Tech stack options](#b-tech-stack-options) | Easy, intermediate, and advanced versions |
| [C. Selected approach](#c-selected-approach) | Why this version was chosen |
| [D. Architecture](#d-architecture) | Modules and data flow |
| [E. Folder structure](#e-folder-structure) | Purpose of every folder and file |
| [F. Installation](#f-installation-windows--maclinux) | Windows and Mac/Linux setup |
| [G. Code walkthrough](#g-code-walkthrough-file-by-file) | What each code file does |
| [Implementation phases](#implementation-plan-12-phases) | 12 phases with checks and common mistakes |
| [H. Virtual simulation workflow](#h-virtual-simulation-workflow) | How the "fake company" works |
| [I. How to run](#i-how-to-run) | Commands, expected output, success criteria |
| [O. Future improvements](#o-future-improvements) | How to upgrade the project later |
| [P. Troubleshooting](#p-troubleshooting) | Common errors and fixes |

GitHub steps, the commit plan, and the screenshot checklist are in [GITHUB_AND_PROOF_PLAN.md](GITHUB_AND_PROOF_PLAN.md).
Resume bullets, LinkedIn posts, and interview Q&A are in [INTERVIEW_PREP.md](INTERVIEW_PREP.md).

---

## A. Project explanation

### In simple language
Imagine you run a supermarket. Every morning you have to decide what to order from suppliers.
- If you order **too little**, the shelf becomes empty. The customer leaves and buys from another shop, so you lose that sale and perhaps the customer too.
- If you order **too much**, your money is stuck in boxes in the storeroom. Milk and bread may expire, and storage space costs money.

A **Retail Sales Forecasting & Inventory Optimization System** does two jobs:
1. **Forecasting:** it looks at past sales and predicts how many units of each product each store will sell in the coming days.
2. **Inventory optimization:** it uses that prediction to tell the manager **when** to order (the reorder point) and **how much** to order (the order quantity), while keeping a small safety cushion (safety stock) for surprises.

### In technical language
The system is a demand-forecasting and replenishment pipeline. It works on **store × SKU × day** time series:
- A supervised regression model learns `units_sold = f(calendar, lagged demand, baseline level, promotion, festival, price, identifiers)`. It is evaluated with time-ordered back-testing and recursive multi-step forecasting.
- Forecast error variability (σ) is used to set **safety stock** for a target **cycle service level** (95%, Z = 1.645) over the supplier **lead time**. **Reorder point** = lead-time demand + safety stock. The order size comes from the **EOQ** (Economic Order Quantity), which balances ordering cost against holding cost, together with an order-up-to target.
- A discrete-event inventory simulation compares policies on KPIs: fill rate, stockout rate, inventory value, and total cost.

### Why it matters in retail
- Retail margins are thin (2–5% net for grocery). Waste and lost sales decide whether a business makes a profit.
- Industry studies typically find that stockouts cost retailers around 4% of revenue, and that most of those customers buy elsewhere.
- Inventory is often a retailer's largest current asset, so lower inventory means more cash available.

### How real companies use such systems
| Company | Use case |
|---|---|
| **D-Mart (Avenue Supermarts)** | Known for a lean assortment and fast inventory turnover; demand-driven replenishment keeps stock lean |
| **Reliance Retail** | Forecasts demand across thousands of stores and SKUs for Smart Bazaar and JioMart, including festival build-up |
| **Amazon** | Probabilistic forecasting models (e.g. DeepAR) decide how much of each product to place in which fulfilment centre |
| **Flipkart** | Forecasts sale-event demand (Big Billion Days) weeks in advance to pre-position inventory |
| **Walmart** | Store-SKU forecasts (its M5 dataset is a famous Kaggle competition) drive automatic replenishment |
| **Blinkit / Zepto** | Dark-store-level forecasts keep a 10-minute delivery promise without overstocking tiny stores |

### How forecasting helps
| Business goal | How the system helps |
|---|---|
| Demand planning | Know next month's volume per store and category |
| Stock planning | Reorder point and order quantity per SKU |
| Avoid stockouts | Safety stock covers forecast error during the lead time |
| Avoid overstock | Days-of-cover > 60 → stop ordering and clear stock with promotions |
| Improve profit | Fewer lost sales plus lower holding and ordering cost |
| Festival readiness | The model learns festival uplift, so stock is built before Diwali, not after |

### Complete workflow
```
Historical sales collection   -> POS exports (here: simulated)
        ↓
Preprocessing                 -> fix dates, duplicates, missing values, outliers
        ↓
Exploratory data analysis     -> trend, seasonality, categories, stores, promos, stockouts
        ↓
Feature engineering           -> calendar, lags, festival-free baseline, price ratio
        ↓
Sales forecasting             -> baseline vs Linear Regression vs Random Forest
        ↓
Inventory optimization        -> safety stock, reorder point, EOQ, target stock
        ↓
Reorder recommendations       -> CRITICAL / REORDER / OVERSTOCK / HEALTHY + order qty
        ↓
Insights dashboard & report   -> charts, Streamlit dashboard, business_insights.md
```

---

## B. Tech stack options

| | **Option A: Easiest** | **Option B: Intermediate** ✅ | **Option C: Advanced** |
|---|---|---|---|
| Tools | Excel/CSV, Pandas, Matplotlib, Jupyter | Python modules, Pandas, NumPy, Seaborn, scikit-learn, Streamlit, pytest | + LightGBM/XGBoost, Prophet/statsmodels, Optuna, MLflow, Docker, Airflow, cloud deploy |
| Model | Moving average / exponential smoothing | Seasonal naive baseline + Linear Regression + Random Forest (global model, recursive) | Gradient boosting, quantile models, deep learning (DeepAR/N-BEATS) |
| Inventory logic | Fixed reorder level | SS + ROP + EOQ + policy simulation | Stochastic optimization, multi-echelon, MOQ and budget constraints |
| Difficulty | ★☆☆ | ★★☆ | ★★★ |
| GPU needed | No | No (runs in ~45 s on a laptop) | Only for deep learning |
| Outcome | Notebook with charts | Modular, tested project with dashboard and measurable business impact | Production-like MLOps system |
| Time to build | 2–3 days | 7–10 days | 4–8 weeks |

**Best for strong GitHub proof quickly: Option B.** It looks like real industry work (modular code, tests, honest validation, business KPIs, and a dashboard), but it does not get stuck on installation problems (Prophet and XGBoost sometimes fail to install on Windows) or on infrastructure.

---

## C. Selected approach
**Option B: Python + scikit-learn + Streamlit.** The reasons:
1. Every library installs with `pip` on Windows, Mac, and Linux without a compiler.
2. Random Forest handles non-linear effects (for example, festival × category) with little tuning.
3. A *global* model (one model for all 60 series, with store and product codes as features) is how industry handles thousands of SKUs.
4. The project measures **business impact** (cost, fill rate) and not only model accuracy. That impact is what interviewers remember.

---

## D. Architecture

### Block diagram
```
                         src/config.py  (all settings)
                                │
┌───────────────────────────────┼─────────────────────────────────────────┐
│                               ▼                                         │
│  [1] data_generator.py ──► data/raw/retail_sales_raw.csv                │
│        demand rules + naive stock policy + injected errors              │
│                               │                                         │
│  [2] data_preprocessing.py ──► data/processed/retail_sales_clean.csv    │
│        dates, text, dupes, negatives, outliers, imputation, revenue     │
│                               │                                         │
│          ┌────────────────────┴──────────────┐                          │
│          ▼                                   ▼                          │
│  [3] eda.py ──► images/01-08         [4] feature_engineering.py         │
│                                              │  17 features             │
│                                              ▼                          │
│                                 [5] forecasting.py                      │
│                         train ≤ 5 Nov 2024 │ test 56 days (recursive)   │
│                         best model → models/  30-day forecast → outputs/│
│                                              │                          │
│                                              ▼                          │
│                              [6] inventory_optimization.py              │
│                     SS / ROP / EOQ / status → recommendations, alerts   │
│                     policy simulation old vs new → policy_comparison    │
│                                              │                          │
│                                              ▼                          │
│                              [7] visualization.py                       │
│                     images/09-16 + reports/business_insights.md         │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  app/streamlit_app.py (reads outputs/)
```

### Module-by-module
| # | Module | Input | Output | Key idea |
|---|---|---|---|---|
| 1 | Data generator | config | raw sales, products, stores CSV | Known business rules generate the data, so we know which patterns should be found |
| 2 | Preprocessing | raw CSVs | clean CSV, data_quality_report.csv | Every fix is counted, so the cleaning can be proven |
| 3 | EDA | clean CSV | 8 charts, eda_summary.csv | Understand the business before modelling |
| 4 | Feature engineering | clean CSV | feature table | Leakage-free history features |
| 5 | Forecasting | features | metrics, test predictions, model, 30-day forecast | Honest recursive back-test |
| 6 | Inventory optimization | forecast + error stats + current stock | recommendations, alerts, policy comparison | Forecast → decision |
| 7 | Visualization | all results | 8 charts + markdown report | Communicate to managers |
| – | Dashboard | outputs/ | web app | Self-service exploration |

### Data flow in one sentence
Raw rows → clean rows → feature rows → model → **forecast rows (store, product, date, units)** → aggregated per store-product → **one recommendation row per store-product** → charts, report, and dashboard.

---

## E. Folder structure
See the tree in the [README](../README.md#folder-structure). Purpose of each part:

| Path | Purpose | Commit to Git? |
|---|---|---|
| `data/raw/` | Simulated "source system" exports (2.6 MB) | Yes, so recruiters can see the data |
| `data/processed/` | Cleaned and feature tables (14 MB, regenerated) | No (in `.gitignore`) |
| `src/` | All reusable logic, one module per stage | Yes |
| `main.py` | Orchestrates the pipeline | Yes |
| `app/` | Streamlit dashboard | Yes |
| `notebooks/` | Storytelling walkthrough for readers who prefer notebooks | Yes |
| `models/` | `best_forecasting_model.joblib` (large, regenerated) + `model_metadata.json` | Only the metadata |
| `outputs/` | CSV deliverables (forecast, recommendations, alerts, metrics) | Yes, as proof |
| `images/` | Auto-generated charts; `images/screenshots/` holds your manual screenshots | Yes |
| `reports/` | Business insights report | Yes |
| `docs/` | Guides and sample terminal output | Yes |
| `tests/` | Unit tests | Yes |
| `requirements.txt` | Library list for `pip install -r` | Yes |
| `.gitignore` | Files Git should not track | Yes |

---

## F. Installation (Windows & Mac/Linux)

**Python version:** 3.11, 3.12 or 3.13 (tested on 3.13). Check it with `python --version`. If Python is missing, install it from python.org; on Windows, tick **"Add python.exe to PATH"**.

### Windows (PowerShell or Command Prompt)
```bash
cd C:\Users\<you>\Projects
# (copy or clone the project folder here)
cd Retail-Sales-Forecasting-Inventory-Optimization
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell says *"running scripts is disabled"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt and run `.venv\Scripts\activate.bat`.

### Mac / Linux
```bash
cd ~/Projects/Retail-Sales-Forecasting-Inventory-Optimization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install libraries one by one (if you are starting from an empty folder)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit pytest jupyter
```

### Generate requirements.txt yourself
```bash
pip freeze > requirements.txt          # exact versions (very long)
```
This project instead keeps a short, hand-written `requirements.txt` with minimum versions, which is easier for others to install.

### Verify that installation worked
```bash
python -c "import pandas, numpy, sklearn, seaborn, streamlit; print('All good', pandas.__version__, sklearn.__version__)"
pytest -q
```
Expected: `All good ...` and `14 passed`.

You know the virtual environment is active when your prompt starts with `(.venv)`. Run `deactivate` to leave it.

---

## G. Code walkthrough (file by file)
The complete, commented code is in the repository. This section explains what each file does and what to say about it in an interview.

### `src/config.py`
Holds all paths and settings: dates, test window (56 days), forecast horizon (30), service level (95%), ordering cost (Rs. 500), holding cost (20%/year), lead time per category, and the overstock threshold (60 days). **Why:** no "magic numbers" are scattered through the code, so changing an assumption takes one edit.

### `src/data_generator.py`: the virtual company
- `generate_master_data()` creates 3 stores and 20 products with selling price, cost price (70–85% of price), and lead time.
- `simulate_demand()` computes `expected = base × store size × trend × weekly × seasonal × festival × salary-week × promo` and then draws Gamma-Poisson noise.
- `simulate_naive_inventory()` runs the store's current rule day by day: reorder when stock + on-order is below 7 days of base sales, order 21 days of base sales, with lead times and 0–2 days of random supplier delay. This produces `opening_stock`, `units_received`, `units_sold = min(demand, stock)`, `closing_stock`, `stockout_flag`, and `lost_sales_units`.
- `inject_data_quality_issues()` adds missing values, duplicates, negative values, outliers, and messy category text.

### `src/data_preprocessing.py`
Parses dates → fixes text through master data → drops duplicates on (date, store, product) → turns negatives into NaN → flags outliers (> median + 6 × IQR per series) → interpolates missing sales in time order → fills missing prices from the list price (minus 10% on promo days) → adds revenue and gross profit. It prints and saves a **data quality report**.

### `src/eda.py`
Saves 8 charts (trend, monthly revenue, category, weekday, store, promotion uplift, seasonality heatmap, stockout rate) and a summary of key numbers (YoY growth, promo uplift, lost sales value).

### `src/feature_engineering.py`
Creates 17 features: calendar, IDs, price ratio, `lag_14`, `lag_28`, `same_weekday_mean_4w`, and a **festival- and promo-free 28-day baseline** with its standard deviation. Every history feature is shifted, so there is no leakage. Lag-1 and lag-7 were tested and removed because they caused compounding over-forecasts in 30-day recursive forecasting (a good interview story).

### `src/forecasting.py`
- `get_models()` returns the Seasonal Naive baseline, Linear Regression (with scaling), and Random Forest (200 trees, depth 14).
- `recursive_forecast()` predicts one day at a time and writes each prediction back into the history so later days can use it. This is how forecasting works in production.
- `train_and_evaluate()` trains on data up to 5 Nov 2024, tests on 56 days, and reports MAE, RMSE, WAPE, weekly WAPE, and bias.
- `run_forecasting()` saves the metrics, retrains the best model on all data, saves it with `joblib`, and forecasts the next 30 days.

### `src/inventory_optimization.py`
- Small pure functions: `safety_stock`, `reorder_point`, `economic_order_quantity`, `classify_stock`, `stockout_date`. Each one is unit-tested.
- `build_recommendations()` combines the forecast, the per-series forecast error σ, and today's closing stock into one action row per store-product.
- `simulate_policy()` is a vectorised day-by-day stock simulator.
- `run_policy_simulation()` replays the 56 test days of real demand under the old rule and the new policy (built only from information available at the cutoff), then computes cost and service KPIs.

### `src/visualization.py`
Saves charts 09–16 (model comparison, actual vs predicted, 30-day forecast, feature importance, status, reorder table, stock simulation, policy comparison) and writes `reports/business_insights.md`.

### `main.py`
Runs steps 1–7 with clear banners. `--skip-simulation` reuses the existing raw data.

### `app/streamlit_app.py`
Sidebar filters (store, category), KPI tiles, and 4 tabs: sales trend, product forecast, inventory actions (with a CSV download), and model & policy comparison.

### `tests/`
`test_inventory.py` checks the formulas against textbook examples and checks that simulated stock never goes negative. `test_features.py` checks that lags use only past values, that festival spikes are excluded from the baseline, and that the WAPE maths is correct.

### Beginner shortcut
If the full pipeline feels like too much at first, open `notebooks/project_walkthrough.ipynb` after running `python main.py`. Then read the modules in the order 1 → 7. Each one also runs alone, for example `python -m src.eda`.

---

## Implementation plan (12 phases)

| Phase | What to do | Why | Expected output | Common mistakes | How to verify |
|---|---|---|---|---|---|
| 1. Environment setup | Install Python, create a venv, install requirements | Isolated and reproducible setup | `(.venv)` prompt | Installing globally; forgetting to activate | `python -c "import sklearn"` works |
| 2. Project structure | Create folders, `config.py`, `.gitignore`, README stub | Professional layout from day 1 | Empty skeleton | Everything in one notebook | Tree matches the README |
| 3. Dataset loading / simulation | `python -m src.data_generator` | Data to work with | `data/raw/*.csv` | Wrong dates; no randomness seed | 43,992 rows; `head()` looks sensible |
| 4. Cleaning | `python -m src.data_preprocessing` | Models fail on dirty data | clean CSV + quality report | Dropping rows with NaN (creates gaps); using `fillna(0)` for sales | `remaining_missing_values: 0`, 43,860 rows |
| 5. EDA | `python -m src.eda` | Find the patterns to model | images 01–08 | Charts without titles or units | You can explain every chart in one sentence |
| 6. Feature engineering | `python -m src.feature_engineering` | Turn time into numbers | feature table | **Leakage**: using today's sales in a feature | `pytest tests/test_features.py` passes |
| 7. Forecasting | `python -m src.forecasting` | Predict demand | metrics, model, forecast | Random train/test split; evaluating with known future lags | RF beats the baseline; bias within ±10% |
| 8. Inventory logic | inside `main.py` step 6 | Turn forecasts into decisions | recommendations + alerts | Mixing units (days vs weeks); negative order quantities | Formula tests pass; numbers look sensible |
| 9. Visualization & insights | step 7 | Communicate | images 09–16, report | Showing only accuracy, not ₹ impact | Report reads well for a non-technical person |
| 10. Testing | `pytest -v` | Prove correctness | 14 passed | Never running tests | Green test run (screenshot it) |
| 11. GitHub publishing | see GITHUB_AND_PROOF_PLAN.md | Proof of work | Public repo | Pushing the venv or huge files | Repo page shows README and images |
| 12. Resume / presentation | see INTERVIEW_PREP.md | Get interviews | Bullets, posts, Q&A | Overclaiming ("real company data") | You can explain it in 60 seconds |

---

## H. Virtual simulation workflow

### How the business is represented
| Real-world thing | In this project |
|---|---|
| Store | `store_id` S01–S03 with city, region, and a size factor |
| Product / SKU | `product_id` P001–P020 with name, category, price, cost, and lead time |
| Day | One row per store × product × date (731 days) |
| Customer demand | `demand`, generated from rules (hidden in real life) |
| Sales | `units_sold = min(demand, stock on hand)` |
| Stock in | `units_received` when a purchase order arrives after its lead time (+ delay) |
| Stock out | `stockout_flag = 1` when demand > stock; the shortfall is `lost_sales_units` |
| Promotion | `promotion = 1` gives a 10% lower price and a 30–60% uplift |
| Festival | `festival_period = 1` inside festival windows |

### How variation is simulated
- **Demand variation:** Gamma noise (shape 8) on the expected rate, then Poisson sampling. This gives the over-dispersion seen in real data.
- **Trend:** `1 + growth × years` with category-specific growth.
- **Seasonality:** weekly multipliers; yearly cosine curves for summer and winter products; salary-week boost.
- **Events:** festival multipliers with ramp-up windows (Diwali: 10 days before).
- **Stock situations:** the naive policy ignores growth and festivals, so stockouts naturally rise over time and around Diwali (Electronics reaches about 26% stockout days).

### Step-by-step execution
1. **Step 1:** activate the venv and run `python -m src.data_generator`. Open `data/raw/retail_sales_raw.csv` in Excel and screenshot it.
2. **Step 2:** run `python -m src.data_preprocessing` and screenshot the data quality report in the terminal.
3. **Step 3:** run `python -m src.eda` and look at `images/01`–`08`.
4. **Step 4:** run `python -m src.feature_engineering` (17 features).
5. **Step 5:** run `python -m src.forecasting` and screenshot the model comparison lines.
6. **Step 6:** run `python main.py --skip-simulation` to run inventory optimization and the policy simulation together with everything else.
7. **Step 7:** open `outputs/inventory_recommendations.csv` and `reports/business_insights.md`.
8. **Step 8:** run `streamlit run app/streamlit_app.py`, click through all 4 tabs, and screen-record it.
9. **Step 9:** run `pytest -v` and screenshot the green result.
10. **Step 10:** experiment by changing `SERVICE_LEVEL` to 0.99 in `config.py`, rerun, and compare safety stock. This makes a great talking point ("higher service costs more inventory").

### How the forecast is generated
The best model is retrained on all history. For 1 Jan 2025 it builds features from real history and predicts. That prediction is appended to the history, and 2 Jan uses it as a lag, and so on until day 30. Future promotions are assumed to be 0, and festivals come from the calendar.

### How inventory decisions are made
For each store-product:
`σ` = std of daily forecast error in the test period → `SS = ⌈1.645 × σ × √L⌉` → `ROP = forecast demand over L days + SS` → `EOQ = √(2DS/H)` → `target = avg × (L + 7) + SS` → status → `order qty = max(EOQ, target − stock)` when the status is CRITICAL or REORDER.

### Proof assets to capture
Full checklist with file names: [GITHUB_AND_PROOF_PLAN.md → Proof checklist](GITHUB_AND_PROOF_PLAN.md#m-screenshots--proof-checklist).

---

## I. How to run

```bash
# 1. activate the environment
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac/Linux

# 2. full pipeline: simulation → cleaning → EDA → features → model → inventory → report
python main.py

# 3. tests
pytest -v

# 4. dashboard
streamlit run app/streamlit_app.py

# Run single stages (optional)
python -m src.data_generator
python -m src.data_preprocessing
python -m src.eda
python -m src.feature_engineering
python -m src.forecasting
```

### Sample terminal output (abridged; full copy in `docs/sample_terminal_output.txt`)
```
STEP 2: Data cleaning & preprocessing
[preprocessing] Data quality report:
    raw_rows                    : 43,992
    duplicates_removed          : 132
    negative_values_fixed       : 42
    outliers_fixed              : 25
    units_sold_imputed          : 505
    clean_rows                  : 43,860
    remaining_missing_values    : 0
STEP 5: Sales forecasting (train, test, forecast)
    Seasonal Naive (baseline)    MAE=13.30  WAPE=54.2%  Weekly WAPE=35.0%  Bias=+31.2%
    Linear Regression            MAE=9.79   WAPE=39.9%  Weekly WAPE=21.3%  Bias=+7.6%
    Random Forest                MAE=9.35   WAPE=38.1%  Weekly WAPE=20.4%  Bias=+6.8%
[forecasting] Best model: Random Forest
STEP 6: Inventory optimisation & policy simulation
[inventory] Z=1.645 for 95% service level | status counts: {'HEALTHY': 38, 'CRITICAL': 11, 'REORDER': 11}
[inventory] 22 reorder alerts | total order value Rs. 1,716,678
                             Fill rate %  Stockout days %  Avg inventory value (Rs)  Total cost (Rs)
Old naive rule                      97.8              6.7                 2062288.6         368379.9
Forecast-driven (ROP + EOQ)         99.2              2.2                 2707373.4         221645.4
Pipeline finished in 43 seconds.
```

### Sample forecast output
`outputs/forecast_next_30_days.csv` has 1,800 rows (60 series × 30 days):
```
date,store_id,product_id,category,forecast_units
2025-01-01,S01,P001,Groceries,31.4
2025-01-01,S01,P002,Groceries,56.65
```
Chart `11_forecast_next_30_days.png` shows each category's history (solid) and forecast (dashed). The weekend peaks and the Republic Day (26 Jan) spike are visible.

### Sample optimized inventory recommendation table
| Store | Product | Stock | Avg/day | Lead time | Safety stock | ROP | EOQ | Status | Order qty |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Shampoo 340ml | 0 | 17.1 | 5 | 32 | 122 | 394 | CRITICAL | 394 |
| S01 | LED Bulb 9W | 0 | 10.2 | 10 | 49 | 134 | 446 | CRITICAL | 446 |
| S01 | Basmati Rice 5kg | 29 | 30.7 | 3 | 37 | 134 | 325 | CRITICAL | 325 |
| S01 | Wheat Atta 10kg | 57 | 25.4 | 3 | 32 | 97 | 359 | REORDER | 359 |

### What success looks like
- The terminal ends with `Pipeline finished in ~40-60 seconds.`
- 16 PNG files in `images/` and 10 CSV files in `outputs/`.
- `reports/business_insights.md` exists and its numbers match the terminal.
- `pytest` shows `14 passed`.
- The dashboard opens at `http://localhost:8501` without errors.

Results are reproducible because the random seed is 42. If you change the seed, the numbers change slightly.

---

## O. Future improvements

| Upgrade | How a student can do it |
|---|---|
| Multi-store forecasting at scale | Add more stores in `STORES`; the global model already handles any number of stores |
| Product-level hierarchy | Forecast at category level and reconcile with SKU forecasts (top-down / bottom-up) |
| Region-wise demand | Add a region code feature and region-specific festivals (Onam, Pongal, Durga Puja) |
| Promotion impact modelling | Separate promotion depth (5/10/20%) and model the uplift by category |
| Price elasticity | Vary prices in the simulator, then fit `log(qty) ~ log(price)` per category |
| Weather/event demand | Add a temperature series (e.g. from the Open-Meteo API) as a feature for fans and ice-cream |
| Better models | `pip install lightgbm`, use quantile regression for P10/P50/P90 forecasts, tune with Optuna |
| Censored demand | Use `lost_sales_units` or unconstrained demand estimation to train on true demand |
| Real-time dashboard | Deploy on Streamlit Community Cloud; refresh data with a scheduled GitHub Action |
| Automated replenishment | Generate purchase-order PDFs/emails per supplier from `reorder_alerts.csv` |
| ERP integration | Read stock from a SQL database (SQLite/PostgreSQL) instead of CSV, and write POs back |
| Anomaly detection | Flag days where actual sales are outside the forecast interval (e.g. Isolation Forest on residuals) |
| Real dataset | Re-run the pipeline on Kaggle's *Walmart M5*, *Rossmann Store Sales*, or *Store Item Demand Forecasting* data |

---

## P. Troubleshooting

| Problem | Reason | Solution | Prevention |
|---|---|---|---|
| `'python' is not recognized` | Python is not on PATH | Reinstall and tick "Add to PATH", or use `py` on Windows | Check `python --version` first |
| `running scripts is disabled` (PowerShell) | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` | Or use `activate.bat` in cmd |
| `ModuleNotFoundError: No module named 'pandas'` | venv not active, or packages installed elsewhere | Activate the venv, then `pip install -r requirements.txt` | Look for `(.venv)` in the prompt |
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong folder | `cd` to the project root and use `python main.py` or `python -m src.eda` | Always run from the root |
| Library version conflicts / pip resolver errors | Old pip, or mixed global packages | `python -m pip install --upgrade pip`; delete `.venv` and recreate it | One venv per project |
| `FileNotFoundError: data/raw/...` | Simulation not run yet | `python main.py` (without `--skip-simulation`) | Run the full pipeline once first |
| Dates appear as text / `.dt accessor` error | CSV read without parsing | `pd.read_csv(..., parse_dates=["date"])` | Always parse dates when loading |
| `ValueError: Input contains NaN` while training | Rows without full history | Drop rows with NaN features (done in `run_feature_engineering`) | Check `df.isna().sum()` |
| Model accuracy looks "too good" | Data leakage (future values in features) or a random split | Use `shift()` and a time-based split | Run `tests/test_features.py` |
| Forecast is much higher than actuals | Recursive error compounding after peaks | Use longer lags and a festival-free baseline (as done here) | Always check bias, not only MAE |
| Negative forecasts | Linear models can go below 0 | `np.clip(pred, 0, None)` (done here) | Clip, or use a non-negative model |
| Safety stock is 0 or NaN | Fewer than 2 test points, or σ is NaN | Falls back to the std of recent sales (built in) | Make sure the test window is longer than 7 days |
| Huge EOQ values | Unit cost near 0 or wrong units | Check that `unit_cost` > 0 and that demand is annual | Unit-test the formulas |
| Plots do not appear | Code uses the `Agg` backend (saves files only) | Open the PNG files in `images/` | This is intended for scripts |
| `UnicodeEncodeError` writing the report on Windows | Default code page | Use `write_text(..., encoding="utf-8")` (done) | Always set the encoding |
| Streamlit "Output files not found" | Pipeline not run | `python main.py`, then start the app | Run the pipeline first |
| Streamlit port in use | Another app is running | `streamlit run app/streamlit_app.py --server.port 8502` | Close old terminals |
| `fatal: not a git repository` | `git init` not run | `git init` in the project root | Check with `git status` |
| `rejected ... fetch first` on push | The remote has commits (e.g. a README created on GitHub) | `git pull origin main --rebase`, then push | Create the GitHub repo **empty** |
| `Authentication failed` on push | GitHub no longer accepts passwords | Use a Personal Access Token or `gh auth login` | Set up GitHub CLI once |
| `file exceeds GitHub's 100 MB limit` | Large data or model files | Add them to `.gitignore`; `git rm --cached <file>` | Keep `data/processed/` and `*.joblib` ignored |
| Images do not show in the README | Wrong relative path or letter case | Use `images/name.png` exactly (paths are case-sensitive on GitHub) | Preview the README on GitHub after each push |
