# GitHub Upload, Commit Strategy & Proof Checklist

- [J. GitHub upload steps](#j-github-upload-steps)
- [L. Commit strategy: 8-day proof plan](#l-commit-strategy-8-day-proof-plan)
- [M. Screenshots & proof checklist](#m-screenshots--proof-checklist)

> **Honesty rule.** The commit history should reflect real work. Go through each stage yourself: run it, read the code, change something, and understand the output before you commit it. Do **not** back-date commits or claim the data is from a real company. Recruiters respect "simulated data, real methods", and interviewers will test whether you understand the code.

---

## J. GitHub upload steps

### 1. Repository details (copy-paste)
| Field | Value |
|---|---|
| **Repository name** | `Retail-Sales-Forecasting-Inventory-Optimization` |
| **Description** | `End-to-end retail demand forecasting (Random Forest) + inventory optimization (Safety Stock, Reorder Point, EOQ) with policy simulation and Streamlit dashboard. Cut simulated stockout days 6.7% → 2.2%.` |
| **Topics** | `data-science` `machine-learning` `time-series-forecasting` `demand-forecasting` `inventory-optimization` `supply-chain` `retail-analytics` `python` `scikit-learn` `pandas` `streamlit` `random-forest` `business-analytics` `eoq` |
| **Visibility** | Public |
| **Initialize with README?** | **No.** Create it empty, because you already have a README |

### 2. One-time Git setup
```bash
git --version
git config --global user.name "Your Name"
git config --global user.email "you@example.com"     # same email as your GitHub account
```

### 3. Check what will be uploaded
Run these commands inside the project folder:
```bash
git init
git branch -M main
git status
```
`.venv/`, `data/processed/*.csv`, `models/*.joblib`, and `__pycache__/` must **not** be listed. `.gitignore` handles this.

**Files that must be included:** `README.md`, `requirements.txt`, `.gitignore`, `main.py`, `src/`, `app/`, `tests/`, `notebooks/`, `docs/`, `data/raw/`, `outputs/`, `images/`, `reports/`, `models/model_metadata.json`, and `LICENSE`.

### 4. Create the repository on GitHub
1. Go to github.com → **+** (top right) → **New repository**.
2. Enter the name and description from the table above, choose **Public**, and leave all "Initialize" boxes **unticked**.
3. Click **Create repository** and copy the HTTPS URL.

### 5. Connect and push
```bash
git remote add origin https://github.com/SNischayPrasad/Retail-Sales-Forecasting-Inventory-Optimization.git
git add .
git commit -m "feat: initial project structure and configuration"
git push -u origin main
```
GitHub asks you to sign in. Passwords no longer work for Git over HTTPS, so use one of these:
- **GitHub CLI (easiest):** install it from cli.github.com, run `gh auth login`, and then `git push` works.
- **Personal Access Token:** GitHub → Settings → Developer settings → Tokens (classic) → tick `repo` scope. Paste the token when Git asks for a password.

### 6. Later pushes
```bash
git status
git add <files>
git commit -m "type: short description"
git push
```

### 7. Meaningful commit messages
Use the format `type: what changed (why)`:
| Type | Use for | Example |
|---|---|---|
| `feat` | new functionality | `feat: add EOQ and reorder point calculation` |
| `fix` | bug fix | `fix: fill missing prices with list price instead of 0` |
| `data` | dataset changes | `data: add festival and promotion effects to simulation` |
| `docs` | documentation | `docs: add results table and screenshots to README` |
| `test` | tests | `test: add leakage tests for lag features` |
| `refactor` | restructure without behaviour change | `refactor: vectorise rolling features (8x faster)` |
| `perf` | speed | `perf: limit recursive window to 35 days` |

Avoid messages like `update`, `final`, `final2`, or `changes`.

### 8. Upload screenshots, GIF, or demo video
- Put manual screenshots in `images/screenshots/` (see the checklist below), then commit and push.
- **GIF:** record the dashboard with ScreenToGif (Windows) or Kap (Mac), keep it under 10 MB, save it as `images/screenshots/dashboard_demo.gif`, and reference it with `![Demo](images/screenshots/dashboard_demo.gif)`.
- **Video (MP4):** open your README on GitHub → click ✏️ Edit → drag the `.mp4` file into the editor. GitHub uploads it and inserts a link. Commit from the web, then run `git pull` locally.
- Alternatively, upload a 2-minute walkthrough to YouTube or LinkedIn and link it at the top of the README.

### 9. Make the repo recruiter-friendly
- In the first 5 lines of the README, say what the project does and the headline result (already done).
- Show the results table and screenshots **before** the installation steps if you want recruiters to see impact first.
- **Pin the repository**: GitHub profile → *Customize your pins* → select this repo.
- In the repo's **About** box (⚙️), add the description, topics, and website (the Streamlit Cloud link, if deployed).
- Add a **LICENSE** (MIT is included) and make sure the README author section has your real links.
- Optional: deploy free on **Streamlit Community Cloud** (share.streamlit.io → New app → choose the repo → `app/streamlit_app.py`). The app needs the outputs and `data/processed/retail_sales_clean.csv`. Either commit that file, or change `.gitignore` before deploying.

---

## L. Commit strategy: 8-day proof plan

On each day, work through that stage, then commit only the files listed. The commands assume you are in the project root.

### Day 1: Project setup
- **Do:** create the venv, install requirements, and read `config.py`.
- **Commit:**
  ```bash
  git add .gitignore requirements.txt pytest.ini LICENSE src/__init__.py src/config.py README.md
  git commit -m "feat: project structure, configuration and environment setup"
  git push -u origin main
  ```
- **Proof:** screenshot `pip install` success plus the `python -c "import sklearn"` check → `images/screenshots/01_environment_setup.png`.
- **Notes for LinkedIn:** "Started a retail demand forecasting & inventory optimization project. Day 1: designed a modular pipeline architecture."

### Day 2: Virtual simulation + cleaning
- **Do:** run `python -m src.data_generator` and `python -m src.data_preprocessing`. Open the raw CSV in Excel.
- **Commit:**
  ```bash
  git add src/data_generator.py src/data_preprocessing.py data/raw/ outputs/data_quality_report.csv
  git commit -m "data: simulate 2-year retail dataset and add cleaning pipeline with quality report"
  ```
- **Proof:** `02_raw_dataset_preview.png` (Excel or `df.head()`), `03_data_quality_report.png` (terminal).
- **Notes:** number of rows, stores, and products; which issues were found and fixed (132 duplicates, 505 imputed values, and so on).

### Day 3: EDA
- **Do:** run `python -m src.eda` and write one insight per chart.
- **Commit:**
  ```bash
  git add src/eda.py images/01_*.png images/02_*.png images/03_*.png images/04_*.png images/05_*.png images/06_*.png images/07_*.png images/08_*.png outputs/eda_summary.csv
  git commit -m "feat: exploratory analysis - trend, seasonality, promotions, stockouts"
  ```
- **Proof:** charts 01, 06, 07, and 08.
- **Notes:** "Saturday is the busiest day; promotions lift sales ~35%; 7.9% stockout days cost an estimated Rs. 1.5 crore."

### Day 4: Feature engineering + tests
- **Commit:**
  ```bash
  git add src/feature_engineering.py tests/__init__.py tests/test_features.py
  git commit -m "feat: leakage-free calendar, lag and festival-free baseline features"
  ```
- **Proof:** `pytest tests/test_features.py -v` output → `04_feature_tests.png`.
- **Notes:** what data leakage is and how you prevented it.

### Day 5: Forecasting model
- **Commit:**
  ```bash
  git add src/forecasting.py outputs/model_metrics.csv outputs/test_predictions.csv outputs/feature_importance.csv outputs/forecast_next_30_days.csv outputs/forecast_error_stats.csv models/model_metadata.json images/09_*.png images/10_*.png images/11_*.png images/12_*.png
  git commit -m "feat: baseline vs linear regression vs random forest with 56-day recursive backtest"
  ```
- **Proof:** terminal model comparison → `05_model_training_output.png`; charts 09 and 10.
- **Notes:** "RF cut error 30% vs baseline; weekly WAPE 20%; found and fixed a +22% post-Diwali over-forecast by redesigning features."

### Day 6: Inventory optimization
- **Commit:**
  ```bash
  git add src/inventory_optimization.py tests/test_inventory.py outputs/inventory_recommendations.csv outputs/reorder_alerts.csv outputs/policy_comparison.csv
  git commit -m "feat: safety stock, reorder point, EOQ recommendations and policy simulation"
  ```
- **Proof:** `06_inventory_recommendations_csv.png` (CSV opened in Excel), full `pytest -v` → `07_all_tests_passing.png`.
- **Notes:** the formulas, and "simulation: stockout days 6.7% → 2.2%, total cost −40%".

### Day 7: Visualization, report, dashboard
- **Commit:**
  ```bash
  git add src/visualization.py main.py app/ images/13_*.png images/14_*.png images/15_*.png images/16_*.png reports/ notebooks/
  git commit -m "feat: result charts, business insights report, Streamlit dashboard and notebook"
  ```
- **Proof:** dashboard screenshots and a 60–90 s screen recording (see section M).
- **Notes:** a short description of the dashboard's features for a LinkedIn post.

### Day 8: Documentation & polish
- **Commit:**
  ```bash
  git add docs/ images/screenshots/ README.md
  git commit -m "docs: README with results, architecture, screenshots and guides"
  git push
  ```
- Then: add topics in the About box, pin the repo, post on LinkedIn (templates are in INTERVIEW_PREP.md), and add the project to your resume.
- **Proof:** `12_github_repo_page.png`, `13_readme_preview.png`.

---

## M. Screenshots & proof checklist

Auto-generated charts are already saved in `images/` by `python main.py`. Save **manual** screenshots in `images/screenshots/` with these exact names:

| # | Proof asset | How to capture | File name |
|---|---|---|---|
| 1 | Environment setup | Terminal after `pip install` + import check | `images/screenshots/01_environment_setup.png` |
| 2 | Dataset preview | `data/raw/retail_sales_raw.csv` in Excel, or notebook `raw.head(10)` | `images/screenshots/02_raw_dataset_preview.png` |
| 3 | Missing values / preprocessing | Terminal data-quality report, or notebook `raw.isna().sum()` | `images/screenshots/03_data_quality_report.png` |
| 4 | EDA charts | auto | `images/01_daily_sales_trend.png` … `images/08_stockout_rate_by_category.png` |
| 5 | Sales trend graph | auto | `images/01_daily_sales_trend.png` |
| 6 | Category-wise sales | auto | `images/03_category_sales.png` |
| 7 | Feature tests | `pytest tests/test_features.py -v` | `images/screenshots/04_feature_tests.png` |
| 8 | Model training output | Terminal step 5 | `images/screenshots/05_model_training_output.png` |
| 9 | Forecast graph | auto | `images/11_forecast_next_30_days.png` |
| 10 | Actual vs predicted | auto | `images/10_actual_vs_predicted.png` |
| 11 | Inventory recommendation table | `outputs/inventory_recommendations.csv` in Excel with filters | `images/screenshots/06_inventory_recommendations_csv.png` |
| 12 | Reorder alerts | auto + dashboard "Inventory actions" tab | `images/14_reorder_alerts_table.png`, `images/screenshots/09_dashboard_reorder_alerts.png` |
| 13 | All tests passing | `pytest -v` | `images/screenshots/07_all_tests_passing.png` |
| 14 | Dashboard overview | Streamlit KPI tiles + Sales trend tab | `images/screenshots/08_dashboard_overview.png` |
| 15 | Dashboard forecast tab | Pick "Basmati Rice 5kg" | `images/screenshots/10_dashboard_forecast.png` |
| 16 | Policy simulation | auto | `images/15_inventory_simulation.png`, `images/16_policy_comparison.png` |
| 17 | GitHub repo page | Browser screenshot after the final push | `images/screenshots/12_github_repo_page.png` |
| 18 | README preview | Scroll to the results section on GitHub | `images/screenshots/13_readme_preview.png` |
| 19 | Demo GIF | 60–90 s: run `main.py` → open the dashboard → click the tabs → download the CSV | `images/screenshots/dashboard_demo.gif` |

**Tables to keep in the repo:** `outputs/model_metrics.csv`, `outputs/inventory_recommendations.csv`, `outputs/reorder_alerts.csv`, `outputs/policy_comparison.csv`, `outputs/data_quality_report.csv`.

**Screen recordings to capture:**
1. Running `python main.py` from start to finish (shows that it really works).
2. A walkthrough of the dashboard tabs.
3. Optional: a 2-minute voice-over explaining the problem → approach → result (great for LinkedIn).

**Reference images in the README:**
```markdown
![Dashboard overview](images/screenshots/08_dashboard_overview.png)
![Demo](images/screenshots/dashboard_demo.gif)
```
Paths are relative to the README and case-sensitive. Use forward slashes, even on Windows.
