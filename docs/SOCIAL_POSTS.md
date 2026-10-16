# Social Media Posts: LinkedIn & Instagram

**GitHub URL (once pushed):** https://github.com/SNischayPrasad/Retail-Sales-Forecasting-Inventory-Optimization

**Ready-made images**
| Asset | File | Use for |
|---|---|---|
| 7-slide carousel (1080×1350) | `images/social/slide_01.png` … `slide_07.png` | Instagram carousel |
| Same carousel as PDF | `images/social/linkedin_carousel.pdf` | LinkedIn "document" post (shows as swipeable slides) |
| Dashboard screenshots | `images/screenshots/08_dashboard_overview.png`, `10_dashboard_forecast.png`, `09_dashboard_reorder_alerts.png` | LinkedIn image post, README |
| Terminal proof | `images/screenshots/05_model_training_output.png`, `07_all_tests_passing.png` | Proof that the code runs |

Regenerate the slides after any change with `python scripts/make_social_posts.py`.

---

## LinkedIn

### Post 1: main launch post (upload `linkedin_carousel.pdf` as a document)
**Document title:** `Retail Demand Forecasting & Inventory Optimization – Project Walkthrough`

```
Empty shelves lose sales. Full storerooms lock up cash. 📦

I wanted to understand how retailers like D-Mart, Reliance Retail and Amazon decide what to stock, so I built an end-to-end Retail Sales Forecasting & Inventory Optimization system.

No company data? I built a virtual supermarket instead: 3 stores, 20 products and 2 years of daily sales, with trend, weekends, summer and winter products, Diwali and Holi spikes, promotions, supplier delays and messy POS data.

What the system does:
🔹 Cleans the data (duplicates, outliers, 505 missing values fixed)
🔹 Forecasts demand with Random Forest: 30% lower error than a seasonal baseline, ~20% WAPE at the weekly level
🔹 Converts forecasts into Safety Stock, Reorder Point and EOQ for every product in every store
🔹 Produces a daily reorder list and a Streamlit dashboard

The result: I replayed 8 weeks of demand under the store's old rule of thumb and under my policy.
✅ Stockout days: 6.7% → 2.2%
✅ Lost revenue: ₹9.2L → ₹2.7L
✅ Total inventory cost: −40%
(Trade-off: 31% more average stock, but it's held where demand actually needs it.)

The biggest lesson: my first model over-forecast by 22% after Diwali. The festival spike was feeding back into its own predictions. Redesigning the features brought the bias down to +7%. Accuracy alone isn't enough. Always check bias.

Swipe through the slides 👉 Code, dashboard and documentation:
🔗 https://github.com/SNischayPrasad/Retail-Sales-Forecasting-Inventory-Optimization

Built with Python, Pandas, scikit-learn, Streamlit and pytest.
Note: the data is simulated, but the methods are the ones used in industry.

I'd love feedback from anyone working in retail, supply chain or demand planning! 🙌

#DataScience #MachineLearning #SupplyChain #RetailAnalytics #DemandForecasting #InventoryManagement #Python #DataAnalytics #BusinessAnalytics #OpenToWork
```

### Post 2: short dashboard post (attach 3 images: `08_dashboard_overview.png`, `10_dashboard_forecast.png`, `09_dashboard_reorder_alerts.png`)
```
Here's the dashboard from my retail forecasting project 📊

A store manager opens it every morning and sees:
→ Next 30 days of demand per product and store
→ Which items are CRITICAL or need reordering, with exact order quantities
→ Forecast accuracy and how the new policy compares to the old one

It's built with Streamlit on top of a Random Forest forecast and classic inventory formulas (safety stock, reorder point, EOQ). In simulation, it cut stockout days by 67%.

Full project 👉 https://github.com/SNischayPrasad/Retail-Sales-Forecasting-Inventory-Optimization

#Streamlit #DataVisualization #SupplyChain #Python #DataScience
```

### Post 3: lesson-learned post (attach `05_model_training_output.png`)
```
My model was 62% accurate, and it was still wrong. Here's why. 🧵

In my retail demand forecasting project, the first model looked fine on error metrics, but it over-forecast total demand by 22%. In a real store, that means ₹ lakhs of excess stock.

The cause: I forecast 30 days ahead one day at a time, feeding each prediction back in as "yesterday's sales". Right after Diwali, the festival spike kept echoing through those features for weeks.

The fix:
1️⃣ Dropped very short lags (you don't know yesterday's sales 30 days in advance anyway)
2️⃣ Added a 28-day "normal day" baseline that ignores festival and promotion days

Result: bias went from +22% to +7%, error dropped from 42.8% to 38.1%, and the pipeline got 6× faster.

Lesson: always check BIAS, not only accuracy, and test your model the way it will actually be used.

Code 👉 https://github.com/SNischayPrasad/Retail-Sales-Forecasting-Inventory-Optimization
#MachineLearning #TimeSeries #DataScience #LessonsLearned #Forecasting
```

### LinkedIn "Projects" section (profile → Add section → Projects)
- **Name:** Retail Sales Forecasting & Inventory Optimization System
- **Description:** `End-to-end demand forecasting (Random Forest, 30% better than baseline) and inventory optimization (safety stock, reorder point, EOQ) for a simulated 3-store retail chain. A policy simulation showed 67% fewer stockout days and 40% lower inventory cost. Includes a Streamlit dashboard and automated business report. Python · Pandas · scikit-learn · Streamlit · pytest.`
- **Link:** the GitHub URL. Add `slide_01.png` as the media thumbnail.

---

## Instagram

**Format:** carousel post. Upload `slide_01.png` → `slide_07.png` in order (the 4:5 portrait size fits the feed exactly).
**Before posting:** put the GitHub URL in your bio (or in a Linktree), because links in Instagram captions are not clickable.

### Caption
```
Empty shelves lose sales. Full storerooms lock up cash. 📦💸

I built an ML system that predicts how much a supermarket will sell, and tells it exactly what to order and when.

Swipe to see how 👉

🛒 Simulated 3 stores × 20 products × 2 years (festivals, promos, supplier delays included)
🤖 Random Forest forecast: 30% better than the baseline
📐 Safety stock + reorder point + EOQ for every product
📉 Result: 67% fewer stockout days, 40% lower inventory cost

Biggest lesson: a model can look accurate and still be biased. Mine over-forecast by 22% after Diwali until I fixed the features 🪔

Full code + dashboard 🔗 link in bio

Simulated data, real industry methods.

.
.
#datascience #machinelearning #python #dataanalytics #supplychain #retail #inventorymanagement #forecasting #coding #studentdeveloper #techstudent #engineeringstudent #datascienceprojects #pythonprojects #codinglife #developer #analytics #ai #placement #internship
```

### Story ideas (24-hour stories, link sticker → GitHub URL)
1. `slide_01.png` + poll sticker: "Would you trust an AI to stock your shop? Yes / No"
2. A screen recording of the dashboard (15 s), with a link sticker to GitHub
3. `slide_06.png` with the text "the results 👀"

### Reel idea (15–30 s)
Screen-record `python main.py` running → cut to charts appearing → cut to the dashboard → end on `slide_06.png`. Use trending audio and the on-screen text "I built an AI inventory manager for a supermarket 🛒".

---

## Posting tips
- Post LinkedIn on **Tue–Thu, 8–10 AM IST**. Reply to every comment in the first hour.
- Tag skills, not random people. Mention the project in your headline: *"Aspiring Data Analyst | Built Retail Demand Forecasting System (Python, ML)"*.
- Don't post all 3 LinkedIn posts in one week. Space them 5–7 days apart.
- Always keep "simulated data" in the post. Recruiters value honesty, and interviewers will ask.
