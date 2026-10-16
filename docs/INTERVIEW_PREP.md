# Resume, LinkedIn & Interview Preparation

All numbers come from `python main.py` (seed 42). If you change the code, update the numbers. Always say the data is **simulated**.

---

## Resume bullet points (pick 2–3)

1. **Built an end-to-end retail demand forecasting system** in Python (Pandas, scikit-learn) for 60 store-SKU series over 2 years of simulated data. A Random Forest model reduced forecast error by **30% vs a seasonal baseline** (weekly WAPE **20%**), validated with a leakage-free 56-day recursive backtest.
2. **Designed inventory optimization logic** (safety stock at 95% service level, reorder point, EOQ) that generates daily reorder alerts. In simulation on real demand it **cut stockout days from 6.7% to 2.2%** and **reduced total inventory cost by 40%** vs the existing rule-of-thumb policy.
3. **Delivered a Streamlit dashboard and an automated business report**. The work included a data-cleaning pipeline (duplicates, outliers, 505 imputed values), 16 analytical charts, and 14 unit tests (pytest), all runnable with one command.

**Shorter version for a one-line project entry:**
*Retail Sales Forecasting & Inventory Optimization | Python, scikit-learn, Streamlit. Forecast-driven reorder policy cut simulated stockouts 67% and inventory cost 40%.*

---

## LinkedIn project descriptions

**1. Results-first post**
> 📦 Empty shelves lose sales. Full storerooms lock up cash. I built a system that balances the two.
>
> **Retail Sales Forecasting & Inventory Optimization** (Python · scikit-learn · Streamlit)
> 🔹 Simulated a 3-store supermarket chain: 20 products, 2 years, festivals, promotions, and supplier delays
> 🔹 Cleaned messy POS-style data (duplicates, outliers, missing values)
> 🔹 Random Forest demand forecast: 30% lower error than the baseline, ~20% weekly WAPE
> 🔹 Converted forecasts into safety stock, reorder points, and EOQ-based order quantities
> 🔹 Replayed 8 weeks of demand: stockout days 6.7% → 2.2%, total cost −40%
>
> The biggest lesson: my first model over-forecast by 17% after Diwali, because recursive forecasts were feeding festival spikes back in as inputs. Redesigning the features fixed it. Always check bias, not only accuracy.
> 🔗 GitHub: <link> #DataScience #SupplyChain #RetailAnalytics #MachineLearning #Python

**2. Learning-journey post**
> Over the last 8 days I built my most complete project so far, an inventory decision system for retail. I had no access to company data, so I wrote a simulator with trend, weekly and yearly seasonality, Indian festivals, and promotions, then built the full pipeline on top: cleaning → EDA → feature engineering → forecasting → inventory optimization → dashboard. What I learned: time-based validation, data leakage, why WAPE beats MAPE for retail, and how safety stock depends on the service level and lead time. Code, charts, and a dashboard demo are on GitHub: <link>

**3. Short featured-section description**
> End-to-end demand forecasting + inventory optimization for a simulated retail chain. Random Forest forecasts → safety stock / reorder point / EOQ → reorder alerts. Policy simulation shows −67% stockout days and −40% total cost. Python, scikit-learn, Streamlit, pytest.

---

## Explaining the project

### To HR (30 seconds, no jargon)
"Shops lose money in two ways: when shelves are empty, customers leave, and when they over-order, money gets stuck in stock. I built a system that learns from past sales to predict how much each product will sell in each store, and then tells the manager when to reorder and how much. I didn't have company data, so I created a realistic simulated supermarket. In a test on two months of demand, my approach reduced out-of-stock days by about two-thirds and cut total inventory cost by about 40%."

### To a technical interviewer (2 minutes)
"I simulated 3 stores × 20 SKUs × 731 days with a multiplicative demand model: trend, weekly and yearly seasonality, festival windows, promotions, and Gamma-Poisson noise. On top of that, I ran a day-by-day naive (s, Q) inventory policy with lead times and supplier delays, so stockouts and lost sales appear naturally. I also injected data-quality issues.

The pipeline cleans the data (dedupe, IQR-based outlier detection, time interpolation) and builds 17 leakage-free features. These include lag-14 and lag-28, a 4-week same-weekday mean, and a 28-day baseline that excludes festival and promotion days. I trained one global model for all series and compared a seasonal-naive baseline, Linear Regression, and Random Forest on a 56-day holdout using *recursive* forecasting, so the test mimics production. RF won with 38% daily WAPE, 20% weekly WAPE, and +6.8% bias.

Then I used the per-series forecast error σ to compute safety stock = Z·σ·√L at a 95% service level, the reorder point = lead-time demand + SS, and EOQ for order size, with an order-up-to target. Finally, I replayed the test period's true demand under the old rule and the new policy: fill rate went from 97.8% to 99.2%, stockout days from 6.7% to 2.2%, and total cost fell 40%, at the price of 31% more average inventory."

### Explaining the business value confidently
- Speak in **rupees and percentages**, not only model metrics: "Rs. 9.2 lakh of lost revenue in 8 weeks dropped to Rs. 2.7 lakh."
- State the **trade-off** yourself before you are asked: "We hold more inventory, but for the right products."
- Connect the work to **decisions**: "Every morning the manager gets a list of 22 items to order, with quantities."
- Be honest about **limits**: "The data is simulated, so the absolute numbers aren't real. The method is what transfers."

---

## 10 interview questions with strong answers

**1. Why did you simulate data instead of using a Kaggle dataset?**
Public datasets (M5, Rossmann) contain sales but rarely contain stock levels, lead times, or costs. I needed those to do inventory optimization. Simulating also meant I *knew* the true patterns (for example, Diwali ×1.8 and Saturday ×1.3), so I could check whether the model learned them. The same pipeline can run on M5 data as a next step.

**2. How did you split the data, and why not a random split?**
I used a time-based split: train up to 5 Nov 2024, test on the last 56 days. A random split leaks the future into training, because the model would see 10 Dec while predicting 9 Dec, and that inflates accuracy. I also forecast the test period *recursively*, feeding predictions back as lags, because in real life you don't know next week's sales when you plan.

**3. What is data leakage, and how did you prevent it?**
Leakage means using information at training time that would not be available at prediction time. All my history features use `shift()` of at least 1 day, and I have unit tests that check that `lag_14` equals the value from 14 days earlier and that today's sales never enter the baseline.

**4. Why WAPE and not MAPE?**
MAPE divides by each actual value, so it explodes on days with low or zero sales (common for slow movers like earbuds). WAPE = Σ|error| / Σactual weights errors by volume, which matches business impact. I also track bias, because a model can have decent WAPE but consistently over-forecast, which creates overstock.

**5. Your Random Forest is only ~62% accurate daily. Is that good?**
At store-SKU-day level, retail demand is very noisy. My simulator's noise alone sets a floor of roughly 30% WAPE. What matters is (a) beating the baseline (30% lower error), (b) accuracy at the decision level (weekly WAPE of 20%), and (c) low bias. Safety stock exists exactly to cover the remaining uncertainty.

**6. Tell me about a problem you faced.**
My first model over-forecast by 22%. I compared one-step and recursive predictions and found that the one-step bias was only 4%. The problem was the Diwali spike entering lag-1 and lag-7 features, which then compounded during recursion. I removed short lags (they aren't known 30 days ahead anyway) and added a 28-day baseline that excludes festival and promotion days. Bias dropped to +6.8% and WAPE improved from 42.8% to 38.1%. The rolling features were also vectorised, which made the pipeline 6× faster.

**7. Explain safety stock and reorder point with numbers.**
Shampoo at Andheri sells about 17 units a day with forecast error σ ≈ 8.5 and a lead time of 5 days. Safety stock = 1.645 × 8.5 × √5 ≈ 32 units. Lead-time demand ≈ 90 units, so the reorder point is about 122. When stock falls to 122, we order. It was at 0, so the status is CRITICAL and we order max(EOQ = 394, target − stock).

**8. What is EOQ, and what are its limitations?**
EOQ = √(2DS/H) minimises yearly ordering cost plus holding cost. It assumes constant demand, no quantity discounts, and no minimum order quantity (MOQ) or shelf-life limits. That's why I combined it with an order-up-to target and would add MOQ, case-pack rounding, and expiry constraints for perishables in a real system.

**9. How would you measure whether this works in a real company?**
Run an A/B pilot: some stores use the new policy while matched stores keep the old one, for 4–8 weeks. Track fill rate, stockout rate, inventory turns, days of cover, waste or expiry, and gross margin. Monitor forecast WAPE and bias weekly, and retrain the model monthly.

**10. How would you scale this to 50,000 SKUs and 500 stores?**
Keep the global-model approach, switch to LightGBM (faster, handles categorical features), train on a feature store in Spark or BigQuery, forecast in batch every night with Airflow, and write recommendations into the ERP or replenishment system. I would add hierarchical reconciliation (SKU → category → store) and quantile forecasts so safety stock comes directly from the P95 forecast.

**Bonus questions to prepare:** Why Random Forest over ARIMA? (One global model across 60 series, it handles exogenous features like promotions and festivals, and needs no per-series tuning.) What is the most important feature? (`normal_day_mean_28`, the festival-free baseline, with 81% importance, followed by the festival flag and same-weekday mean.)
