# Interview Q&A — Customer Churn Analysis Project

Tailored to this specific project's decisions, not generic ML trivia.

---

**Q1. Walk me through this project end to end.**

I followed CRISP-DM on a synthetic-but-realistic telecom churn dataset of 7,500 customers. I started with data understanding (shape, missingness, duplicates), cleaned the data with documented, justified decisions, engineered 9 business-meaningful features, trained and compared 8 classification algorithms with stratified cross-validation, evaluated the winner with metrics translated into business language, and closed with 31 grounded insights and a prioritized action plan tied to real dollar figures — $205K in monthly revenue and $8.79M in CLV at risk.

**Q2. Why did you generate a synthetic dataset instead of using a public one like Telco Customer Churn from Kaggle?**

Two reasons. First, it avoids any licensing ambiguity — I own every row. Second, and more importantly, I could engineer specific, known business patterns into the data (e.g., a real logistic relationship between contract type, complaints, tenure, and churn probability) rather than inherit whatever noise happens to exist in a public dataset. That means every insight I report is traceable to a mechanism I built and can defend under scrutiny in an interview.

**Q3. Why did Logistic Regression outperform XGBoost here?**

Because the churn signal in this data is largely additive — contract type, complaints, and tenure each contribute somewhat independently rather than through complex multiplicative interactions. Logistic Regression's 0.7835 test AUC barely edged out Gradient Boosting's 0.7801, which tells me the tree-based models weren't finding much extra non-linear signal to exploit. In a real engagement, I'd take that as a signal the linear model is a defensible choice — it's also the most explainable one, which matters when a retention team needs to understand *why* a customer was flagged.

**Q4. Why ROC-AUC as the primary selection metric instead of accuracy?**

Churn is imbalanced (31.7% positive class), so a model that just predicts "no churn" for everyone gets 68% accuracy while being useless. ROC-AUC measures whether the model correctly ranks churners above non-churners across all thresholds, which is the property that actually matters for a ranked retention-outreach list.

**Q5. Your recall is only 46%. Isn't that a problem?**

It's a real limitation, and I called it out directly in `Model_Evaluation.md` rather than hiding it. At the default 0.5 threshold the model misses more than half of actual churners. Because a missed churner usually costs more than a wasted retention offer, I'd recommend lowering the threshold in production — trading some precision for higher recall — and I documented that trade-off explicitly instead of just reporting the headline AUC number.

**Q6. Explain the RiskScore feature — isn't that circular, since it's built from the same data as the target?**

RiskScore is a heuristic, not a leak — it's built purely from known business rules (contract type, payment method, complaint count, tenure, tech support), with no reference to the actual Churn label at construction time. It turned out to be the single most important feature to the Random Forest, which is actually a useful finding: it shows a transparent business rule captures most of the model's signal, which matters for stakeholder trust.

**Q7. How did you validate that your engineered features actually help, rather than just adding noise?**

For each engineered feature I checked group-level churn rates or correlations before trusting it — e.g., I confirmed churned customers average a RiskScore of 33.3 vs. 16.1 for retained customers, and that AvgMonthlyRevenue has much lower collinearity with Tenure than raw TotalCharges does (which was the whole point of engineering it). That validation step is in `notebooks/04_feature_engineering.ipynb`.

**Q8. How would you deploy this in production?**

`src/predict.py` is already structured as the inference entry point — it loads the persisted model, scaler, and label encoders and scores new records. The next steps would be wrapping that in a lightweight API (FastAPI), adding drift monitoring since customer behavior shifts over time, and scheduling periodic retraining. I outlined these as tiered bonus features in the README (Streamlit dashboard for beginners, FastAPI + Docker for intermediate, MLflow + CI/CD + cloud deployment for advanced).

**Q9. What would you do differently with more time or real data?**

I'd add calibration analysis (are predicted probabilities well-calibrated, not just well-ranked), test cost-sensitive learning to directly optimize for the retention-offer-vs-missed-churner trade-off, and validate whether the engineered RiskScore weights should be re-estimated rather than hand-set once real historical retention-campaign outcomes are available.

**Q10. What's the single most actionable insight from this whole project?**

Contract length. The gap between month-to-month (44.3% churn) and two-year contracts (11.7% churn) is the largest single effect in the entire dataset, and it's also the easiest lever to pull — a modest incentive to convert month-to-month customers to annual plans is the highest-ROI recommendation in `Executive_Summary.md`.

---

## Common Mistakes to Avoid When Discussing This Project

- Don't claim the model "predicts churn with 78% accuracy" — 0.78 is the ROC-AUC, not accuracy (which is 74.5%). Mixing these up is an immediate red flag to an interviewer.
- Don't present RiskScore and EngagementScore as if they came from the ML model — they're hand-built heuristics that happen to also be useful model inputs.
- Don't overstate the dataset as "real telecom data" — be upfront that it's synthetic with engineered patterns, and explain why that was a deliberate choice, not a limitation you're hiding.

## How to Explain This Project in 60 Seconds

"I built an end-to-end churn analysis on a 7,500-customer telecom dataset — from data cleaning through a 26-analysis EDA, feature engineering, an 8-model comparison, and a full evaluation with SHAP explainability. The headline finding is that contract type, internet service tier, and complaint frequency are the three biggest churn drivers, worth $205K a month in at-risk revenue. The final model hits 0.78 ROC-AUC, and I translated every technical result into a specific, prioritized business recommendation rather than stopping at the model metrics."
