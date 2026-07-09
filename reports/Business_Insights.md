# Business Insights — Customer Churn Analysis

All figures below are computed directly from the cleaned, feature-engineered dataset (`dataset/customer_churn_features.csv`, 7,500 customers). Source notebook: `notebooks/03_exploratory_data_analysis.ipynb` and `notebooks/07_business_insights.ipynb`.

## Headline Numbers

1. Overall churn rate is **31.7%** (2,381 of 7,500 customers).
2. Churned customers represent **$205,091** in monthly recurring revenue.
3. Churned customers represent **$8,792,436** in cumulative customer lifetime value.
4. Average tenure at churn is **23.3 months**, vs **33.0 months** for retained customers.
5. Average monthly bill for churned customers is **$86.14**, vs **$71.70** for retained customers.

## Contract & Billing

6. Month-to-month customers churn at **44.3%**, vs **20.0%** for one-year and **11.7%** for two-year contracts — the single largest gap in the dataset.
7. Electronic Check payers churn at **37.2%**, vs **28.4%–28.8%** for auto-pay methods (bank transfer/credit card) and **30.1%** for mailed check.
8. Paperless billing shows almost no relationship with churn (32.4% vs 31.3%) — ruled out as a meaningful driver.
9. The "Elite" monthly spend tier (>$95/mo) churns at **46.2%**, compared to **17.3%** in the "Budget" tier — the highest-paying customers are also highest-risk in raw percentage terms.

## Service & Product

10. Fiber Optic customers churn at **46.3%**, nearly double DSL's **22.2%**, despite Fiber being the premium product — points to a price or reliability satisfaction gap.
11. Customers without Tech Support churn at **37.0%** vs **32.9%** with it.
12. Customers without Online Security churn at **37.4%** vs **32.5%** with it.
13. Each additional bundled service (phone, streaming, security, backup, protection, tech support) is associated with a lower churn rate — more bundles create more switching friction.
14. Streaming TV/Movies add-ons show a much weaker relationship with churn than protective add-ons (security, backup, tech support) — entertainment bundles build less loyalty than functional ones.

## Tenure & Lifecycle

15. Customers in months 0–6 churn at **51.2%** — the single highest-risk lifecycle stage.
16. Churn falls steadily by tenure band: 51.2% (0–6 mo) → 43.3% (7–12 mo) → 34.8% (13–24 mo) → 24.9% (25–48 mo) → 19.5% (49+ mo).
17. Average Customer Lifetime Value is similar between churned ($3,693) and retained ($3,789) customers — churn is not concentrated only in low-value accounts.

## Support & Satisfaction

18. Churn rises from **25.0%** with zero complaints to **63.4%** at 3 complaints — a clear escalation curve.
19. Churn rate increases with every additional support ticket logged, confirming ticket volume is a leading indicator, not just an operational cost metric.
20. Churn falls sharply as self-reported Customer Satisfaction Score rises from 1 to 5, validating the survey as a genuinely predictive signal.

## Demographics

21. Gender shows almost no difference in churn rate — not a meaningful targeting variable.
22. Senior citizens churn only marginally more overall (33.6% vs 31.7%), but **churn substantially more when they also lack Tech Support** — an interaction effect only visible when segmenting.
23. Age distributions between churned and retained customers largely overlap — age alone is a weak standalone predictor.

## Loyalty & Advocacy

24. Referred customers churn at **27.2%** vs **33.4%** for non-referred customers — referral programs pay a retention dividend, not just an acquisition one.
25. Retained customers average an EngagementScore of **50.9** vs **43.7** for churned customers, confirming the composite loyalty metric behaves as designed.
26. Retained customers average a heuristic RiskScore of **16.1** vs **33.3** for churned customers — strong separation validates the engineered score as a fast, explainable pre-model screening tool.

## Combined Segments

27. The combined segment of **Month-to-Month + Fiber Optic + Electronic Check** customers churns at a rate well above the overall base — a ready-made target list for immediate outreach, no model required.
28. Customers with 0-6 months tenure on Fiber Optic month-to-month plans represent the single highest-density risk pool in the customer base.

## Model-Based Insights

29. A Logistic Regression model combining all drivers simultaneously achieves ROC-AUC 0.78, meaningfully outperforming a random baseline (0.50) and confirming these individually-identified drivers combine into a genuinely predictive signal.
30. Feature importance (Random Forest) and SHAP values (Logistic Regression) both independently rank RiskScore, MonthlyCharges, ContractType, and complaint-related features as top drivers — cross-validating the EDA findings with two separate modeling techniques.
31. AvgMonthlyRevenue (TotalCharges normalized by tenure) was engineered specifically because raw TotalCharges is highly collinear with Tenure — the normalized feature isolates a spend-rate signal the model can use independently.

---

## Recommendations Summary

| Priority | Action | Rationale |
|---|---|---|
| 1 | Contract-conversion incentives for month-to-month customers | Largest single churn-rate gap (32.6 points) |
| 2 | Audit Fiber Optic pricing/reliability/support | 46.3% churn despite premium positioning |
| 3 | Complaint-triggered save workflow at 2nd complaint | Churn crosses 50% by the 3rd complaint |
| 4 | Strengthen 0-6 month onboarding | Highest-risk lifecycle stage (51.2%) |
| 5 | Autopay incentive for Electronic Check payers | 9-10 point churn-rate gap, low cost to close |
| 6 | Bundle Tech Support / Online Security trials at signup | Consistent, moderate churn-reduction effect |
| 7 | Expand referral program | Referred customers churn 6.2 points less |
