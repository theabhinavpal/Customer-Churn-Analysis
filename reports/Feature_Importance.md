# Feature Importance Report — Customer Churn Analysis

Three independent methods were used to identify churn drivers, and all three converge on the same top features — strong evidence the model has learned genuine business patterns.

## 1. Random Forest Feature Importance (Gini-based)

Computed in `notebooks/06_model_evaluation.ipynb`, saved to `models/feature_importance.csv`.

| Rank | Feature | Importance |
|---|---|---|
| 1 | RiskScore | 0.195 |
| 2 | MonthlyCharges | 0.090 |
| 3 | AvgMonthlyRevenue | 0.086 |
| 4 | ContractType | 0.066 |
| 5 | TotalCharges | 0.059 |
| 6 | CustomerLifetimeValue | 0.059 |
| 7 | Tenure | 0.054 |
| 8 | SupportTickets | 0.048 |
| 9 | Age | 0.047 |
| 10 | ComplaintFrequency | 0.046 |
| 11 | EngagementScore | 0.037 |
| 12 | ServiceCount | 0.021 |
| 13 | InternetService | 0.020 |
| 14 | PaymentMethod | 0.019 |
| 15 | MaritalStatus | 0.018 |

*(Full ranking of all 30 features in `models/feature_importance.csv`.)*

## 2. Permutation Importance

Directionally consistent with the Gini-based ranking above: shuffling `RiskScore`, `MonthlyCharges`, or `ContractType` causes the largest drop in model ROC-AUC, confirming these aren't just frequently-split-on features but features the model genuinely depends on for correct ranking.

## 3. SHAP Values (Logistic Regression — the deployed model)

Computed via `shap.LinearExplainer` in `notebooks/06_model_evaluation.ipynb` (summary plot saved to `images/plots/shap_summary.png`).

- **RiskScore, MonthlyCharges, and month-to-month ContractType** push predictions toward churn.
- **Tenure, TechSupport = Yes, OnlineSecurity = Yes, and ReferralStatus = Yes** push predictions away from churn.
- Complaint-related features (`NumberOfComplaints`, `ComplaintFrequency`) consistently push toward churn, with effect size increasing at higher complaint counts.

## Business Interpretation

The engineered `RiskScore` — a simple, fully explainable heuristic built from known churn drivers (contract type, payment method, complaints, tenure, tech support) — is the single most important input to the model. This is a useful finding on its own: it means a large share of the model's predictive power can be reproduced with a transparent business rule, which matters for stakeholder trust and for explaining individual predictions to frontline retention staff without invoking "black box" ML.

The remaining lift the ML model provides over the raw RiskScore comes primarily from `MonthlyCharges`, `AvgMonthlyRevenue`, and `CustomerLifetimeValue` — the billing/value dimension — which the heuristic score does not directly encode.

## Top Churn Drivers, Ranked by Real-World Effect Size

(Cross-referenced against `Business_Insights.md` for the underlying percentages.)

1. Contract type (Month-to-Month vs Two Year): 44.3% vs 11.7% churn
2. Internet service (Fiber Optic vs DSL): 46.3% vs 22.2% churn
3. Complaint count (3 vs 0): 63.4% vs 25.0% churn
4. Tenure group (0-6 months vs 49+ months): 51.2% vs 19.5% churn
5. Payment method (Electronic Check vs Auto-pay): 37.2% vs ~28.5% churn
