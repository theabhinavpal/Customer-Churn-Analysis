# Executive Summary — Customer Churn Analysis

**Prepared for:** Executive & Customer Retention Leadership
**Dataset:** 7,500 customer records (telecom/subscription business, synthetic)
**Approach:** CRISP-DM — data understanding through model deployment readiness

---

## The Problem

**31.7% of customers churned** in the observed period — nearly 1 in 3. This represents:

- **$205,091** in monthly recurring revenue at risk
- **$8,792,436** in cumulative customer lifetime value at risk

Churn is not evenly distributed — it concentrates heavily in a few identifiable, actionable segments.

## The Three Biggest Drivers

| Driver | High-Risk Segment | Churn Rate | Low-Risk Segment | Churn Rate |
|---|---|---|---|---|
| Contract length | Month-to-Month | 44.3% | Two Year | 11.7% |
| Internet service | Fiber Optic | 46.3% | DSL | 22.2% |
| Support complaints | 3 complaints | 63.4% | 0 complaints | 25.0% |

Tenure compounds all of these: customers in their **first 6 months churn at 51.2%**, versus **19.5%** for customers past 4 years.

## What the Model Adds

A Logistic Regression model, selected from 8 candidate algorithms after cross-validated comparison, achieves a **ROC-AUC of 0.78** — meaning it reliably ranks at-risk customers above safe ones. Because the winning model is linear, its outputs double as an explainable risk score the retention team can act on directly, without a "black box" objection from stakeholders.

## Recommended Next Steps

1. Launch a contract-conversion incentive targeting month-to-month customers.
2. Audit the Fiber Optic customer experience (pricing, reliability, support).
3. Build a complaint-count trigger that flags customers for proactive outreach at their 2nd complaint.
4. Strengthen the first-6-month onboarding journey.
5. Shift Electronic Check payers toward autopay with a small incentive.

Full detail, methodology, and supporting numbers are in `Business_Insights.md`, `Model_Evaluation.md`, and the notebooks in `/notebooks`.
