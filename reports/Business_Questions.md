# Business Questions — Customer Churn Analysis

The questions this project was built to answer, organized by stakeholder.

## For Executive Leadership

1. What percentage of our customers are churning, and what is that costing us in revenue?
2. Which customer segments are most at risk, and how large is each segment?
3. What's the single highest-ROI action we could take to reduce churn?
4. Can we predict which specific customers are likely to leave before they do?

## For the Retention / Customer Success Team

5. At what point in the customer lifecycle is churn risk highest?
6. Does the number of support tickets or complaints predict churn, and if so, at what threshold should we intervene?
7. Do add-on services (tech support, security, streaming) reduce churn, and which ones matter most?
8. Are referred customers more loyal than customers acquired through other channels?

## For Product & Pricing

9. Does contract length affect churn, and by how much?
10. Is there a relationship between monthly spend and churn — are we losing our most valuable customers or our least valuable ones?
11. Why might Fiber Optic customers churn more than DSL customers despite paying for a premium product?
12. Does payment method (autopay vs. manual) influence churn?

## For Data Science / Analytics

13. Which features are the strongest predictors of churn, and do multiple independent methods (EDA, tree-based importance, SHAP) agree?
14. What model architecture best fits this problem — is churn driven by simple additive effects or complex feature interactions?
15. How should the classification threshold be tuned given the relative cost of a missed churner vs. a wasted retention offer?
16. What is the model's expected reliability (cross-validation stability) before it's trusted for production decisions?

## How This Repository Answers Each Question

| Question # | Answered In |
|---|---|
| 1, 2, 3 | `reports/Executive_Summary.md`, `reports/Business_Insights.md` |
| 4 | `src/predict.py`, `models/trained_model.pkl` |
| 5, 6, 7, 8 | `notebooks/03_exploratory_data_analysis.ipynb`, `reports/Business_Insights.md` |
| 9, 10, 11, 12 | `notebooks/03_exploratory_data_analysis.ipynb` |
| 13, 14 | `reports/Feature_Importance.md`, `notebooks/06_model_evaluation.ipynb` |
| 15, 16 | `reports/Model_Evaluation.md` |
