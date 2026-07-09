# Model Evaluation Report — Customer Churn Analysis

## Methodology

- **Train/test split:** 80/20, stratified on the target to preserve the 31.7% churn rate in both sets.
- **Cross-validation:** Stratified 5-fold on the training set, scored on ROC-AUC.
- **Feature set:** 30 features (16 numeric/engineered + 15 categorical, label-encoded), numeric features standardized with `StandardScaler` fit on the training set only (no leakage).
- **Model selection metric:** Test-set ROC-AUC (appropriate for an imbalanced binary target where ranking risk matters more than raw accuracy).

## Model Comparison

| Model | Test ROC-AUC | CV ROC-AUC (mean ± std) |
|---|---|---|
| **Logistic Regression (winner)** | **0.7835** | 0.7880 ± 0.0118 |
| Gradient Boosting | 0.7801 | 0.7840 ± 0.0105 |
| Random Forest | 0.7707 | 0.7771 ± 0.0115 |
| XGBoost | 0.7698 | 0.7661 ± 0.0093 |
| Naive Bayes | 0.7641 | 0.7700 ± 0.0087 |
| SVM | 0.7550 | 0.7480 ± 0.0057 |
| Decision Tree | 0.7522 | 0.7494 ± 0.0104 |
| K-Nearest Neighbors | 0.7384 | 0.7404 ± 0.0066 |

**Why Logistic Regression won:** The churn signal in this dataset is largely additive — contract type, complaints, tenure, and monthly charges each contribute fairly independently rather than through complex interactions. A linear model captures that well, and its narrow win over Gradient Boosting (0.0034 AUC) confirms tree-based models aren't finding much extra non-linear signal to exploit. This is a business-favorable outcome: the winning model is also the most transparent one.

## Winning Model: Detailed Metrics (Test Set, threshold = 0.5)

| Metric | Value | Business Meaning |
|---|---|---|
| Accuracy | 74.5% | Overall correct predictions |
| Precision | 63.4% | Of customers flagged as churn risks, 63.4% actually churn — controls wasted retention offers |
| Recall | 46.2% | Of customers who actually churned, 46.2% were caught in advance — controls missed saves |
| F1 Score | 53.5% | Harmonic balance of precision and recall |
| ROC-AUC | 78.4% | Probability the model ranks a random churner above a random non-churner |

### Confusion Matrix (test set, n=1,500)

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 897 (True Negative) | 127 (False Positive) |
| **Actual: Churn** | 256 (False Negative) | 220 (True Positive) |

### Classification Report

```
              precision    recall  f1-score   support

    No Churn       0.78      0.88      0.82      1024
       Churn       0.63      0.46      0.53       476

    accuracy                           0.74      1500
   macro avg       0.71      0.67      0.68      1500
weighted avg       0.73      0.74      0.73      1500
```

## Threshold Trade-off Guidance

At the default 0.5 threshold, the model misses more churners than it catches (recall 46.2%). Because a missed churner typically costs the business more than a wasted retention offer to a customer who wouldn't have left, **lowering the classification threshold (e.g. to 0.35–0.40)** would catch more true churners at the cost of more false positives — a trade-off the retention team should tune based on the actual cost of an outreach offer vs. the value of a saved customer.

## Cross-Validation Stability

Standard deviation across the 5 folds is ~0.01 for the top 4 models, indicating performance is stable and not an artifact of a lucky train/test split.

## Artifacts Produced

- `models/trained_model.pkl` — the winning Logistic Regression model
- `models/feature_scaler.pkl` — fitted StandardScaler for numeric features
- `models/label_encoders.pkl` — fitted LabelEncoders for categorical features
- `models/model_metadata.pkl` — feature column lists and model name for reproducible inference
- `models/model_comparison.csv` — full 8-model comparison table
- `images/plots/confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `correlation_heatmap.png`, `shap_summary.png`

## Limitations & Assumptions

- Dataset is synthetic (see README) — engineered patterns mirror real telecom churn dynamics but exact coefficients won't transfer to a live business without recalibration.
- The model is trained on a single snapshot in time; a production system would need periodic retraining and drift monitoring (see README's "Future Improvements").
- Recall at the default threshold (46.2%) is moderate; see the threshold guidance above before using this model to drive automated retention spend.
