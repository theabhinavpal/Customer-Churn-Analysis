"""
train_model.py
----------------
Builds the preprocessing + modeling pipeline: encodes categoricals, scales
numerics, trains and compares several classifiers, selects the best one by
ROC-AUC on a held-out test set, and persists the winning model + scaler.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import config
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "Age", "SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges",
    "NumberOfComplaints", "CustomerSatisfactionScore", "SupportTickets",
    "CustomerLifetimeValue", "ServiceCount", "AvgMonthlyRevenue",
    "ComplaintFrequency", "EngagementScore", "RiskScore", "HighValueCustomer",
    "LongTermCustomer",
    "Gender", "MaritalStatus", "Dependents", "ContractType", "InternetService",
    "PhoneService", "StreamingTV", "StreamingMovies", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "PaymentMethod",
    "PaperlessBilling", "ReferralStatus",
]

CAT_COLS = [
    "Gender", "MaritalStatus", "Dependents", "ContractType", "InternetService",
    "PhoneService", "StreamingTV", "StreamingMovies", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "PaymentMethod",
    "PaperlessBilling", "ReferralStatus",
]

NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def build_model_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    label_encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    y = (df["Churn"] == "Yes").astype(int)
    return X, y, label_encoders


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=config.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE,
        ),
        "SVM": SVC(probability=True, random_state=config.RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
    }


def run_training():
    df = pd.read_csv(config.FEATURED_DATA_PATH)
    X, y, encoders = build_model_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    models = get_candidate_models()
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="roc_auc")
        results[name] = {
            "model": model,
            "test_auc": test_auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
        }
        logger.info(f"{name}: Test AUC={test_auc:.4f} | CV AUC={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_model = results[best_name]["model"]
    logger.info(f"Best model: {best_name} (Test AUC={results[best_name]['test_auc']:.4f})")

    utils.save_object(best_model, config.MODEL_PATH)
    utils.save_object(scaler, config.SCALER_PATH)
    utils.save_object(encoders, config.MODELS_DIR / "label_encoders.pkl")
    utils.save_object({"feature_cols": FEATURE_COLS, "cat_cols": CAT_COLS, "num_cols": NUM_COLS, "best_model_name": best_name},
                       config.MODELS_DIR / "model_metadata.pkl")

    return {
        "results": results,
        "best_name": best_name,
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
    }


if __name__ == "__main__":
    run_training()
