"""
config.py
---------
Central configuration: file paths, column groups, and modeling constants.
Keeping these in one place avoids magic strings scattered across the pipeline.
"""

from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images" / "plots"

RAW_DATA_PATH = DATASET_DIR / "customer_churn.csv"
CLEAN_DATA_PATH = DATASET_DIR / "customer_churn_clean.csv"
FEATURED_DATA_PATH = DATASET_DIR / "customer_churn_features.csv"

MODEL_PATH = MODELS_DIR / "trained_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# ---- Columns ----
TARGET = "Churn"
ID_COL = "CustomerID"

CATEGORICAL_COLS = [
    "Gender", "MaritalStatus", "Dependents", "ContractType", "InternetService",
    "PhoneService", "StreamingTV", "StreamingMovies", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "PaymentMethod",
    "PaperlessBilling", "ReferralStatus",
]

NUMERIC_COLS = [
    "Age", "SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges",
    "NumberOfComplaints", "CustomerSatisfactionScore", "SupportTickets",
    "CustomerLifetimeValue",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
