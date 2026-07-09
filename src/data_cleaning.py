"""
data_cleaning.py
----------------
Deterministic, documented cleaning steps applied to the raw churn dataset.

Cleaning decisions (and why):
  1. Standardize `Gender` casing (raw data has some ALL-CAPS entries) -> keeps
     downstream categorical encoding from creating duplicate categories.
  2. Drop exact duplicate customer rows (25 injected duplicates simulate a
     common real-world export issue: same customer pulled twice from source
     systems).
  3. Impute missing `TotalCharges` using Tenure * MonthlyCharges, which is a
     domain-accurate estimate rather than a generic mean/median fill.
  4. Cast `SeniorCitizen` to a clean 0/1 int and `Churn` to a clean Yes/No.
  5. Detect (but do not blindly drop) outliers in MonthlyCharges/Tenure using
     the IQR method; these are business-plausible (e.g., high-tier bundles)
     so we flag rather than remove them.
  6. Enforce logical data types across the frame.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
    df["Gender"] = df["Gender"].str.strip().str.title()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "CustomerID"] + ["CustomerID"])
    # also drop rows that are full duplicates of another CustomerID's record (injected dupes keep same ID)
    df = df.drop_duplicates(subset="CustomerID", keep="first")
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def impute_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    missing_before = df["TotalCharges"].isna().sum()
    est = df["Tenure"] * df["MonthlyCharges"]
    df["TotalCharges"] = df["TotalCharges"].fillna(est)
    logger.info(f"Imputed {missing_before} missing TotalCharges values using Tenure x MonthlyCharges")
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["Age"] = df["Age"].astype(int)
    df["Tenure"] = df["Tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df["NumberOfComplaints"] = df["NumberOfComplaints"].astype(int)
    df["SupportTickets"] = df["SupportTickets"].astype(int)
    df["CustomerSatisfactionScore"] = df["CustomerSatisfactionScore"].astype(int)
    df["CustomerLifetimeValue"] = df["CustomerLifetimeValue"].astype(float)
    df["LastInteractionDate"] = pd.to_datetime(df["LastInteractionDate"])
    df["Churn"] = df["Churn"].str.strip().str.title()
    return df


def flag_outliers_iqr(df: pd.DataFrame, cols=("MonthlyCharges", "Tenure", "CustomerLifetimeValue")) -> pd.DataFrame:
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag_col = f"{col}_Outlier"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        n_flagged = df[flag_col].sum()
        logger.info(f"{col}: flagged {n_flagged} outliers (IQR bounds [{lower:.2f}, {upper:.2f}])")
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence and return an analysis-ready DataFrame."""
    df = df.copy()
    df = standardize_gender(df)
    df = drop_duplicates(df)
    df = impute_total_charges(df)
    df = enforce_dtypes(df)
    df = flag_outliers_iqr(df)

    # Final sanity checks
    assert df["Churn"].isin(["Yes", "No"]).all(), "Unexpected Churn label found"
    assert df["TotalCharges"].isna().sum() == 0, "TotalCharges still has missing values"
    assert df["CustomerID"].is_unique, "Duplicate CustomerID remains after cleaning"

    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df
