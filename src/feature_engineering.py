"""
feature_engineering.py
-----------------------
Creates business-meaningful derived features on top of the cleaned dataset.

Each feature is included because it captures a pattern a raw column can't:
  * TenureGroup            -> churn risk is non-linear across tenure (new vs loyal)
  * MonthlySpendCategory   -> price-sensitivity segments for marketing/pricing teams
  * HighValueCustomer      -> flags customers worth prioritizing in retention efforts
  * LongTermCustomer       -> simple loyalty flag (>= 24 months)
  * ServiceCount           -> more bundled services = more "switching friction"
  * AvgMonthlyRevenue      -> normalizes TotalCharges by tenure for a spend rate view
  * ComplaintFrequency     -> complaints per year of tenure (rate, not raw count)
  * EngagementScore        -> composite of satisfaction, referrals, and low complaints
  * RiskScore              -> composite heuristic (0-100) combining known churn drivers,
                              useful as a fast, explainable score alongside the ML model
"""

import numpy as np
import pandas as pd


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 6, 12, 24, 48, 1000]
    labels = ["0-6 Months", "7-12 Months", "13-24 Months", "25-48 Months", "49+ Months"]
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins, labels=labels)
    return df


def add_monthly_spend_category(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 35, 65, 95, 1000]
    labels = ["Budget", "Standard", "Premium", "Elite"]
    df["MonthlySpendCategory"] = pd.cut(df["MonthlyCharges"], bins=bins, labels=labels)
    return df


def add_high_value_flag(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["CustomerLifetimeValue"].quantile(0.75)
    df["HighValueCustomer"] = (df["CustomerLifetimeValue"] >= threshold).astype(int)
    return df


def add_long_term_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["LongTermCustomer"] = (df["Tenure"] >= 24).astype(int)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    service_cols = [
        "PhoneService", "StreamingTV", "StreamingMovies", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
    ]
    df["ServiceCount"] = df[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
    return df


def add_avg_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df["AvgMonthlyRevenue"] = df["TotalCharges"] / df["Tenure"].replace(0, 1)
    return df


def add_complaint_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df["ComplaintFrequency"] = df["NumberOfComplaints"] / (df["Tenure"].replace(0, 1) / 12)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    referral_pts = (df["ReferralStatus"] == "Yes").astype(int) * 15
    satisfaction_pts = df["CustomerSatisfactionScore"] * 15
    complaint_penalty = np.minimum(df["NumberOfComplaints"] * 10, 40)
    score = referral_pts + satisfaction_pts - complaint_penalty
    df["EngagementScore"] = np.clip(score, 0, 100)
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Explainable 0-100 heuristic risk score (independent of the ML model),
    useful for a quick business-facing scorecard alongside model predictions."""
    score = (
        (df["ContractType"] == "Month-to-Month").astype(int) * 25
        + (df["PaymentMethod"] == "Electronic Check").astype(int) * 12
        + (df["InternetService"] == "Fiber Optic").astype(int) * 12
        + np.minimum(df["NumberOfComplaints"] * 6, 24)
        + np.minimum(df["SupportTickets"] * 3, 15)
        - np.minimum(df["Tenure"] / 3, 20)
        - (df["TechSupport"] == "Yes").astype(int) * 8
        - (df["OnlineSecurity"] == "Yes").astype(int) * 6
    )
    df["RiskScore"] = np.clip(score, 0, 100).round(1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_tenure_group(df)
    df = add_monthly_spend_category(df)
    df = add_high_value_flag(df)
    df = add_long_term_flag(df)
    df = add_service_count(df)
    df = add_avg_monthly_revenue(df)
    df = add_complaint_frequency(df)
    df = add_engagement_score(df)
    df = add_risk_score(df)
    return df
