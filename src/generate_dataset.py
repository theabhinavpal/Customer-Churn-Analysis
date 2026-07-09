"""
generate_dataset.py
--------------------
Generates a realistic, synthetic telecom/subscription customer churn dataset
with deliberately engineered business patterns (not random noise):

  * Month-to-month contracts churn far more than one/two-year contracts
  * High monthly charges + low tenure => elevated churn risk
  * Frequent support tickets / complaints => elevated churn risk
  * Electronic check payment users churn more than auto-pay users
  * Senior citizens with no tech support churn more
  * Long-tenure, high-CLV customers are "sticky" (low churn)
  * Fiber optic internet customers churn more than DSL (price sensitivity)
  * Customers with add-on services (security/backup/protection) are stickier

The dataset is fully synthetic -> no copyright/privacy concerns, and every
downstream insight can be traced back to a rule implemented here, so it is
defensible in an interview setting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N = 7500


def generate() -> pd.DataFrame:
    customer_id = [f"CUST-{100000+i}" for i in range(N)]

    gender = RNG.choice(["Male", "Female"], size=N, p=[0.505, 0.495])

    # Age skewed toward working-age adults, with a senior tail
    age = np.clip(RNG.normal(41, 14, N), 18, 85).astype(int)
    senior_citizen = (age >= 65).astype(int)

    marital_status = RNG.choice(["Single", "Married", "Divorced", "Widowed"],
                                 size=N, p=[0.38, 0.42, 0.14, 0.06])
    dependents = np.where(
        marital_status == "Married",
        RNG.choice(["Yes", "No"], size=N, p=[0.55, 0.45]),
        RNG.choice(["Yes", "No"], size=N, p=[0.18, 0.82]),
    )

    # Tenure (months) - Pareto-ish: many short-tenure, long tail of loyal customers
    tenure = np.clip(RNG.exponential(24, N), 0, 72).astype(int)

    contract_type = RNG.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=N, p=[0.55, 0.24, 0.21]
    )
    # Longer contracts correlate with longer tenure (real-world pattern)
    tenure = np.where(contract_type == "Two Year", np.clip(tenure + RNG.integers(12, 40, N), 0, 72), tenure)
    tenure = np.where(contract_type == "One Year", np.clip(tenure + RNG.integers(4, 20, N), 0, 72), tenure)

    internet_service = RNG.choice(["DSL", "Fiber Optic", "No"], size=N, p=[0.34, 0.44, 0.22])
    phone_service = RNG.choice(["Yes", "No"], size=N, p=[0.90, 0.10])

    def yes_no_dependent_on_internet(p_yes):
        out = np.where(internet_service == "No", "No Internet Service",
                        RNG.choice(["Yes", "No"], size=N, p=[p_yes, 1 - p_yes]))
        return out

    streaming_tv = yes_no_dependent_on_internet(0.45)
    streaming_movies = yes_no_dependent_on_internet(0.44)
    online_security = yes_no_dependent_on_internet(0.32)
    online_backup = yes_no_dependent_on_internet(0.35)
    device_protection = yes_no_dependent_on_internet(0.34)
    tech_support = yes_no_dependent_on_internet(0.30)

    payment_method = RNG.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer (Auto)", "Credit Card (Auto)"],
        size=N, p=[0.34, 0.19, 0.24, 0.23],
    )
    paperless_billing = RNG.choice(["Yes", "No"], size=N, p=[0.59, 0.41])

    # Base monthly charge driven by internet type + add-ons
    base = np.select(
        [internet_service == "Fiber Optic", internet_service == "DSL", internet_service == "No"],
        [79.0, 56.0, 21.0],
    )
    addon_cost = (
        (streaming_tv == "Yes") * 9.5 + (streaming_movies == "Yes") * 9.5 +
        (online_security == "Yes") * 5.5 + (online_backup == "Yes") * 5.0 +
        (device_protection == "Yes") * 5.0 + (tech_support == "Yes") * 6.0 +
        (phone_service == "Yes") * 6.5
    )
    monthly_charges = np.round(base + addon_cost + RNG.normal(0, 3.5, N), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 130.0)

    total_charges = np.round(monthly_charges * tenure * RNG.uniform(0.94, 1.02, N), 2)

    # Support tickets & complaints: higher for fiber + month-to-month (frustration proxy)
    ticket_lambda = 1.0 + (internet_service == "Fiber Optic") * 1.2 + (contract_type == "Month-to-Month") * 0.8
    support_tickets = RNG.poisson(ticket_lambda)
    complaints = np.minimum(support_tickets, RNG.poisson(0.4 + support_tickets * 0.15))

    satisfaction = np.clip(
        np.round(RNG.normal(3.6 - complaints * 0.25 - (contract_type == "Month-to-Month") * 0.2, 0.7)), 1, 5
    ).astype(int)

    referral_status = RNG.choice(["Yes", "No"], size=N, p=[0.27, 0.73])

    today = datetime(2026, 6, 30)
    last_interaction_days_ago = RNG.integers(1, 400, N)
    last_interaction_date = [today - timedelta(days=int(d)) for d in last_interaction_days_ago]

    clv = np.round(monthly_charges * (tenure + RNG.normal(18, 8, N)).clip(min=1) * RNG.uniform(0.9, 1.15, N), 2)

    # ---- Churn probability model (this IS the ground truth pattern) ----
    logit = (
        -2.55
        + 1.55 * (contract_type == "Month-to-Month")
        + 0.55 * (contract_type == "One Year")
        + 0.70 * (internet_service == "Fiber Optic")
        - 0.85 * (tenure / 72)
        + 0.014 * (monthly_charges - 60)
        + 0.28 * complaints
        + 0.16 * support_tickets
        + 0.55 * (payment_method == "Electronic Check")
        - 0.35 * (tech_support == "Yes")
        - 0.30 * (online_security == "Yes")
        - 0.20 * (device_protection == "Yes")
        + 0.30 * (senior_citizen == 1) * (tech_support != "Yes")
        - 0.25 * (referral_status == "Yes")
        - 0.18 * (satisfaction - 3)
        + RNG.normal(0, 0.55, N)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(0, 1, N) < prob_churn).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Gender": gender,
        "Age": age,
        "SeniorCitizen": senior_citizen,
        "MaritalStatus": marital_status,
        "Dependents": dependents,
        "Tenure": tenure,
        "ContractType": contract_type,
        "InternetService": internet_service,
        "PhoneService": phone_service,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "NumberOfComplaints": complaints,
        "CustomerSatisfactionScore": satisfaction,
        "SupportTickets": support_tickets,
        "LastInteractionDate": [d.strftime("%Y-%m-%d") for d in last_interaction_date],
        "ReferralStatus": referral_status,
        "CustomerLifetimeValue": clv,
        "Churn": churn_label,
    })

    # Inject a small, realistic amount of messiness for the cleaning notebook/script to fix
    dirty_idx = RNG.choice(N, size=int(N * 0.015), replace=False)
    df.loc[dirty_idx[: len(dirty_idx)//3], "TotalCharges"] = np.nan
    df.loc[dirty_idx[len(dirty_idx)//3: 2*len(dirty_idx)//3], "Gender"] = df.loc[dirty_idx[len(dirty_idx)//3: 2*len(dirty_idx)//3], "Gender"].str.upper()
    dup_idx = RNG.choice(N, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config

    data = generate()
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.RAW_DATA_PATH, index=False)
    print("Rows:", len(data))
    print(data["Churn"].value_counts(normalize=True))
