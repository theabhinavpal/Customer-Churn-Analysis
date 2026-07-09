"""
predict.py
----------
Loads the trained model + preprocessing artifacts and scores new customer
records. Intended as the entry point for batch scoring or a future API layer.
"""

import pandas as pd
import config
import utils
from train_model import FEATURE_COLS, CAT_COLS, NUM_COLS


def load_artifacts():
    model = utils.load_object(config.MODEL_PATH)
    scaler = utils.load_object(config.SCALER_PATH)
    encoders = utils.load_object(config.MODELS_DIR / "label_encoders.pkl")
    return model, scaler, encoders


def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of customers (already feature-engineered) and return
    churn probability + label."""
    model, scaler, encoders = load_artifacts()

    X = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        le = encoders[col]
        X[col] = X[col].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
        X[col] = le.transform(X[col])

    X[NUM_COLS] = scaler.transform(X[NUM_COLS])

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["ChurnProbability"] = proba.round(4)
    out["ChurnPrediction"] = pred
    return out


if __name__ == "__main__":
    sample = pd.read_csv(config.FEATURED_DATA_PATH).sample(10, random_state=1)
    scored = predict_churn(sample)
    print(scored[["CustomerID", "Churn", "ChurnProbability", "ChurnPrediction"]])
