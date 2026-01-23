from __future__ import annotations

import pandas as pd
from utils import load_model, FeatureSchema


def predict_churn(input_df: pd.DataFrame) -> pd.DataFrame:
    schema = FeatureSchema()
    model = load_model()

    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = input_df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = pred
    return out
