from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from utils import FeatureSchema, MODEL_PATH
from predict import predict_churn

app = FastAPI(title="Churn Prediction API", version="1.0")

schema = FeatureSchema()


class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    support_calls_last_month: int = Field(..., ge=0, le=20)
    contract_type: str
    internet_service: str
    payment_method: str
    senior_citizen: int = Field(..., ge=0, le=1)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Churn Prediction API is running",
        "model_file": str(MODEL_PATH.name),
    }


@app.post("/predict")
def predict(payload: ChurnRequest):
    df = pd.DataFrame([payload.model_dump()])
    out = predict_churn(df)
    return {
        "churn_probability": float(out["churn_probability"].iloc[0]),
        "churn_prediction": int(out["churn_prediction"].iloc[0]),
    }
