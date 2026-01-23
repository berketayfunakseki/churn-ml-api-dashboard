from __future__ import annotations

import streamlit as st
import pandas as pd
from predict import predict_churn
from utils import FeatureSchema

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.caption("A simple ML demo: input customer info → get churn probability.")

schema = FeatureSchema()

with st.form("churn_form"):
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges (€)", 0.0, 150.0, 55.0, step=1.0)
    support_calls = st.slider("Support calls (last month)", 0, 10, 1)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"])
    payment = st.selectbox("Payment method", ["Credit card", "Bank transfer", "Electronic check", "Mailed check"])
    senior = st.selectbox("Senior citizen", [0, 1])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_calls_last_month": support_calls,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
    }
    df = pd.DataFrame([row])
    result = predict_churn(df)

    proba = float(result["churn_probability"].iloc[0])
    pred = int(result["churn_prediction"].iloc[0])

    st.subheader("Result")
    st.metric("Churn probability", f"{proba:.2%}")
    st.write("Prediction:", "⚠️ Likely to churn" if pred == 1 else "✅ Likely to stay")
