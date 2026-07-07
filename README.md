# Customer Churn Prediction — ML API + Dashboard

An end-to-end churn prediction project: trains a classifier, evaluates it with proper metrics, and serves predictions through both a REST API and an interactive dashboard.

## What it does

- Preprocesses data with a scikit-learn `Pipeline` (no manual leakage-prone steps)
- Trains and evaluates a classifier — Accuracy, ROC-AUC, confusion matrix
- Exports the trained model with `joblib`
- Serves predictions via a **FastAPI** endpoint
- Visualises results in a **Streamlit** dashboard for non-technical stakeholder review

## Tech stack

Python · scikit-learn · FastAPI · Streamlit · joblib

## Running locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# train the model (writes churn_model.joblib)
python src/train.py

# run the dashboard
streamlit run src/dashboard_app.py

# or run the API
uvicorn src.api:app --reload
```

## Data

A small synthetic dataset (`data/sample_churn.csv`) is included so the pipeline runs immediately. Swap it for a real dataset — the preprocessing pipeline handles it without changes.

## Project structure

```
.
├─ data/
│  └─ sample_churn.csv
├─ reports/
│  ├─ confusion_matrix.png
│  └─ roc_curve.png
├─ src/
│  ├─ train.py
│  ├─ predict.py
│  ├─ api.py
│  ├─ dashboard_app.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## What I'd improve next

- Cross-validation instead of a single train/test split
- Feature importance visualisation in the dashboard
- Model versioning if retrained on new data

---
Berke Tayfun Akseki — [berketayfunakseki.com](https://berketayfunakseki.com)
