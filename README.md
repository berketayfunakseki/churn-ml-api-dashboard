# Customer Churn Prediction — End-to-End (ML + API + Dashboard)

A complete, portfolio-ready project that trains a **customer churn model**, evaluates it with proper metrics, and exposes predictions through:
- ✅ a **FastAPI** prediction service
- ✅ a **Streamlit** interactive dashboard

This repo is designed to look great for **Master's applications** and internships (CS + Management profile).

---

## What this project demonstrates
- Data preprocessing with a **scikit-learn Pipeline**
- Train/test split + model selection
- Metrics: **Accuracy, ROC-AUC, Confusion Matrix**
- Model export with **joblib**
- Simple MLOps-style structure (scripts + reports)
- API deployment-ready FastAPI endpoint
- Streamlit dashboard with live predictions

---

## Quickstart

### 1) Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the model (creates `churn_model.joblib`)
```bash
python src/train.py
```

### 3) Run the Streamlit dashboard
```bash
streamlit run src/dashboard_app.py
```

### 4) Run the FastAPI service
```bash
uvicorn src.api:app --reload
```

---

## Data
A small **synthetic dataset** is included in `data/sample_churn.csv` so the project runs immediately.
You can replace it with a real dataset later — the pipeline is ready.

---

## Project Structure
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

---

## Author
Berke Tayfun Akseki — LUISS (CS & Management)

