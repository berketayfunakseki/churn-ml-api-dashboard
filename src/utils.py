from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "sample_churn.csv"
MODEL_PATH = ROOT / "churn_model.joblib"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class FeatureSchema:
    numeric: tuple[str, ...] = (
        "tenure_months",
        "monthly_charges",
        "support_calls_last_month",
        "senior_citizen",
    )
    categorical: tuple[str, ...] = (
        "contract_type",
        "internet_service",
        "payment_method",
    )
    target: str = "churn"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(model, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)
