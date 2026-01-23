from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from utils import FeatureSchema, load_data, save_model, MODEL_PATH, REPORTS_DIR


def build_pipeline(schema: FeatureSchema) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(schema.numeric)),
            ("cat", categorical_transformer, list(schema.categorical)),
        ]
    )

    clf = LogisticRegression(max_iter=2000)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return model


def main() -> None:
    schema = FeatureSchema()
    df = load_data()

    X = df[list(schema.numeric) + list(schema.categorical)]
    y = df[schema.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = build_pipeline(schema)
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ ROC-AUC:  {auc:.3f}")

    # Save model
    save_model(model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

    # Save plots
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    print(f"📊 Plots saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
