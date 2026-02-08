"""
Training utilities for supervised fertilizer models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "multi_supervised.csv"


def train_models(data_path: str | Path = DATA_PATH) -> dict:
    """
    Train classifier and regressor from a CSV dataset.

    Parameters
    ----------
    data_path : str | Path
        Path to the training CSV (must include required columns).

    Returns
    -------
    dict
        Metrics dictionary (train/val/test).
    """
    data_path = Path(data_path)

    df = pd.read_csv(data_path)

    need = [
        "crop", "stage", "soilN", "soilP", "soilK", "ph",
        "fertilizer_type", "fertilizer_amount_g"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


    df["fertilizer_type"] = df["fertilizer_type"].astype(str)
    df["fertilizer_amount_g"] = pd.to_numeric(df["fertilizer_amount_g"], errors="coerce")


    df = df.dropna(subset=["fertilizer_amount_g"]).copy()

    X_type = df[["crop", "stage", "soilN", "soilP", "soilK", "ph"]]
    y_type = df["fertilizer_type"]

    X_amount = df[["crop", "stage", "soilN", "soilP", "soilK", "ph", "fertilizer_type"]]
    y_amount = df["fertilizer_amount_g"].astype(float)

    idx_train, idx_temp = train_test_split(
        df.index, test_size=0.30, random_state=42, stratify=y_type
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42, stratify=y_type.loc[idx_temp]
    )

    X_train_t, X_val_t, X_test_t = X_type.loc[idx_train], X_type.loc[idx_val], X_type.loc[idx_test]
    y_train_t, y_val_t, y_test_t = y_type.loc[idx_train], y_type.loc[idx_val], y_type.loc[idx_test]

    X_train_a, X_val_a, X_test_a = X_amount.loc[idx_train], X_amount.loc[idx_val], X_amount.loc[idx_test]
    y_train_a, y_val_a, y_test_a = y_amount.loc[idx_train], y_amount.loc[idx_val], y_amount.loc[idx_test]


    prep_type = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["crop", "stage"]),
            ("num", "passthrough", ["soilN", "soilP", "soilK", "ph"]),
        ]
    )

    clf = Pipeline(steps=[
        ("prep", prep_type),
        ("model", RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            max_depth=16,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt"
        ))
    ])

    clf.fit(X_train_t, y_train_t)

    train_acc = float(accuracy_score(y_train_t, clf.predict(X_train_t)))
    val_acc = float(accuracy_score(y_val_t, clf.predict(X_val_t)))
    test_acc = float(accuracy_score(y_test_t, clf.predict(X_test_t)))


    prep_amount = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["crop", "stage", "fertilizer_type"]),
            ("num", "passthrough", ["soilN", "soilP", "soilK", "ph"]),
        ]
    )

    reg_amount = Pipeline(steps=[
        ("prep", prep_amount),
        ("model", RandomForestRegressor(
            n_estimators=700,
            random_state=42,
            max_depth=18,
            min_samples_leaf=2,
            min_samples_split=6
        ))
    ])

    reg_amount.fit(X_train_a, y_train_a)

    pred_train = reg_amount.predict(X_train_a)
    pred_val = reg_amount.predict(X_val_a)
    pred_test = reg_amount.predict(X_test_a)

    train_mae = float(mean_absolute_error(y_train_a, pred_train))
    val_mae = float(mean_absolute_error(y_val_a, pred_val))
    test_mae = float(mean_absolute_error(y_test_a, pred_test))

    train_rmse = float(np.sqrt(mean_squared_error(y_train_a, pred_train)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val_a, pred_val)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test_a, pred_test)))

    # Save artifacts next to this script
    joblib.dump(clf, BASE_DIR / "model_fert_type.joblib")
    joblib.dump(reg_amount, BASE_DIR / "model_fert_amount.joblib")

    metrics = {
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "type_classifier": {
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
        },
        "amount_regressor": {
            "train_mae": round(train_mae, 6),
            "val_mae": round(val_mae, 6),
            "test_mae": round(test_mae, 6),
            "train_rmse": round(train_rmse, 6),
            "val_rmse": round(val_rmse, 6),
            "test_rmse": round(test_rmse, 6),
        }
    }

    with open(BASE_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("type test acc:", round(test_acc, 4))
    print("amount test mae:", round(test_mae, 6), "rmse:", round(test_rmse, 6))
    print("saved: model_fert_type.joblib , model_fert_amount.joblib , model_metrics.json")

    return metrics


def main() -> None:
    """Entry point for training when running this file directly."""
    train_models(DATA_PATH)


if __name__ == "__main__":
    main()
