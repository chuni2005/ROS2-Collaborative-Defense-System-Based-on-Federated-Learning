import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def convert_type(value):
    if (isinstance(value, (int, float, np.number)) and not pd.isna(value)) and not isinstance(value, bool):
        return value

    if pd.isna(value) or value == "":
        return ""

    try:
        return pd.to_numeric(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != "attack"]

    if hasattr(df, "map"):
        df[feature_cols] = df[feature_cols].map(convert_type)
    else:
        df[feature_cols] = df[feature_cols].applymap(convert_type)

    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    if "attack" in df.columns:
        attack_mapping = {
            "observe": 0,
            "metasploit SYN flood": 1,
            "nmap discovery": 1,
            "nmap SYN flood": 1,
            "ros2 node crashing": 1,
            "ros2 reconnaissance": 1,
            "ros2 reflection": 1,
        }
        df["attack"] = df["attack"].replace(attack_mapping).infer_objects(copy=False)
        df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)

    df = df.drop(
        columns=[col for col in df.columns if "Unnamed" in col or "timestamp" in col],
        errors="ignore",
    )

    non_numeric_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns
    for col in non_numeric_cols:
        if col != "attack":
            df[col] = pd.Categorical(df[col]).codes

    df.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in df.columns]
    return df.astype(np.float32)


def evaluate_model(model_path: str | Path, test_csv_path: str | Path) -> None:
    model_path = Path(model_path)
    test_csv_path = Path(test_csv_path)

    df = pd.read_csv(test_csv_path, low_memory=False)
    df = preprocess_data(df)

    if "attack" not in df.columns:
        raise ValueError("The test file does not contain an 'attack' column.")

    X = df.drop(columns=["attack"])
    y_true = df["attack"].astype(int)

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    dtest = xgb.DMatrix(X)
    y_prob = booster.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf = confusion_matrix(y_true, y_pred)

    print("=== Global Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Test data: {test_csv_path}")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model" / "stratified_strategy" / "global_model_latest.ubj"
    test_csv_path = base_dir / "test-data" / "stratified_strategy" / "test.csv"

    evaluate_model(model_path, test_csv_path)
