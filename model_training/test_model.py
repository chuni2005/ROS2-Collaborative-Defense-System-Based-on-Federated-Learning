from pathlib import Path

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

from preprocessing import preprocess_data, load_category_maps, DEFAULT_CATEGORY_MAPS_FILENAME


def evaluate_model(
    model_path: str | Path,
    test_csv_path: str | Path,
    category_maps_path: str | Path,
) -> None:
    model_path = Path(model_path)
    test_csv_path = Path(test_csv_path)

    category_maps = load_category_maps(category_maps_path)
    df = pd.read_csv(test_csv_path, low_memory=False)
    df = preprocess_data(df, category_maps)

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
    category_maps_path = base_dir / DEFAULT_CATEGORY_MAPS_FILENAME

    evaluate_model(model_path, test_csv_path, category_maps_path)