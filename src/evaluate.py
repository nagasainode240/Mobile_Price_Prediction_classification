
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobile_price_model.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model():
    # Load the trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    # Load testing data
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Save confusion matrix
    pd.DataFrame(matrix).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        index=False,
    )

    # Save evaluation report
    report_text = (
        "Mobile Price Prediction - Model Evaluation\n"
        "==========================================\n\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{matrix}\n"
    )

    report_path = REPORT_DIR / "model_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    evaluate_model()