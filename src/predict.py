from pathlib import Path
import pandas as pd
import joblib

# Project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load modes
MODEL_PATH = PROJECT_ROOT / "models" / "mobile_price_model.pkl"

# Load test data
X_test_path = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

# Load trained model
model = joblib.load(MODEL_PATH)

# Predict
predictions = model.predict(X_test)

print("\nMobile Price Predictions")
print("=" * 40)

# Show first 10 predictions
for i in range(10):
    print(
        f"Phone {i + 1}: "
        f"Actual Price Range = {y_test.iloc[i]}, "
        f"Predicted Price Range = {predictions[i]}"
    )