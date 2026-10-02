from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

X_train_path = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
X_test_path = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
y_train_path = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
y_test_path = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "mobile_price_model.pkl"

# Load processed data
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

y_train = pd.read_csv(y_train_path).squeeze()
y_test = pd.read_csv(y_test_path).squeeze()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)