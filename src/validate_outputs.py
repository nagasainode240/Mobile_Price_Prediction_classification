import numpy as np
from config import PROCESSED_DIR, PREPROCESSOR_PATH

REQUIRED = [
    PROCESSED_DIR / "X_train_final.npy", PROCESSED_DIR / "X_test_final.npy",
    PROCESSED_DIR / "y_train.npy", PROCESSED_DIR / "y_test.npy",
    PROCESSED_DIR / "dataset_metadata.json", PREPROCESSOR_PATH,
]


def validate_outputs():
    print("[INFO] Validating pipeline outputs...")
    errors = [f"Missing file: {p}" for p in REQUIRED if not p.is_file()]
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False

    X_train = np.load(PROCESSED_DIR / "X_train_final.npy")
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
        errors.append("Row count mismatch between X and y")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Train/test feature count mismatch")
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        errors.append("NaN values found in processed features")
    if not set(np.unique(y_train)).issubset({0, 1, 2, 3}):
        errors.append("Unexpected labels in y_train")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False
    print(f"[SUCCESS] Outputs valid. Train {X_train.shape}, Test {X_test.shape}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_outputs() else 1)
