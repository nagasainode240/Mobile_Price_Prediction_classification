import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import RAW_PATH, TARGET, BINARY_COLS, PREPROCESSOR_PATH, PROCESSED_DIR, MODELS_DIR


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")
    df = pd.read_csv(RAW_PATH, encoding="latin1")

    X = df.drop(TARGET, axis=1)
    y = df[TARGET].astype(int)

    # Split BEFORE fitting any transformer to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bin_cols = [c for c in BINARY_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in bin_cols]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    bin_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("bin", bin_pipeline, bin_cols),
    ])

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train_final.npy", X_train_final)
    np.save(PROCESSED_DIR / "X_test_final.npy", X_test_final)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy(dtype=np.int64))
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "dataset_name": "Mobile Price Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "binary_features": bin_cols,
        "target": TARGET,
        "classes": sorted(int(c) for c in y.unique()),
    }
    with open(PROCESSED_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
