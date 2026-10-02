import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_NAME, TRACKING_URI, EXPERIMENT_NAME, PROCESSED_DIR
from preprocess_pipeline import run_preprocessing

CANDIDATES = [
    ("LogisticRegression", LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
    ("GradientBoosting", GradientBoostingClassifier, {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}),
]


def train_and_register():
    print("[INFO] Training candidates and registering versions...")

    # Lab 6 must work even if Lab 5 was never run
    if not (PROCESSED_DIR / "X_train_final.npy").exists():
        print("[INFO] Processed arrays not found - running preprocessing first.")
        run_preprocessing()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = np.load(PROCESSED_DIR / "X_train_final.npy")
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    for name, cls, params in CANDIDATES:
        with mlflow.start_run(run_name=name):
            model = cls(**params).fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")

            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("precision", precision_score(y_test, preds, average="macro", zero_division=0))
            mlflow.log_metric("recall", recall_score(y_test, preds, average="macro", zero_division=0))
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=["sklearn.tree._tree.Tree"]
)
            print(f"[INFO] {name}: f1_macro = {f1:.4f} (registered)")

    print("[SUCCESS] All candidates trained and registered.")


if __name__ == "__main__":
    train_and_register()
