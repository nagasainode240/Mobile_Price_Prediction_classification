import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from config import PROCESSED_DIR, ARTIFACT_DIR


def run_deterministic_test():
    print("Running Reproducibility Validation...")
    X_train = np.load(PROCESSED_DIR / "X_train_final.npy")
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

    scores = []
    for _ in range(2):
        model = RandomForestClassifier(**params).fit(X_train, y_train)
        scores.append(f1_score(y_test, model.predict(X_test), average="macro"))

    ok = scores[0] == scores[1]
    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_f1": scores[0],
        "execution_2_f1": scores[1],
        "is_strictly_reproducible": bool(ok),
        "status": "PASSED" if ok else "FAILED",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 F1: {scores[0]:.6f}")
    print(f"Execution 2 F1: {scores[1]:.6f}")
    print("SUCCESS: Pipeline is 100% reproducible." if ok else "FAILED: Pipeline is non-deterministic.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    run_deterministic_test()
