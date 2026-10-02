import json
import mlflow
from mlflow.tracking import MlflowClient
from config import MODEL_NAME, TRACKING_URI, PREPROCESSOR_PATH, ARTIFACT_DIR, PROJECT_ROOT


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(champion.run_id)
        m = run.data.metrics
        stage = champion.current_stage if champion.current_stage not in (None, "None") else "Production"

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": MODEL_NAME,
                "version": int(champion.version),
                "alias": "champion",
                "current_stage": stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "accuracy": m.get("accuracy"),
                "precision_macro": m.get("precision"),
                "recall_macro": m.get("recall"),
                "f1_macro": m.get("f1_score"),
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / "production_model_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[SUCCESS] Report generated for Version {champion.version}")
        print(f"[INFO] Saved to: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    generate_registry_report()
