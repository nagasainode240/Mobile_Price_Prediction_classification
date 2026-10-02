import mlflow
from mlflow.tracking import MlflowClient
from config import MODEL_NAME, TRACKING_URI

CHAMPION_ALIAS = "champion"


def promote_champion():
    print("[INFO] Comparing model versions...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("[ERROR] No registered versions found.")
        raise SystemExit(1)

    scored = []
    for mv in versions:
        f1 = client.get_run(mv.run_id).data.metrics.get("f1_score", -1)
        scored.append((f1, int(mv.version), mv))
        print(f"  Version {mv.version}: f1_score={f1:.4f}")

    # Highest F1 wins; ties go to the newest version
    best_f1, best_ver, _ = max(scored, key=lambda t: (t[0], t[1]))

    # Alias is the modern, supported way to mark the champion
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, str(best_ver))

    # Classic stage transition (deprecated in newer MLflow, so never fatal)
    for _, ver, mv in scored:
        stage = "Production" if ver == best_ver else "Archived"
        try:
            client.transition_model_version_stage(MODEL_NAME, mv.version, stage)
        except Exception as e:
            print(f"[WARN] Stage transition skipped: {type(e).__name__}")
            break

    print(f"[SUCCESS] Version {best_ver} is the Champion (f1={best_f1:.4f}).")


if __name__ == "__main__":
    promote_champion()
