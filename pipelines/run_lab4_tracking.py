
from pathlib import Path
import subprocess
import sys
import csv
import re
from datetime import datetime

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Scripts executed during the experiment
SCRIPTS = [
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py",
]

def main():
    # Unique experiment information
    run_time = datetime.now()
    run_id = run_time.strftime("%Y%m%d_%H%M%S")

    log_path = LOG_DIR / "lab4_tracking.log"
    tracking_path = ARTIFACT_DIR / "experiment_tracking.csv"

    accuracy = ""
    status = "SUCCESS"

    print("=" * 55)
    print("MOBILE PRICE PREDICTION - EXPERIMENT TRACKING")
    print("=" * 55)
    print(f"Experiment ID: {run_id}")
    print(f"Started at: {run_time.isoformat(timespec='seconds')}")

    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("Mobile Price Prediction - Lab 4 Tracking\n")
            log.write(f"Experiment ID: {run_id}\n")
            log.write(f"Started: {run_time.isoformat()}\n")

            for script in SCRIPTS:
                print(f"\nRunning {script} ...")
                log.write(f"\n--- Running {script} ---\n")
                log.flush()

                result = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / script)],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )

                print(result.stdout)
                log.write(result.stdout)

                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                    log.write(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(f"{script} failed")

                # Read the evaluation accuracy
                if script == "src/evaluate.py":
                    match = re.search(
                        r"Accuracy:\s*([\d.]+)",
                        result.stdout
                    )
                    if match:
                        accuracy = float(match.group(1))

            log.write("\nExperiment completed successfully.\n")

    except Exception as error:
        status = "FAILED"
        print(f"\nExperiment failed: {error}")

    # Save experiment parameters and results
    file_exists = tracking_path.exists()

    with tracking_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists or tracking_path.stat().st_size == 0:
            writer.writerow([
                "experiment_id",
                "timestamp",
                "model",
                "dataset",
                "test_size",
                "random_state",
                "accuracy",
                "status",
            ])

        writer.writerow([
            run_id,
            run_time.isoformat(timespec="seconds"),
            "Current model from src/train.py",
            "Mobile Price Classification",
            0.20,
            42,
            accuracy,
            status,
        ])

    print("\n" + "=" * 55)
    print(f"Experiment status: {status}")
    print(f"Accuracy: {accuracy if accuracy != '' else 'Not recorded'}")
    print(f"Tracking CSV: {tracking_path}")
    print(f"Execution log: {log_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()