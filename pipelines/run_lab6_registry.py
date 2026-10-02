import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "src/train_registry.py",           # trains candidates, registers versions
    "src/automate_lifecycle.py",       # compares versions, promotes the Champion
    "src/generate_registry_report.py", # writes the JSON deployment artifact
]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry and Lifecycle")
    print("[INFO] =========================================")

    for script in SCRIPTS:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} caught an error.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")


if __name__ == "__main__":
    execute_pipeline()
