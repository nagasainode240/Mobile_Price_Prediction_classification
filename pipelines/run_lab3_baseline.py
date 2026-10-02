
from pathlib import Path
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py",
]


def main():
    log_path = LOG_DIR / "lab3_baseline.log"

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Mobile Price Prediction - Lab 3 Baseline\n")
        log.write(f"Started: {datetime.now().isoformat()}\n\n")

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
                raise RuntimeError(
                    f"{script} failed. Check {log_path}"
                )

        log.write("\nBaseline pipeline completed successfully.\n")

    print("\nBaseline pipeline completed successfully!")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()