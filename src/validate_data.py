import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
from config import RAW_PATH, ARTIFACT_DIR


def get_mobile_schema():
    """Strict schema specification for the Mobile Price dataset."""
    binary = lambda: Column(pa.Int, Check.isin([0, 1]))
    return pa.DataFrameSchema({
        "battery_power": Column(pa.Int, Check.ge(0)),
        "blue": binary(),
        "clock_speed": Column(pa.Float, Check.ge(0.0)),
        "dual_sim": binary(),
        "fc": Column(pa.Int, Check.ge(0)),
        "four_g": binary(),
        "int_memory": Column(pa.Int, Check.ge(0)),
        "m_dep": Column(pa.Float, Check.in_range(0.0, 2.0)),
        "mobile_wt": Column(pa.Int, Check.gt(0)),
        "n_cores": Column(pa.Int, Check.in_range(1, 16)),
        "pc": Column(pa.Int, Check.ge(0)),
        "px_height": Column(pa.Int, Check.ge(0)),
        "px_width": Column(pa.Int, Check.ge(0)),
        "ram": Column(pa.Int, Check.gt(0)),
        "sc_h": Column(pa.Int, Check.ge(0)),
        "sc_w": Column(pa.Int, Check.ge(0)),
        "talk_time": Column(pa.Int, Check.ge(0)),
        "three_g": binary(),
        "touch_screen": binary(),
        "wifi": binary(),
        "price_range": Column(pa.Int, Check.isin([0, 1, 2, 3])),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_mobile_schema()
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = ARTIFACT_DIR / output_report_name
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    if RAW_PATH.exists():
        ok = validate_schema(pd.read_csv(RAW_PATH, encoding="latin1"),
                             output_report_name="baseline_validation.csv")
        raise SystemExit(0 if ok else 1)   # non-zero exit halts the pipeline
    print(f"[ERROR] Target file not found at: {RAW_PATH}")
    raise SystemExit(1)
