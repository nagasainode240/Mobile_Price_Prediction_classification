"""Shared constants so every script uses the same paths and column names."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

TARGET = "price_range"
BINARY_COLS = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]

MODEL_NAME = "Mobile_Price_Production_Model"
EXPERIMENT_NAME = "Mobile_Price_Registry"
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
