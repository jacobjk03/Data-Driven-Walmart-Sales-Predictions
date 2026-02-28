"""App and training configuration."""
import os
from pathlib import Path

# Data: use DATA_DIR env or default to ./data
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SALES_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"

# MLflow: use sqlite in project root so plain "mlflow ui" (from project root) shows all runs
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MLFLOW_DB = (_PROJECT_ROOT / "mlflow.db").resolve().as_posix()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{_MLFLOW_DB}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "exponential-smoothing")

# Model artifact directory (for FastAPI to load when not using MLflow registry)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# Train/test split
TRAIN_TEST_SPLIT = 0.8

# Exponential Smoothing defaults (Holt-Winters seasonal)
DEFAULT_SEASONAL_PERIODS = 7  # weekly seasonality
DEFAULT_TREND = "add"
DEFAULT_SEASONAL = "add"
