#!/usr/bin/env python3
"""
Train Exponential Smoothing on store-level sales and log experiment with MLflow.
Run from project root: python train.py [--store-id CA_1] [--data-dir data]
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

from src.config import (
    DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    TRAIN_TEST_SPLIT,
    DEFAULT_SEASONAL_PERIODS,
    DEFAULT_TREND,
    DEFAULT_SEASONAL,
)
from src.data import load_store_level_data, get_store_series
from src.model import train_exponential_smoothing, predict, save_model


def main():
    parser = argparse.ArgumentParser(description="Train Exponential Smoothing with MLflow")
    parser.add_argument("--store-id", default="CA_1", help="Store to train on (e.g. CA_1, TX_1)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing M5 CSV files")
    parser.add_argument("--trend", default=DEFAULT_TREND, choices=("add", "mul"))
    parser.add_argument("--seasonal", default=DEFAULT_SEASONAL, choices=("add", "mul"))
    parser.add_argument("--seasonal-periods", type=int, default=DEFAULT_SEASONAL_PERIODS)
    parser.add_argument("--train-split", type=float, default=TRAIN_TEST_SPLIT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    sales_path = args.data_dir / "sales_train_validation.csv"
    calendar_path = args.data_dir / "calendar.csv"
    if not sales_path.exists() or not calendar_path.exists():
        raise FileNotFoundError(
            f"Data not found. Place M5 files in {args.data_dir}: sales_train_validation.csv, calendar.csv"
        )

    store_level = load_store_level_data(sales_path, calendar_path)
    series = get_store_series(store_level, args.store_id)

    n = len(series)
    train_size = int(n * args.train_split)
    train_series = series.iloc[:train_size]
    test_series = series.iloc[train_size:]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"exp_smoothing_{args.store_id}"):
        mlflow.log_params({
            "store_id": args.store_id,
            "trend": str(args.trend),
            "seasonal": str(args.seasonal),
            "seasonal_periods": args.seasonal_periods,
            "train_size": train_size,
            "test_size": len(test_series),
        })

        fitted = train_exponential_smoothing(
            train_series,
            trend=args.trend,
            seasonal=args.seasonal,
            seasonal_periods=args.seasonal_periods,
        )

        # Evaluate: forecast test length and compute RMSE
        steps = len(test_series)
        pred = predict(fitted, steps=steps)
        rmse = np.sqrt(mean_squared_error(test_series.values, pred))
        mlflow.log_metric("rmse", rmse)

        # Save model artifact for this run
        artifact_path = Path("model.pkl")
        save_model(fitted, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        artifact_path.unlink(missing_ok=True)

        # Also save to fixed MODEL_DIR so the API can load without MLflow server
        args.model_dir.mkdir(parents=True, exist_ok=True)
        save_model(fitted, args.model_dir / f"exp_smoothing_{args.store_id}.pkl")

        print(f"RMSE: {rmse:.4f}")
        print(f"Model saved to {args.model_dir / f'exp_smoothing_{args.store_id}.pkl'}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
