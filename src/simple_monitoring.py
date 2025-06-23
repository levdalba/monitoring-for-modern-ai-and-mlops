"""Simplified monitoring script for ML models using standard metrics."""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn import ensemble, model_selection
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_utils import log_metrics_to_db, create_db_tables, test_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
FILENAME = "raw_data.csv"
MODELS_DIR = "models"


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def calculate_data_stats(data, prefix=""):
    """Calculate basic data statistics."""
    stats = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        stats[f"{prefix}{col}_mean"] = data[col].mean()
        stats[f"{prefix}{col}_std"] = data[col].std()
        stats[f"{prefix}{col}_min"] = data[col].min()
        stats[f"{prefix}{col}_max"] = data[col].max()
    return stats


def calculate_drift_metrics(reference_data, current_data):
    """Calculate simple drift metrics by comparing means."""
    drift_metrics = {}
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in current_data.columns:
            ref_mean = reference_data[col].mean()
            curr_mean = current_data[col].mean()
            ref_std = reference_data[col].std()

            # Calculate drift as normalized difference
            drift_score = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            drift_metrics[f"drift_{col}"] = drift_score

    return drift_metrics


def main():
    """Main monitoring workflow."""
    try:
        logger.info("Starting simplified ML monitoring workflow")

        # 1. Setup database
        if not test_db_connection():
            raise Exception("Database connection failed")
        create_db_tables()

        # 2. Load data
        raw_data = pd.read_csv(f"{DATA_DIR}/{FILENAME}")
        raw_data = raw_data.set_index("dteday")
        logger.info(f"Loaded data with shape: {raw_data.shape}")

        # 3. Define periods
        train_dates = ("2011-01-02 00:00:00", "2011-03-06 23:00:00")
        prediction_batches = [
            ("2011-03-07 00:00:00", "2011-03-13 23:00:00"),
            ("2011-03-14 00:00:00", "2011-03-20 23:00:00"),
            ("2011-03-21 00:00:00", "2011-03-27 23:00:00"),
        ]

        # 4. Define features
        target = "cnt"
        numerical_features = [
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "mnth",
            "hr",
            "weekday",
        ]
        categorical_features = ["season", "holiday", "workingday"]
        feature_columns = numerical_features + categorical_features

        # 5. Train model
        model_path = Path(f"{MODELS_DIR}/model.joblib")
        if model_path.exists():
            logger.info("Loading existing model")
            regressor = joblib.load(model_path)
        else:
            logger.info("Training new model")
            sample_data = raw_data.loc[
                "2011-01-01 00:00:00":"2011-01-28 23:00:00"
            ].reset_index()

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                sample_data[feature_columns],
                sample_data[target],
                test_size=0.3,
                random_state=42,
            )

            regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
            regressor.fit(X_train, y_train)

            Path(MODELS_DIR).mkdir(exist_ok=True)
            joblib.dump(regressor, model_path)
            logger.info(f"Model trained and saved to {model_path}")

        # 6. Prepare reference data
        reference_data = raw_data.loc[train_dates[0] : train_dates[1]]
        reference_data["prediction"] = regressor.predict(
            reference_data[feature_columns]
        )
        reference_data = reference_data.reset_index(drop=True)
        logger.info(f"Reference data prepared with shape: {reference_data.shape}")

        # 7. Run monitoring for each batch
        for i, current_dates in enumerate(prediction_batches):
            logger.info(f"Processing batch {i+1}: {current_dates}")

            # Prepare current data
            current_data = raw_data.loc[current_dates[0] : current_dates[1]]
            current_prediction = regressor.predict(current_data[feature_columns])
            current_data["prediction"] = current_prediction
            current_data = current_data.reset_index(drop=True)

            # Calculate performance metrics
            perf_metrics = calculate_metrics(
                current_data[target], current_data["prediction"]
            )

            # Calculate drift metrics
            drift_metrics = calculate_drift_metrics(
                reference_data[feature_columns], current_data[feature_columns]
            )

            # Calculate data quality metrics
            quality_metrics = {
                "num_rows": len(current_data),
                "num_columns": len(current_data.columns),
                "missing_values": current_data.isnull().sum().sum(),
                "missing_percentage": (
                    current_data.isnull().sum().sum()
                    / (len(current_data) * len(current_data.columns))
                )
                * 100,
            }

            # Combine all metrics
            all_metrics = {}
            all_metrics.update({f"performance_{k}": v for k, v in perf_metrics.items()})
            all_metrics.update(drift_metrics)
            all_metrics.update({f"quality_{k}": v for k, v in quality_metrics.items()})

            # Log to database
            batch_name = f"batch_{current_dates[1].split(' ')[0]}"  # Extract date part
            log_metrics_to_db(
                metrics=all_metrics, dataset_name="bike_sharing", data_period=batch_name
            )

            logger.info(
                f"Successfully processed batch {i+1} with {len(all_metrics)} metrics"
            )
            logger.info(
                f"Performance metrics: MAE={perf_metrics['mae']:.2f}, RMSE={perf_metrics['rmse']:.2f}, MAPE={perf_metrics['mape']:.2f}%"
            )

        logger.info(
            f"Monitoring completed successfully for {len(prediction_batches)} batches"
        )
        logger.info("Metrics have been logged to PostgreSQL database")
        logger.info("You can now view the metrics in Grafana at http://localhost:3000")
        logger.info("Default login: admin/admin")

    except Exception as e:
        logger.error(f"Monitoring workflow failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
