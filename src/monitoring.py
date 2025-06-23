"""Main monitoring script that calculates metrics and logs them to PostgreSQL."""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn import ensemble, model_selection

from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently import Report
from evidently.metrics.regression import RegressionQualityMetric

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_utils import log_metrics_to_db, create_db_tables, test_db_connection
from src.metrics import (
    get_model_monitoring_metrics,
    get_comprehensive_monitoring_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
FILENAME = "raw_data.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def setup_column_mapping():
    """Set up column mapping for Evidently."""
    target = "cnt"
    prediction = "prediction"
    datetime_col = "dteday"
    numerical_features = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
    categorical_features = ["season", "holiday", "workingday"]

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.datetime = datetime_col
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    return column_mapping, target, prediction, numerical_features, categorical_features


def load_and_prepare_data():
    """Load and prepare the bike sharing dataset."""
    try:
        # Load data
        raw_data = pd.read_csv(f"{DATA_DIR}/{FILENAME}")
        raw_data = raw_data.set_index("dteday")

        # Define time periods
        train_dates = ("2011-01-02 00:00:00", "2011-03-06 23:00:00")
        prediction_batches = [
            ("2011-03-07 00:00:00", "2011-03-13 23:00:00"),
            ("2011-03-14 00:00:00", "2011-03-20 23:00:00"),
            ("2011-03-21 00:00:00", "2011-03-27 23:00:00"),
        ]

        logger.info(f"Loaded data with shape: {raw_data.shape}")
        return raw_data, train_dates, prediction_batches

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def train_model(raw_data, train_dates, feature_columns, target):
    """Train a Random Forest model."""
    try:
        # Get training data
        sample_data = raw_data.loc[
            "2011-01-01 00:00:00":"2011-01-28 23:00:00"
        ].reset_index()

        # Split data
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            sample_data[feature_columns],
            sample_data[target],
            test_size=0.3,
            random_state=42,
        )

        # Train model
        regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
        regressor.fit(X_train, y_train)

        # Save model
        Path(MODELS_DIR).mkdir(exist_ok=True)
        model_path = Path(f"{MODELS_DIR}/model.joblib")
        joblib.dump(regressor, model_path)

        logger.info(f"Model trained and saved to {model_path}")
        return regressor

    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise


def prepare_reference_data(raw_data, train_dates, regressor, feature_columns):
    """Prepare reference dataset with predictions."""
    try:
        reference_data = raw_data.loc[train_dates[0] : train_dates[1]]
        reference_data["prediction"] = regressor.predict(
            reference_data[feature_columns]
        )
        reference_data = reference_data.reset_index(drop=True)

        logger.info(f"Reference data prepared with shape: {reference_data.shape}")
        return reference_data

    except Exception as e:
        logger.error(f"Failed to prepare reference data: {str(e)}")
        raise


def run_monitoring_for_batch(
    raw_data, current_dates, regressor, feature_columns, reference_data, column_mapping
):
    """Run monitoring for a single batch."""
    try:
        logger.info(f"Processing batch: {current_dates}")

        # Prepare current data
        current_data = raw_data.loc[current_dates[0] : current_dates[1]]
        current_prediction = regressor.predict(current_data[feature_columns])
        current_data["prediction"] = current_prediction
        current_data = current_data.reset_index(drop=True)

        # Create monitoring report
        model_report = Report(metrics=[RegressionQualityMetric()])
        model_report.run(current_data=current_data, reference_data=reference_data)

        # Extract metrics
        model_metrics = get_model_monitoring_metrics(model_report)

        # Get comprehensive metrics
        comprehensive_metrics = get_comprehensive_monitoring_metrics(
            reference_data, current_data, column_mapping
        )

        # Combine all metrics
        all_metrics = {**model_metrics, **comprehensive_metrics}

        # Add timestamp info
        batch_name = f"batch_{current_dates[1].split(' ')[0]}"  # Extract date part

        # Log to database
        log_metrics_to_db(
            metrics=all_metrics, dataset_name="bike_sharing", data_period=batch_name
        )

        logger.info(
            f"Successfully processed batch {current_dates} with {len(all_metrics)} metrics"
        )
        return all_metrics

    except Exception as e:
        logger.error(f"Failed to process batch {current_dates}: {str(e)}")
        raise


def main():
    """Main monitoring workflow."""
    try:
        logger.info("Starting ML monitoring workflow")

        # 1. Setup database
        if not test_db_connection():
            raise Exception("Database connection failed")
        create_db_tables()

        # 2. Setup configuration
        column_mapping, target, prediction, numerical_features, categorical_features = (
            setup_column_mapping()
        )
        feature_columns = numerical_features + categorical_features

        # 3. Load data
        raw_data, train_dates, prediction_batches = load_and_prepare_data()

        # 4. Train model (or load existing)
        model_path = Path(f"{MODELS_DIR}/model.joblib")
        if model_path.exists():
            logger.info("Loading existing model")
            regressor = joblib.load(model_path)
        else:
            logger.info("Training new model")
            regressor = train_model(raw_data, train_dates, feature_columns, target)

        # 5. Prepare reference data
        reference_data = prepare_reference_data(
            raw_data, train_dates, regressor, feature_columns
        )

        # 6. Run monitoring for each batch
        all_batch_metrics = []
        for current_dates in prediction_batches:
            batch_metrics = run_monitoring_for_batch(
                raw_data,
                current_dates,
                regressor,
                feature_columns,
                reference_data,
                column_mapping,
            )
            all_batch_metrics.append(batch_metrics)

        logger.info(
            f"Monitoring completed successfully for {len(prediction_batches)} batches"
        )
        logger.info("Metrics have been logged to PostgreSQL database")
        logger.info("You can now view the metrics in Grafana at http://localhost:3000")
        logger.info("Default login: admin/admin")

        return all_batch_metrics

    except Exception as e:
        logger.error(f"Monitoring workflow failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
