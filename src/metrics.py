"""Metrics calculation using Evidently AI for ML monitoring."""

import logging
from typing import Dict, Any

import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently import Report
from evidently.metrics.regression import RegressionQualityMetric
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    """
    Calculate basic regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    try:
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
            * 100,  # Convert to percentage
        }
        logger.info(f"Calculated regression metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to calculate regression metrics: {str(e)}")
        raise


def get_model_monitoring_metrics(regression_quality_report: Report) -> Dict[str, float]:
    """
    Extract metrics from Evidently regression quality report.

    Args:
        regression_quality_report: Evidently Report with RegressionQualityMetric

    Returns:
        Dictionary of extracted metrics
    """
    try:
        metrics = {}
        report_dict = regression_quality_report.as_dict()

        # Extract metrics from the first metric (RegressionQualityMetric)
        current_metrics = report_dict["metrics"][0]["result"]["current"]

        metrics["me"] = current_metrics["mean_error"]
        metrics["mae"] = current_metrics["mean_abs_error"]
        metrics["rmse"] = current_metrics["rmse"]
        metrics["mape"] = current_metrics["mean_abs_perc_error"]

        logger.info(f"Extracted monitoring metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to extract monitoring metrics: {str(e)}")
        raise


def get_comprehensive_monitoring_metrics(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> Dict[str, Any]:
    """
    Calculate comprehensive monitoring metrics for Grafana dashboard.

    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (production data)
        column_mapping: Evidently column mapping

    Returns:
        Dictionary with all monitoring metrics
    """
    try:
        all_metrics = {}

        # Basic Data Quality Metrics (simple calculations)
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
        all_metrics.update({f"quality_{k}": v for k, v in quality_metrics.items()})

        # Simple drift calculation (statistical comparison)
        drift_metrics = {}
        for col in column_mapping.numerical_features or []:
            if col in reference_data.columns and col in current_data.columns:
                ref_mean = reference_data[col].mean()
                curr_mean = current_data[col].mean()
                drift_metrics[f"{col}_mean_drift"] = abs(curr_mean - ref_mean) / (
                    ref_mean + 1e-8
                )

        all_metrics.update({f"drift_{k}": v for k, v in drift_metrics.items()})

        logger.info(f"Calculated {len(all_metrics)} comprehensive monitoring metrics")
        return all_metrics

    except Exception as e:
        logger.error(f"Failed to calculate comprehensive monitoring metrics: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage with sample data
    logger.info("Metrics calculation module loaded successfully")
