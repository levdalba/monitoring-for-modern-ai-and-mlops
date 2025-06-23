"""Database utilities for ML monitoring metrics storage."""

import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class ModelPerformance(Base):
    """Table for storing model performance metrics over time."""

    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    dataset_name = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    data_period = Column(String(50), nullable=True)  # e.g., "2023-01", "week_1"


# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "monitoring_db",
    "user": "admin",
    "password": "admin",
}


def get_db_engine():
    """Create and return SQLAlchemy engine."""
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(connection_string)


def create_db_tables():
    """Create all database tables."""
    try:
        engine = get_db_engine()
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def get_db_session():
    """Get database session."""
    engine = get_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def log_metrics_to_db(
    metrics: Dict[str, Any], dataset_name: str, data_period: str = None
):
    """
    Log metrics to the database.

    Args:
        metrics: Dictionary of metric names and values
        dataset_name: Name of the dataset being monitored
        data_period: Period identifier (e.g., "2023-01", "week_1")
    """
    try:
        session = get_db_session()
        timestamp = datetime.utcnow()

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                performance_record = ModelPerformance(
                    timestamp=timestamp,
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    data_period=data_period,
                )
                session.add(performance_record)

        session.commit()
        logger.info(
            f"Successfully logged {len(metrics)} metrics to database for dataset '{dataset_name}'"
        )

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to log metrics to database: {str(e)}")
        raise
    finally:
        session.close()


def get_metrics_from_db(dataset_name: str = None, limit: int = 100) -> pd.DataFrame:
    """
    Retrieve metrics from the database.

    Args:
        dataset_name: Filter by dataset name (optional)
        limit: Maximum number of records to retrieve

    Returns:
        DataFrame with metrics data
    """
    try:
        engine = get_db_engine()

        if dataset_name:
            query = """
                SELECT * FROM model_performance 
                WHERE dataset_name = %(dataset_name)s 
                ORDER BY timestamp DESC 
                LIMIT %(limit)s
            """
            df = pd.read_sql_query(
                query, engine, params={"dataset_name": dataset_name, "limit": limit}
            )
        else:
            query = """
                SELECT * FROM model_performance 
                ORDER BY timestamp DESC 
                LIMIT %(limit)s
            """
            df = pd.read_sql_query(query, engine, params={"limit": limit})

        logger.info(f"Retrieved {len(df)} records from database")
        return df

    except Exception as e:
        logger.error(f"Failed to retrieve metrics from database: {str(e)}")
        raise


def test_db_connection():
    """Test database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(
            f"Database connection successful. PostgreSQL version: {version['version']}"
        )
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test database connection and create tables
    if test_db_connection():
        create_db_tables()
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")
