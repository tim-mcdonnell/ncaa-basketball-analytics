"""
Logging utilities for Airflow DAGs.
These utilities provide enhanced logging capabilities for NCAA Basketball Analytics.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from airflow.models import TaskInstance


def setup_task_logger(task_name: str) -> logging.Logger:
    """
    Set up a logger for a specific task with appropriate formatting.

    :param task_name: Name of the task for which to set up the logger
    :return: Configured logger instance
    """
    logger = logging.getLogger(f"ncaa_basketball.{task_name}")
    logger.setLevel(logging.INFO)

    # Create a handler that writes to a task-specific log file
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def log_task_progress(
    logger: logging.Logger, stage: str, message: str, data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log task progress information with structured data.

    :param logger: Logger instance to use
    :param stage: Current stage of the task (e.g., 'start', 'in_progress', 'complete')
    :param message: Log message
    :param data: Additional structured data to include in the log (optional)
    """
    log_data = {"timestamp": datetime.now().isoformat(), "stage": stage, "message": message}

    if data:
        log_data["data"] = data

    logger.info(json.dumps(log_data))


def log_task_execution(task_instance: TaskInstance, message: str) -> None:
    """
    Log information about a task execution with standardized format.

    :param task_instance: Airflow TaskInstance object
    :param message: Log message
    """
    task_instance.log.info(
        f"[NCAA-BASKETBALL] {message} | "
        f"dag_id={task_instance.dag_id}, "
        f"task_id={task_instance.task_id}, "
        f"execution_date={task_instance.execution_date.isoformat()}"
    )


def log_data_stats(
    logger: logging.Logger,
    data_type: str,
    record_count: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    additional_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log statistics about data being processed.

    :param logger: Logger instance to use
    :param data_type: Type of data (e.g., 'games', 'players', 'features')
    :param record_count: Number of records processed
    :param start_date: Start date for the data range (optional)
    :param end_date: End date for the data range (optional)
    :param additional_stats: Additional statistics to include in the log (optional)
    """
    stats = {
        "data_type": data_type,
        "record_count": record_count,
    }

    if start_date:
        stats["start_date"] = start_date

    if end_date:
        stats["end_date"] = end_date

    if additional_stats:
        stats.update(additional_stats)

    log_task_progress(logger, "data_stats", f"Processed {record_count} {data_type} records", stats)
