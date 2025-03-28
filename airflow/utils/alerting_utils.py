"""
Alerting utilities for Airflow DAGs.
These utilities provide alerting mechanisms for NCAA Basketball Analytics pipelines.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from airflow.hooks.base import BaseHook
from airflow.models import TaskInstance, DagRun
import requests


logger = logging.getLogger("ncaa_basketball.alerting")


def send_slack_alert(
    webhook_url: str, message: str, attachments: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Send an alert to a Slack channel using a webhook.

    :param webhook_url: Slack webhook URL
    :param message: Message to send
    :param attachments: Optional Slack message attachments
    :return: True if the alert was sent successfully, False otherwise
    """
    try:
        payload = {
            "text": message,
        }

        if attachments:
            payload["attachments"] = attachments

        response = requests.post(
            webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}
        )

        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {str(e)}")
        return False


def send_email_alert(
    to: Union[str, List[str]], subject: str, html_content: str, from_email: Optional[str] = None
) -> bool:
    """
    Send an email alert.

    :param to: Recipient email address(es)
    :param subject: Email subject
    :param html_content: HTML email content
    :param from_email: Sender email address (optional)
    :return: True if the email was sent successfully, False otherwise
    """
    try:
        from airflow.utils.email import send_email

        # Convert string to list if needed
        recipients = to if isinstance(to, list) else [to]

        send_email(to=recipients, subject=subject, html_content=html_content, from_email=from_email)

        return True
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")
        return False


def alert_task_failure(
    task_instance: TaskInstance, alert_channels: Optional[List[str]] = None
) -> None:
    """
    Send alerts for task failures.

    :param task_instance: Failed Airflow TaskInstance
    :param alert_channels: Channels to alert (default: ['slack', 'email'])
    """
    if alert_channels is None:
        alert_channels = ["slack", "email"]

    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = task_instance.execution_date.isoformat()
    exception = task_instance.get_template_context().get("exception", "Unknown error")

    message = (
        f"❌ Task Failure: {task_id} in DAG {dag_id}\n"
        f"Execution date: {execution_date}\n"
        f"Error: {str(exception)}"
    )

    if "slack" in alert_channels:
        try:
            # Assumes a Slack connection is set up in Airflow
            slack_webhook = BaseHook.get_connection("slack_alerts").password
            send_slack_alert(
                webhook_url=slack_webhook,
                message=message,
                attachments=[
                    {
                        "color": "danger",
                        "fields": [
                            {"title": "Task ID", "value": task_id, "short": True},
                            {"title": "DAG ID", "value": dag_id, "short": True},
                            {"title": "Execution Date", "value": execution_date, "short": True},
                        ],
                    }
                ],
            )
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    if "email" in alert_channels:
        try:
            # Get email recipients from Airflow configuration
            from airflow.configuration import conf

            email_addresses = conf.get("email", "email_backend")
            html_content = f"""
            <h2>Airflow Task Failure</h2>
            <p><strong>Task:</strong> {task_id}</p>
            <p><strong>DAG:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {execution_date}</p>
            <p><strong>Error:</strong></p>
            <pre>{str(exception)}</pre>
            """

            send_email_alert(
                to=email_addresses,
                subject=f"Airflow Alert: Task {task_id} in DAG {dag_id} failed",
                html_content=html_content,
            )
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")


def alert_dag_success(dag_run: DagRun) -> None:
    """
    Send alerts for successful DAG completions (for important DAGs).

    :param dag_run: Completed Airflow DagRun
    """
    # List of important DAGs to alert on successful completion
    important_dags = ["model_training_dag", "prediction_dag"]

    dag_id = dag_run.dag_id

    if dag_id not in important_dags:
        return

    execution_date = dag_run.execution_date.isoformat()
    end_date = dag_run.end_date.isoformat() if dag_run.end_date else "N/A"

    message = (
        f"✅ DAG Completed Successfully: {dag_id}\n"
        f"Execution date: {execution_date}\n"
        f"Completion time: {end_date}"
    )

    try:
        # Assumes a Slack connection is set up in Airflow
        slack_webhook = BaseHook.get_connection("slack_alerts").password
        send_slack_alert(
            webhook_url=slack_webhook,
            message=message,
            attachments=[
                {
                    "color": "good",
                    "fields": [
                        {"title": "DAG ID", "value": dag_id, "short": True},
                        {"title": "Execution Date", "value": execution_date, "short": True},
                        {"title": "Completion Time", "value": end_date, "short": True},
                    ],
                }
            ],
        )
    except Exception as e:
        logger.error(f"Failed to send DAG success alert: {str(e)}")


def alert_data_quality_issue(
    dag_id: str, task_id: str, issue_description: str, data_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send alerts for data quality issues.

    :param dag_id: ID of the DAG where the issue was detected
    :param task_id: ID of the task where the issue was detected
    :param issue_description: Description of the data quality issue
    :param data_metrics: Optional metrics about the problematic data
    """
    message = f"⚠️ Data Quality Issue in {task_id} (DAG: {dag_id})\n" f"Issue: {issue_description}"

    try:
        # Assumes a Slack connection is set up in Airflow
        slack_webhook = BaseHook.get_connection("slack_alerts").password

        attachments = [
            {
                "color": "warning",
                "fields": [
                    {"title": "Task ID", "value": task_id, "short": True},
                    {"title": "DAG ID", "value": dag_id, "short": True},
                    {"title": "Issue", "value": issue_description, "short": False},
                ],
            }
        ]

        if data_metrics:
            # Add data metrics as additional fields
            fields = []
            for key, value in data_metrics.items():
                fields.append({"title": key, "value": str(value), "short": True})

            attachments.append({"color": "warning", "title": "Data Metrics", "fields": fields})

        send_slack_alert(webhook_url=slack_webhook, message=message, attachments=attachments)
    except Exception as e:
        logger.error(f"Failed to send data quality alert: {str(e)}")
