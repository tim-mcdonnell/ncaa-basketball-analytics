# Logging Strategy and Implementation

## Overview

This document outlines the logging strategy for the NCAA Basketball Analytics project. A comprehensive logging system is critical for monitoring, debugging, and auditing the data pipelines, feature engineering processes, and model training/prediction flows.

## Logging Architecture

The project implements a multi-tiered logging architecture that balances detail, performance, and usability:

![Logging Architecture](https://i.imgur.com/placeholder.png)

### Key Principles

1. **Consistency**: Uniform logging format and levels across all components
2. **Granularity**: Appropriate detail level for different system components
3. **Performance**: Minimal impact on execution speed, especially for data-intensive operations
4. **Traceability**: Clear request/operation IDs for tracking related log entries
5. **Searchability**: Well-structured logs for easy filtering and querying

## Log Levels and Usage

The project uses Python's standard logging levels with specific guidelines:

| Level | Numeric Value | Usage |
|-------|--------------|-------|
| DEBUG | 10 | Detailed information for diagnosing problems (data samples, intermediate calculations) |
| INFO | 20 | Confirmation that things are working as expected (pipeline steps, feature calculations) |
| WARNING | 30 | Indication that something unexpected happened, but the process can continue (data quality issues, missing values) |
| ERROR | 40 | Error conditions that prevent a function from working properly (API failures, calculation errors) |
| CRITICAL | 50 | Critical errors that prevent the system from continuing (database connection failures, data corruption) |

### Level Guidelines by Component

| Component | Default Level | Details |
|-----------|---------------|---------|
| API Client | INFO | DEBUG for request/response details in development |
| Data Processing | INFO | DEBUG for transformation samples |
| Feature Engineering | INFO | Track feature calculation with sampling at DEBUG |
| Model Training | INFO | Detailed metrics at DEBUG |
| Prediction | INFO | Prediction details at DEBUG |
| Airflow | INFO | Task transitions, execution times |
| Web Dashboard | WARNING | Client interactions at INFO |

## Directory Structure

Logs are organized hierarchically to balance centralization and component separation:

```
logs/
├── airflow/
│   ├── dags/
│   │   ├── data_collection_YYYY-MM-DD.log
│   │   ├── data_processing_YYYY-MM-DD.log
│   │   ├── feature_engineering_YYYY-MM-DD.log
│   │   ├── model_training_YYYY-MM-DD.log
│   │   └── prediction_YYYY-MM-DD.log
│   └── scheduler/
│       └── scheduler_YYYY-MM-DD.log
├── api/
│   └── espn_api_YYYY-MM-DD.log
├── data/
│   ├── ingestion_YYYY-MM-DD.log
│   └── transformation_YYYY-MM-DD.log
├── features/
│   ├── team_features_YYYY-MM-DD.log
│   ├── player_features_YYYY-MM-DD.log
│   └── game_features_YYYY-MM-DD.log
├── models/
│   ├── training_YYYY-MM-DD.log
│   ├── evaluation_YYYY-MM-DD.log
│   └── prediction_YYYY-MM-DD.log
├── dashboard/
│   └── dashboard_YYYY-MM-DD.log
└── app_YYYY-MM-DD.log
```

## Log Format

The project uses a structured JSON log format with consistent fields:

```json
{
  "timestamp": "2025-03-25T10:23:54.123Z",
  "level": "INFO",
  "logger": "ncaa_basketball.features.team_features",
  "process_id": 12345,
  "thread_id": 123456,
  "trace_id": "abcd1234",
  "operation_id": "feat_calc_team_123",
  "message": "Calculated rolling average points for team: North Carolina",
  "extra": {
    "team_id": "52",
    "window_size": 10,
    "calculation_time_ms": 145,
    "feature_count": 25
  }
}
```

### Standard Fields

- `timestamp`: ISO 8601 format with millisecond precision
- `level`: Log level (DEBUG, INFO, etc.)
- `logger`: Hierarchical logger name
- `process_id`: Operating system process identifier
- `thread_id`: Thread identifier for multi-threaded operations
- `trace_id`: Unique identifier for tracing request flows
- `operation_id`: Identifier for specific operations (e.g., feature calculation batch)
- `message`: Human-readable log message
- `extra`: Component-specific structured data

## Log Rotation and Retention

The project implements the following log rotation and retention policies:

- **Rotation**: Daily rotation with timestamp suffix
- **Compression**: Logs older than 1 day are compressed with gzip
- **Retention**: 
  - Critical logs: 1 year
  - Error logs: 6 months
  - Warning logs: 3 months
  - Info logs: 1 month
  - Debug logs: 1 week

### Implementation

Log rotation is handled using Python's `RotatingFileHandler` with custom retention logic:

```python
import logging
import logging.handlers
import os
import datetime
import gzip
import shutil

# Set up rotation
log_handler = logging.handlers.TimedRotatingFileHandler(
    filename="logs/features/team_features.log",
    when="midnight",
    interval=1,
    backupCount=30  # Keep 30 days of logs
)

# Implement compression for rotated logs
def compress_rotated_log(source_path):
    with open(source_path, 'rb') as f_in:
        with gzip.open(f"{source_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source_path)

log_handler.rotator = compress_rotated_log
```

## Logging Implementation

### Base Logger Configuration

The project uses a centralized logger configuration in `src/config/logging.py`:

```python
import logging
import logging.config
import os
import json
from datetime import datetime
import uuid
from pathlib import Path

# Ensure log directories exist
def ensure_log_dirs():
    log_dirs = [
        "logs/airflow/dags",
        "logs/airflow/scheduler",
        "logs/api",
        "logs/data",
        "logs/features",
        "logs/models",
        "logs/dashboard"
    ]
    for dir_path in log_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

ensure_log_dirs()

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self):
        super(JsonFormatter, self).__init__()
        
    def format(self, record):
        logobj = {}
        logobj["timestamp"] = datetime.utcnow().isoformat() + "Z"
        logobj["level"] = record.levelname
        logobj["logger"] = record.name
        logobj["process_id"] = record.process
        logobj["thread_id"] = record.thread
        
        # Extract trace_id and operation_id if available
        logobj["trace_id"] = getattr(record, "trace_id", str(uuid.uuid4())[:8])
        logobj["operation_id"] = getattr(record, "operation_id", None)
        
        # Format the message
        logobj["message"] = record.getMessage()
        
        # Add extra fields
        if hasattr(record, "extra"):
            logobj["extra"] = record.extra
        
        return json.dumps(logobj)

def configure_logging(config=None):
    """Configure logging based on project settings"""
    
    if config is None:
        # Default configuration
        log_level = os.environ.get("LOG_LEVEL", "INFO")
    else:
        # Use configuration from settings
        log_level = config.get("log_level", "INFO")
    
    # Base config with console and default handlers
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": "logs/app.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "ncaa_basketball": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # Add component-specific file handlers
    components = [
        ("ncaa_basketball.api", "logs/api/espn_api"),
        ("ncaa_basketball.data.ingestion", "logs/data/ingestion"),
        ("ncaa_basketball.data.transformation", "logs/data/transformation"),
        ("ncaa_basketball.features.team_features", "logs/features/team_features"),
        ("ncaa_basketball.features.player_features", "logs/features/player_features"),
        ("ncaa_basketball.features.game_features", "logs/features/game_features"),
        ("ncaa_basketball.models.training", "logs/models/training"),
        ("ncaa_basketball.models.evaluation", "logs/models/evaluation"),
        ("ncaa_basketball.models.prediction", "logs/models/prediction"),
        ("ncaa_basketball.visualization", "logs/dashboard/dashboard")
    ]
    
    for logger_name, log_path in components:
        handler_name = logger_name.split(".")[-1] + "_file"
        
        # Add handler config
        logging_config["handlers"][handler_name] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": f"{log_path}.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf-8"
        }
        
        # Add logger config
        logging_config["loggers"][logger_name] = {
            "level": log_level,
            "handlers": ["console", handler_name],
            "propagate": False
        }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Return root logger
    return logging.getLogger("ncaa_basketball")

# Singleton pattern for project logger
def get_logger(name=None):
    """Get a logger instance with the given name"""
    if name is None:
        return logging.getLogger("ncaa_basketball")
    
    return logging.getLogger(f"ncaa_basketball.{name}")
```

### Using the Logger in Components

Example usage in a feature calculation module:

```python
from src.config.logging import get_logger
import uuid

# Get component logger
logger = get_logger("features.team_features")

def calculate_team_features(team_id, season_id, window_sizes, incremental=True):
    """Calculate team features for modeling"""
    
    # Generate operation ID for this feature calculation
    operation_id = f"team_feat_{team_id}_{season_id}_{uuid.uuid4()[:8]}"
    
    # Log with extra context
    logger.info(
        f"Starting feature calculation for team {team_id}, season {season_id}",
        extra={
            "operation_id": operation_id,
            "team_id": team_id,
            "season_id": season_id,
            "window_sizes": window_sizes,
            "incremental": incremental
        }
    )
    
    try:
        # Calculation logic...
        
        # Log progress
        logger.debug(
            f"Calculated window features for team {team_id}",
            extra={
                "operation_id": operation_id,
                "window_size": 10,
                "metrics": ["points_avg", "defensive_rating_avg"],
                "sample_count": 30
            }
        )
        
        # Final success log
        logger.info(
            f"Completed feature calculation for team {team_id}, season {season_id}",
            extra={
                "operation_id": operation_id,
                "feature_count": 150,
                "execution_time_ms": 2500
            }
        )
        
        return feature_data
        
    except Exception as e:
        # Log error with full context
        logger.error(
            f"Error calculating features for team {team_id}: {str(e)}",
            extra={
                "operation_id": operation_id,
                "error_type": type(e).__name__,
                "team_id": team_id
            },
            exc_info=True  # Include stack trace
        )
        raise
```

## Airflow Integration

Airflow has its own logging system that is integrated with the project's logging:

```python
# In DAG definition
from src.config.logging import get_logger
import uuid

logger = get_logger("airflow.dags.feature_engineering")

def _calculate_features(**context):
    """Task function for feature calculation"""
    # Generate trace ID for this DAG run
    trace_id = context['run_id']
    
    # Use task instance for operation ID
    operation_id = f"{context['task_instance'].task_id}_{context['execution_date']}"
    
    # Add trace/operation to task context for logging
    context['task_instance'].xcom_push(key='trace_id', value=trace_id)
    context['task_instance'].xcom_push(key='operation_id', value=operation_id)
    
    # Log with trace context
    logger.info(
        "Starting feature calculation task",
        extra={
            "trace_id": trace_id,
            "operation_id": operation_id,
            "airflow_dag_id": context['dag'].dag_id,
            "airflow_task_id": context['task'].task_id,
            "execution_date": context['execution_date'].isoformat()
        }
    )
    
    # Feature calculation logic...
    
    return result
```

## Log Analysis and Monitoring

The project implements several approaches for log analysis:

### 1. Log Parsing Script

A utility script (`tools/log_analysis.py`) to parse and analyze logs:

```python
import json
import glob
import pandas as pd
from pathlib import Path

def parse_logs(log_dir, component=None, level=None, start_date=None, end_date=None):
    """Parse logs into a DataFrame for analysis"""
    
    # Determine which log files to process
    if component:
        log_files = list(Path(log_dir).glob(f"**/{component}*.log*"))
    else:
        log_files = list(Path(log_dir).glob("**/*.log*"))
    
    # Read and parse logs
    log_entries = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    log_entries.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(log_entries)
    
    # Apply filters
    if level:
        df = df[df['level'] == level]
    
    if start_date:
        df = df[pd.to_datetime(df['timestamp']) >= pd.to_datetime(start_date)]
    
    if end_date:
        df = df[pd.to_datetime(df['timestamp']) <= pd.to_datetime(end_date)]
    
    return df

def analyze_errors(log_dir, component=None, days=7):
    """Analyze error patterns in logs"""
    df = parse_logs(
        log_dir, 
        component=component, 
        level="ERROR",
        start_date=pd.Timestamp.now() - pd.Timedelta(days=days)
    )
    
    # Get error counts by component
    error_counts = df['logger'].value_counts()
    
    # Get most common error messages
    error_messages = df['message'].value_counts().head(10)
    
    # Time series of errors
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    errors_by_day = df.groupby('date').size()
    
    return {
        'total_errors': len(df),
        'error_counts_by_component': error_counts,
        'most_common_errors': error_messages,
        'errors_by_day': errors_by_day
    }

# Example usage
error_analysis = analyze_errors('logs', component='features', days=7)
print(f"Total errors in last 7 days: {error_analysis['total_errors']}")
print("\nErrors by component:")
print(error_analysis['error_counts_by_component'])
print("\nMost common errors:")
print(error_analysis['most_common_errors'])
```

### 2. Integration with Dashboard

The logging system integrates with the Plotly Dash application for real-time monitoring:

```python
# In dashboard.py
import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import pandas as pd
from src.config.logging import get_logger
from tools.log_analysis import parse_logs

logger = get_logger("visualization.dashboard")

# Dashboard components
def create_logging_tab():
    """Create a logging monitoring tab for the dashboard"""
    return html.Div([
        html.H3("System Logs"),
        
        html.Div([
            html.Label("Component:"),
            dcc.Dropdown(
                id='log-component-dropdown',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'API Client', 'value': 'api'},
                    {'label': 'Data Processing', 'value': 'data'},
                    {'label': 'Feature Engineering', 'value': 'features'},
                    {'label': 'Model Training', 'value': 'models'},
                    {'label': 'Prediction', 'value': 'prediction'}
                ],
                value='all'
            ),
            
            html.Label("Time Range:"),
            dcc.DatePickerRange(
                id='log-date-range',
                start_date=pd.Timestamp.now() - pd.Timedelta(days=7),
                end_date=pd.Timestamp.now()
            ),
            
            html.Label("Log Level:"),
            dcc.Dropdown(
                id='log-level-dropdown',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Debug', 'value': 'DEBUG'},
                    {'label': 'Info', 'value': 'INFO'},
                    {'label': 'Warning', 'value': 'WARNING'},
                    {'label': 'Error', 'value': 'ERROR'},
                    {'label': 'Critical', 'value': 'CRITICAL'}
                ],
                value='ERROR'
            ),
            
            html.Button('Refresh', id='log-refresh-button')
        ]),
        
        dcc.Graph(id='log-timeline-graph'),
        
        html.H4("Log Entries"),
        html.Div(id='log-entries-table')
    ])

@callback(
    [Output('log-timeline-graph', 'figure'),
     Output('log-entries-table', 'children')],
    [Input('log-refresh-button', 'n_clicks'),
     Input('log-component-dropdown', 'value'),
     Input('log-date-range', 'start_date'),
     Input('log-date-range', 'end_date'),
     Input('log-level-dropdown', 'value')]
)
def update_log_displays(n_clicks, component, start_date, end_date, level):
    # Log the monitoring action
    logger.info(
        "Log monitoring dashboard refreshed",
        extra={
            "component_filter": component,
            "level_filter": level,
            "date_range": f"{start_date} to {end_date}"
        }
    )
    
    # Component filter mapping
    component_map = {
        'all': None,
        'api': 'api',
        'data': 'data',
        'features': 'features',
        'models': 'models',
        'prediction': 'prediction'
    }
    
    # Level filter
    level_filter = None if level == 'all' else level
    
    # Parse logs
    logs_df = parse_logs(
        'logs',
        component=component_map.get(component),
        level=level_filter,
        start_date=start_date,
        end_date=end_date
    )
    
    # Create timeline figure
    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
    logs_df['hour'] = logs_df['timestamp'].dt.floor('H')
    count_by_hour = logs_df.groupby(['hour', 'level']).size().reset_index(name='count')
    
    fig = px.line(
        count_by_hour, 
        x='hour', 
        y='count', 
        color='level',
        title='Log Entries Over Time'
    )
    
    # Create log entries table
    if len(logs_df) > 100:
        logs_df = logs_df.tail(100)  # Limit to last 100 logs
    
    table_data = logs_df[['timestamp', 'level', 'logger', 'message']].to_dict('records')
    table = create_data_table(table_data, id='log-table', pagination=True)
    
    return fig, table
```

## Best Practices

### 1. Contextual Logging

Always include relevant context with log messages:

```python
# BAD - Missing context
logger.info("Feature calculation complete")

# GOOD - Includes context
logger.info(
    "Feature calculation complete",
    extra={
        "team_id": team_id,
        "feature_count": 125,
        "execution_time_ms": 1500
    }
)
```

### 2. Structured Error Handling

Use try/except blocks with appropriate logging:

```python
try:
    # Operation that might fail
    result = process_data(data)
    return result
except ValueError as e:
    # Log specific error with context
    logger.error(
        f"Invalid data format: {str(e)}",
        extra={
            "data_id": data_id,
            "error_location": "process_data"
        }
    )
    raise  # Re-raise if needed
except Exception as e:
    # Log unexpected errors
    logger.critical(
        f"Unexpected error in data processing: {str(e)}",
        extra={
            "data_id": data_id
        },
        exc_info=True  # Include stack trace
    )
    raise
```

### 3. Performance Considerations

Avoid expensive logging operations in critical paths:

```python
# Only construct expensive debug logs when the level is enabled
if logger.isEnabledFor(logging.DEBUG):
    detailed_stats = calculate_detailed_stats(data)  # Expensive operation
    logger.debug(
        "Detailed statistics calculated",
        extra={"stats": detailed_stats}
    )
```

### 4. Sensitive Data Handling

Avoid logging sensitive information:

```python
# BAD - Logs API key
logger.debug(f"API request with key: {api_key}")

# GOOD - Masks sensitive data
logger.debug(
    "API request initiated",
    extra={
        "api_key_used": bool(api_key),
        "api_endpoint": endpoint
    }
)
```

## Implementation in Feature Engineering

Example implementation in the feature engineering component:

```python
# src/features/base.py
from src.config.logging import get_logger
import uuid
import time
from typing import Dict, Any, List, Optional
import pandas as pd

logger = get_logger("features.base")

class BaseFeature:
    """Base class for all features"""
    
    def __init__(self, name: str, version: int = 1):
        self.name = name
        self.version = version
        self.logger = get_logger(f"features.{name}")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate feature values"""
        # Generate operation ID
        operation_id = f"{self.name}_v{self.version}_{uuid.uuid4()[:8]}"
        
        start_time = time.time()
        
        # Log calculation start
        self.logger.info(
            f"Starting calculation of feature: {self.name}",
            extra={
                "operation_id": operation_id,
                "feature_version": self.version,
                "input_rows": len(data),
                "kwargs": kwargs
            }
        )
        
        try:
            # Actual calculation (implemented by subclasses)
            result = self._calculate_impl(data, operation_id=operation_id, **kwargs)
            
            # Log success
            execution_time = time.time() - start_time
            self.logger.info(
                f"Feature calculation complete: {self.name}",
                extra={
                    "operation_id": operation_id,
                    "feature_version": self.version,
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "output_rows": len(result),
                    "feature_columns": result.columns.tolist()
                }
            )
            
            return result
            
        except Exception as e:
            # Log failure
            execution_time = time.time() - start_time
            self.logger.error(
                f"Feature calculation failed: {self.name}",
                extra={
                    "operation_id": operation_id,
                    "feature_version": self.version,
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise
    
    def _calculate_impl(self, data: pd.DataFrame, operation_id: str, **kwargs) -> pd.DataFrame:
        """Implementation of feature calculation (to be overridden by subclasses)"""
        raise NotImplementedError("Subclasses must implement _calculate_impl")
    
    def _log_progress(self, operation_id: str, progress: float, message: str, **extra):
        """Log calculation progress"""
        self.logger.info(
            message,
            extra={
                "operation_id": operation_id,
                "progress": progress,
                **extra
            }
        )
```

## Airflow DAG Logging Example

Example implementation in an Airflow DAG:

```python
# airflow/dags/data_collection/espn_teams_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import uuid

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.config.logging import get_logger, configure_logging
from src.config.settings import load_config

# Initialize logging and config
config = load_config()
logger = get_logger("airflow.dags.data_collection")

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'espn_teams_collection',
    default_args=default_args,
    description='Collect team data from ESPN API',
    schedule_interval='0 4 * * *',  # Daily at 4 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['data_collection', 'espn', 'teams'],
) as dag:
    
    # Function to create trace context for the entire DAG run
    def create_dag_trace_context(**context):
        trace_id = f"dag_{context['dag_run'].run_id[-8:]}"
        
        # Log DAG start
        logger.info(
            f"Starting DAG: {context['dag'].dag_id}",
            extra={
                "trace_id": trace_id,
                "airflow_dag_id": context['dag'].dag_id,
                "airflow_run_id": context['dag_run'].run_id,
                "execution_date": context['execution_date'].isoformat()
            }
        )
        
        # Store trace ID for task instances
        context['ti'].xcom_push(key='trace_id', value=trace_id)
        
        return trace_id
    
    # Task to fetch teams for a season with proper logging
    def fetch_teams_for_season(year, **context):
        # Get trace ID from DAG context
        trace_id = context['ti'].xcom_pull(task_ids='create_trace_context')
        
        # Create operation ID for this task
        operation_id = f"fetch_teams_{year}_{context['task_instance'].try_number}"
        
        # Get task-specific logger
        task_logger = get_logger("airflow.dags.data_collection.fetch_teams")
        
        # Log task start
        task_logger.info(
            f"Starting task: fetch_teams_for_season, year={year}",
            extra={
                "trace_id": trace_id,
                "operation_id": operation_id,
                "year": year,
                "airflow_task_id": context['task'].task_id,
                "execution_date": context['execution_date'].isoformat()
            }
        )
        
        try:
            # Task implementation
            from src.api.client import ESPNApiClient
            from src.data.ingestion import save_raw_data
            
            # Create API client
            api_client = ESPNApiClient(
                base_url=config.espn_api.base_url,
                timeout=config.espn_api.request_timeout
            )
            
            # Fetch data (with detailed debug logging inside the client)
            task_logger.info(f"Fetching teams data for year {year}", extra={"trace_id": trace_id, "operation_id": operation_id})
            endpoint = config.espn_api.endpoints.teams.format(year=year)
            teams_data = api_client.get_all_paginated_data(endpoint)
            
            # Log data statistics
            task_logger.info(
                f"Retrieved {len(teams_data)} teams for year {year}",
                extra={
                    "trace_id": trace_id,
                    "operation_id": operation_id,
                    "team_count": len(teams_data),
                    "conferences": len(set(t.get('conference', {}).get('id') for t in teams_data if t.get('conference')))
                }
            )
            
            # Save data
            output_file = f"data/raw/teams/teams_{year}.json"
            save_raw_data(teams_data, output_file)
            
            # Log task completion
            task_logger.info(
                f"Task complete: fetch_teams_for_season, year={year}",
                extra={
                    "trace_id": trace_id,
                    "operation_id": operation_id,
                    "output_file": output_file,
                    "file_size_bytes": os.path.getsize(output_file)
                }
            )
            
            return f"Fetched {len(teams_data)} teams for {year}"
            
        except Exception as e:
            # Log task failure
            task_logger.error(
                f"Task failed: fetch_teams_for_season, year={year}",
                extra={
                    "trace_id": trace_id,
                    "operation_id": operation_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise
    
    # Create task to initialize trace context
    create_trace_context_task = PythonOperator(
        task_id='create_trace_context',
        python_callable=create_dag_trace_context,
        provide_context=True,
    )
    
    # Create tasks to fetch teams data
    current_year = config.espn_api.seasons.current_year
    
    fetch_current_year_task = PythonOperator(
        task_id=f'fetch_teams_{current_year}',
        python_callable=fetch_teams_for_season,
        op_kwargs={'year': current_year},
        provide_context=True,
    )
    
    # Define task dependencies
    create_trace_context_task >> fetch_current_year_task
```

## Conclusion

This logging strategy provides comprehensive visibility into all aspects of the NCAA Basketball Analytics project. It balances detail and performance while ensuring that operational issues can be quickly identified and resolved. The structured approach with consistent context and formatting makes it easy to analyze logs and monitor system health.
