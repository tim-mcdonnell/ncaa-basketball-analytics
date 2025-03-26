# Airflow Workflow Definitions

## Overview

This document outlines the Apache Airflow workflows (DAGs) for the NCAA Basketball Analytics project. These workflows orchestrate data collection, processing, feature engineering, model training, and prediction generation.

## Workflow Architecture

The project uses a modular approach to Airflow workflows, with separate DAGs for different responsibilities:

1. **Data Collection DAGs**: Fetch data from ESPN API
2. **Data Processing DAGs**: Transform raw data into structured formats
3. **Feature Engineering DAGs**: Calculate features for modeling
4. **Model Training DAGs**: Train and evaluate predictive models
5. **Prediction DAGs**: Generate predictions for upcoming games

## DAG Organization

All DAGs are stored in the `airflow/dags/` directory with the following structure:

```
airflow/dags/
├── __init__.py
├── constants.py                  # Shared constants across DAGs
├── utils/                        # Shared utilities
│   ├── __init__.py
│   ├── callbacks.py              # Custom callbacks
│   ├── operators.py              # Custom operators
│   └── sensors.py                # Custom sensors
├── data_collection/              # Data collection DAGs
│   ├── __init__.py
│   ├── espn_teams_dag.py
│   ├── espn_games_dag.py
│   ├── espn_players_dag.py
│   └── espn_rankings_dag.py
├── data_processing/              # Data processing DAGs
│   ├── __init__.py
│   ├── process_games_dag.py
│   ├── process_teams_dag.py
│   └── process_players_dag.py
├── feature_engineering/          # Feature engineering DAGs
│   ├── __init__.py
│   ├── team_features_dag.py
│   ├── player_features_dag.py
│   └── game_features_dag.py
├── model_training/               # Model training DAGs
│   ├── __init__.py
│   ├── train_lightgbm_dag.py
│   └── train_pytorch_dag.py
└── prediction/                   # Prediction DAGs
    ├── __init__.py
    └── generate_predictions_dag.py
```

## Core DAG Examples

### 1. Data Collection DAG: ESPN Teams

```python
"""
DAG to collect team data from ESPN API
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.api.client import ESPNApiClient
from src.config.settings import load_config
from src.data.ingestion import save_raw_data

config = load_config()

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
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Get current year and previous years to collect
    current_year = config.espn_api.seasons.current_year
    historical_years = list(range(config.espn_api.seasons.start_year, current_year + 1))
    
    # Create API client
    api_client = ESPNApiClient(
        base_url=config.espn_api.base_url,
        timeout=config.espn_api.request_timeout
    )
    
    # Define function to fetch teams for a season
    def fetch_teams_for_season(year, **kwargs):
        """Fetch team data for a specific season"""
        endpoint = config.espn_api.endpoints.teams.format(year=year)
        teams_data = api_client.get_all_paginated_data(endpoint)
        
        # Save raw data as JSON
        output_file = f"data/raw/teams/teams_{year}.json"
        save_raw_data(teams_data, output_file)
        
        return f"Fetched {len(teams_data)} teams for {year}"
    
    # Group tasks by recent vs historical data
    with TaskGroup(group_id='recent_seasons') as recent_seasons_group:
        for year in range(current_year - 2, current_year + 1):
            fetch_task = PythonOperator(
                task_id=f'fetch_teams_{year}',
                python_callable=fetch_teams_for_season,
                op_kwargs={'year': year},
            )
    
    with TaskGroup(group_id='historical_seasons') as historical_seasons_group:
        for year in range(config.espn_api.seasons.start_year, current_year - 2):
            # Use a single task to fetch historical data in batches
            if year % 5 == 0:  # Create tasks in batches of 5 years
                years_batch = list(range(year, min(year + 5, current_year - 2)))
                
                def fetch_teams_batch(years, **kwargs):
                    results = []
                    for year in years:
                        endpoint = config.espn_api.endpoints.teams.format(year=year)
                        teams_data = api_client.get_all_paginated_data(endpoint)
                        
                        output_file = f"data/raw/teams/teams_{year}.json"
                        save_raw_data(teams_data, output_file)
                        
                        results.append(f"Fetched {len(teams_data)} teams for {year}")
                    return "\n".join(results)
                
                fetch_batch_task = PythonOperator(
                    task_id=f'fetch_teams_{year}_to_{year + len(years_batch) - 1}',
                    python_callable=fetch_teams_batch,
                    op_kwargs={'years': years_batch},
                )
    
    # Define execution order
    start >> recent_seasons_group >> historical_seasons_group >> end
```

### 2. Data Processing DAG: Process Games

```python
"""
DAG to process raw game data into structured format
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.data.transformation import process_raw_games
from src.config.settings import load_config

config = load_config()

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'process_games_data',
    default_args=default_args,
    description='Process raw game data into structured format',
    schedule_interval='0 6 * * *',  # Daily at 6 AM (after data collection)
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['data_processing', 'games'],
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Get current year 
    current_year = config.espn_api.seasons.current_year
    
    # Sensor to wait for current year data
    wait_for_current_data = FileSensor(
        task_id=f'wait_for_current_data_{current_year}',
        filepath=f'data/raw/games/games_{current_year}.json',
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60 * 2,  # 2 hour timeout
        mode='reschedule',  # Don't block a worker
    )
    
    # Process current year data
    process_current_data = PythonOperator(
        task_id=f'process_games_{current_year}',
        python_callable=process_raw_games,
        op_kwargs={
            'year': current_year,
            'input_file': f'data/raw/games/games_{current_year}.json',
            'output_directory': f'{config.duckdb.storage.parquet_directory}/games',
            'db_path': config.duckdb.database_path
        }
    )
    
    # Check if we need to reprocess historical data (weekly)
    def should_process_historical():
        # Process historical data on Mondays
        return datetime.now().weekday() == 0
    
    check_historical = PythonOperator(
        task_id='check_historical',
        python_callable=should_process_historical,
    )
    
    # Process historical data if needed
    process_historical = PythonOperator(
        task_id='process_historical_games',
        python_callable=lambda **kwargs: [
            process_raw_games(
                year=year,
                input_file=f'data/raw/games/games_{year}.json',
                output_directory=f'{config.duckdb.storage.parquet_directory}/games',
                db_path=config.duckdb.database_path
            )
            for year in range(config.espn_api.seasons.start_year, current_year)
        ],
        trigger_rule='all_done',  # Run even if check_historical returns False
    )
    
    start >> wait_for_current_data >> process_current_data
    start >> check_historical >> process_historical 
    [process_current_data, process_historical] >> end
```

### 3. Feature Engineering DAG: Team Features

```python
"""
DAG to generate team features for modeling
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.features.team_features import calculate_team_features
from src.config.settings import load_config

config = load_config()

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'generate_team_features',
    default_args=default_args,
    description='Generate team features for modeling',
    schedule_interval='0 8 * * *',  # Daily at 8 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['feature_engineering', 'teams'],
    params={
        'force_full_recalculation': Param(False, type='boolean', description='Force recalculation of all features'),
    },
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Branch based on whether we're doing incremental or full recalculation
    def determine_calculation_type(**context):
        # Check if force_full_recalculation is set
        if context['params'].get('force_full_recalculation', False):
            return 'full_recalculation'
            
        # Check the day of week - do full recalculation on Sundays
        if datetime.now().weekday() == 6:
            return 'full_recalculation'
            
        # Otherwise do incremental
        return 'incremental_calculation'
    
    branch_task = BranchPythonOperator(
        task_id='determine_calculation_type',
        python_callable=determine_calculation_type,
    )
    
    # Full recalculation task
    full_recalculation = PythonOperator(
        task_id='full_recalculation',
        python_callable=calculate_team_features,
        op_kwargs={
            'incremental': False,
            'seasons': list(range(config.espn_api.seasons.start_year, config.espn_api.seasons.current_year + 1)),
            'window_sizes': config.features.computation.window_sizes,
            'output_directory': config.features.storage.directory,
            'db_path': config.duckdb.database_path
        }
    )
    
    # Incremental calculation task
    incremental_calculation = PythonOperator(
        task_id='incremental_calculation',
        python_callable=calculate_team_features,
        op_kwargs={
            'incremental': True,
            'seasons': [config.espn_api.seasons.current_year],
            'window_sizes': config.features.computation.window_sizes,
            'output_directory': config.features.storage.directory,
            'db_path': config.duckdb.database_path
        }
    )
    
    # Join the branches
    join = EmptyOperator(
        task_id='join',
        trigger_rule='one_success',
    )
    
    # Log feature statistics
    def log_feature_stats(**context):
        calculation_type = context['ti'].xcom_pull(task_ids='determine_calculation_type')
        if calculation_type == 'full_recalculation':
            print("Completed full recalculation of team features")
        else:
            print("Completed incremental calculation of team features")
        
        # In a real implementation, we would query the database
        # to get feature statistics and log them
        return "Feature calculation complete"
    
    log_stats = PythonOperator(
        task_id='log_feature_statistics',
        python_callable=log_feature_stats,
    )
    
    start >> branch_task
    branch_task >> [full_recalculation, incremental_calculation]
    [full_recalculation, incremental_calculation] >> join >> log_stats >> end
```

### 4. Model Training DAG: PyTorch Model

```python
"""
DAG to train PyTorch neural network model
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from datetime import datetime, timedelta
import os
import sys
import mlflow
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.models.training import train_pytorch_model
from src.models.evaluation import evaluate_model
from src.config.settings import load_config

config = load_config()

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'train_pytorch_model',
    default_args=default_args,
    description='Train PyTorch neural network model',
    schedule_interval='0 1 * * 0',  # Weekly on Sunday at 1 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model_training', 'pytorch'],
    params={
        'experiment_name': Param('ncaa-basketball-predictions', type='string'),
        'model_type': Param('lstm', type='string', enum=['mlp', 'lstm', 'transformer']),
    },
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Set up MLflow tracking
    def setup_mlflow(**context):
        mlflow.set_tracking_uri(config.models.mlflow.tracking_uri)
        experiment_name = context['params'].get('experiment_name', config.models.mlflow.experiment_name)
        
        # Create experiment if it doesn't exist
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(config.models.mlflow.model_registry, experiment_name)
            )
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    setup_mlflow_task = PythonOperator(
        task_id='setup_mlflow',
        python_callable=setup_mlflow,
    )
    
    # Prepare training data
    def prepare_training_data(**context):
        """Load features and prepare datasets for model training"""
        import polars as pl
        import duckdb
        
        # Connect to database
        conn = duckdb.connect(config.duckdb.database_path)
        
        # Get the latest feature dataset
        features_query = f"""
        SELECT * FROM '{config.features.storage.directory}/team_features_latest.parquet'
        WHERE season >= {config.espn_api.seasons.current_year - 5}
        """
        
        features_df = pl.from_arrow(conn.execute(features_query).arrow())
        
        # In a real implementation, we would:
        # 1. Join with game outcomes
        # 2. Create train/test split
        # 3. Normalize features
        # 4. Create sequences for LSTM if needed
        
        # For now, just return some sample info about the dataset
        return {
            "num_samples": features_df.shape[0],
            "num_features": features_df.shape[1] - 3,  # Subtract ID columns
            "seasons": features_df.select("season").unique().sort().to_dict(as_series=False),
            "feature_names": features_df.columns
        }
    
    prepare_data_task = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data,
    )
    
    # Train model
    def train_model(**context):
        """Train PyTorch model with the prepared data"""
        # In a real implementation, this would:
        # 1. Load the prepared data from XCom
        # 2. Configure the model based on params
        # 3. Train using PyTorch
        # 4. Track with MLflow
        # 5. Save the best model
        
        model_type = context['params'].get('model_type', 'lstm')
        experiment_id = context['ti'].xcom_pull(task_ids='setup_mlflow')
        
        # Would call the actual training function
        # model, history = train_pytorch_model(...)
        
        # For this example, just return mock results
        results = {
            "model_type": model_type,
            "experiment_id": experiment_id,
            "training_accuracy": 0.82,
            "validation_accuracy": 0.78,
            "training_time_seconds": 1200,
            "model_path": f"data/models/pytorch_{model_type}_{datetime.now().strftime('%Y%m%d')}.pth"
        }
        
        # In a real implementation, we would log to MLflow
        return results
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    # Evaluate model
    def evaluate_model_performance(**context):
        """Evaluate model on test data"""
        training_results = context['ti'].xcom_pull(task_ids='train_model')
        
        # In a real implementation, this would:
        # 1. Load the trained model
        # 2. Evaluate on held-out test data
        # 3. Calculate performance metrics
        # 4. Log to MLflow
        
        # For this example, just return mock results
        evaluation_results = {
            "model_type": training_results["model_type"],
            "test_accuracy": 0.76,
            "test_rmse": 8.5,
            "test_mae": 6.2,
            "baseline_accuracy": 0.70,
            "performance_improvement": "+8.6%"
        }
        
        return evaluation_results
    
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_performance,
    )
    
    # Register model in MLflow if it's good enough
    def register_model_if_improved(**context):
        """Register model in MLflow if it performs better than current production model"""
        evaluation_results = context['ti'].xcom_pull(task_ids='evaluate_model')
        training_results = context['ti'].xcom_pull(task_ids='train_model')
        
        # In a real implementation, we would:
        # 1. Compare to current production model
        # 2. Register in MLflow if better
        # 3. Update model version
        
        # For this example:
        if evaluation_results["test_accuracy"] > 0.75:
            # Would register model in MLflow
            # mlflow.pytorch.log_model(...)
            # mlflow.register_model(...)
            
            return {
                "registered": True,
                "model_name": f"pytorch_{training_results['model_type']}",
                "version": 1,
                "accuracy": evaluation_results["test_accuracy"]
            }
        else:
            return {
                "registered": False,
                "reason": "Performance below threshold"
            }
    
    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model_if_improved,
    )
    
    start >> setup_mlflow_task >> prepare_data_task >> train_model_task >> evaluate_model_task >> register_model_task >> end
```

### 5. Prediction DAG: Generate Predictions

```python
"""
DAG to generate predictions for upcoming games
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os
import sys
import mlflow
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.models.prediction import generate_predictions
from src.config.settings import load_config

config = load_config()

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'generate_game_predictions',
    default_args=default_args,
    description='Generate predictions for upcoming games',
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['prediction', 'games'],
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')
    
    # Get upcoming games
    def fetch_upcoming_games(**context):
        """Fetch list of upcoming games to predict"""
        import duckdb
        import polars as pl
        from datetime import datetime, timedelta
        
        # Connect to database
        conn = duckdb.connect(config.duckdb.database_path)
        
        # Get games in the next 7 days
        today = datetime.now().strftime('%Y-%m-%d')
        next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT * FROM '{config.duckdb.storage.parquet_directory}/games/games_{config.espn_api.seasons.current_year}.parquet'
        WHERE game_date BETWEEN '{today}' AND '{next_week}'
        AND home_score IS NULL  -- Games that haven't been played yet
        """
        
        upcoming_games = pl.from_arrow(conn.execute(query).arrow())
        
        # In a real implementation, we would return actual game data
        # For now, just simulate some upcoming games
        if upcoming_games.shape[0] == 0:
            # Create some mock data if no real games found
            upcoming_games = pl.DataFrame({
                "game_id": ["g1", "g2", "g3"],
                "game_date": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)],
                "home_team_id": ["t1", "t3", "t5"],
                "away_team_id": ["t2", "t4", "t6"],
                "season": [config.espn_api.seasons.current_year] * 3
            })
        
        # Save to a temporary location
        upcoming_games.write_parquet(f"{config.duckdb.storage.parquet_directory}/temp/upcoming_games.parquet")
        
        return {
            "num_games": upcoming_games.shape[0],
            "date_range": [today, next_week]
        }
    
    fetch_games_task = PythonOperator(
        task_id='fetch_upcoming_games',
        python_callable=fetch_upcoming_games,
    )
    
    # Load best model from MLflow
    def load_best_model(**context):
        """Load the best model from MLflow registry"""
        mlflow.set_tracking_uri(config.models.mlflow.tracking_uri)
        
        # In a real implementation, we would:
        # 1. Query MLflow for the latest registered model
        # 2. Load the model
        # 3. Return model info
        
        # For this example, just return mock info
        model_info = {
            "model_type": "pytorch_lstm",
            "version": 1,
            "accuracy": 0.78,
            "registered_date": datetime.now().strftime('%Y-%m-%d'),
            "loaded": True
        }
        
        return model_info
    
    load_model_task = PythonOperator(
        task_id='load_best_model',
        python_callable=load_best_model,
    )
    
    # Generate predictions
    def predict_games(**context):
        """Generate predictions for upcoming games"""
        model_info = context['ti'].xcom_pull(task_ids='load_best_model')
        upcoming_games_info = context['ti'].xcom_pull(task_ids='fetch_upcoming_games')
        
        import duckdb
        import polars as pl
        
        # Load upcoming games
        upcoming_games = pl.read_parquet(f"{config.duckdb.storage.parquet_directory}/temp/upcoming_games.parquet")
        
        # In a real implementation, we would:
        # 1. Load team features
        # 2. Prepare inputs for each game
        # 3. Run model inference
        # 4. Store predictions
        
        # For this example, just create mock predictions
        predictions = []
        for i in range(upcoming_games.shape[0]):
            game = upcoming_games.row(i)
            predictions.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
                "predicted_home_win_prob": 0.65,
                "predicted_score_diff": 5.2,
                "confidence": 0.8,
                "model_version": model_info["version"],
                "prediction_date": datetime.now().strftime('%Y-%m-%d')
            })
        
        # Save predictions
        predictions_df = pl.DataFrame(predictions)
        predictions_df.write_parquet(f"{config.features.storage.directory}/predictions/predictions_{datetime.now().strftime('%Y%m%d')}.parquet")
        
        # Update latest predictions
        predictions_df.write_parquet(f"{config.features.storage.directory}/predictions/latest_predictions.parquet")
        
        return {
            "num_predictions": len(predictions),
            "prediction_date": datetime.now().strftime('%Y-%m-%d')
        }
    
    predict_task = PythonOperator(
        task_id='predict_games',
        python_callable=predict_games,
    )
    
    # Archive old predictions and update DuckDB
    def archive_and_update_db(**context):
        """Archive old predictions and update database"""
        import duckdb
        
        # Connect to database
        conn = duckdb.connect(config.duckdb.database_path)
        
        # Update the predictions table in DuckDB
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id VARCHAR,
            prediction_date DATE,
            home_team_id VARCHAR,
            away_team_id VARCHAR,
            predicted_home_win_prob DOUBLE,
            predicted_score_diff DOUBLE,
            confidence DOUBLE,
            model_version INTEGER,
            PRIMARY KEY (game_id, prediction_date)
        );
        
        -- Insert latest predictions
        INSERT OR REPLACE INTO game_predictions
        SELECT 
            game_id,
            prediction_date::DATE,
            home_team_id,
            away_team_id,
            predicted_home_win_prob,
            predicted_score_diff,
            confidence,
            model_version
        FROM '{config.features.storage.directory}/predictions/latest_predictions.parquet';
        """)
        
        return "Database updated with latest predictions"
    
    update_db_task = PythonOperator(
        task_id='archive_and_update_db',
        python_callable=archive_and_update_db,
    )
    
    start >> fetch_games_task
    start >> load_model_task
    [fetch_games_task, load_model_task] >> predict_task >> update_db_task >> end
```

## DAG Scheduling

The DAGs are scheduled to run at appropriate intervals:

| DAG | Schedule | Purpose |
|-----|----------|---------|
| Data Collection | Daily (4 AM) | Fetch latest data from ESPN API |
| Data Processing | Daily (6 AM) | Process raw data after collection |
| Feature Engineering | Daily (8 AM) | Calculate features for modeling |
| Model Training | Weekly (Sunday 1 AM) | Train new models with accumulated data |
| Prediction Generation | Daily (9 AM) | Generate predictions for upcoming games |

## Airflow Configuration

The project uses the following Airflow configuration:

```ini
# airflow.cfg excerpts

[core]
dags_folder = /path/to/ncaa-basketball-analytics/airflow/dags
load_examples = False
max_active_runs_per_dag = 1

[scheduler]
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
min_file_process_interval = 30
dag_dir_list_interval = 30
print_stats_interval = 30
run_duration = -1
num_runs = -1
processor_poll_interval = 1
min_file_parsing_loop_time = 1
scheduler_idle_sleep_time = 1
scheduler_health_check_threshold = 30

[webserver]
base_url = http://localhost:8080
web_server_host = 0.0.0.0
web_server_port = 8080
```

## Custom Airflow Extensions

The project includes several custom extensions to Airflow:

### Custom Operators

```python
# airflow/dags/utils/operators.py

from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
import duckdb
import polars as pl

class DuckDBQueryOperator(BaseOperator):
    """
    Execute a DuckDB query
    """
    
    @apply_defaults
    def __init__(
        self,
        db_path,
        sql,
        output_file=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self.sql = sql
        self.output_file = output_file
        
    def execute(self, context):
        conn = duckdb.connect(self.db_path)
        result = conn.execute(self.sql)
        
        if self.output_file:
            if self.output_file.endswith('.parquet'):
                result.arrow().to_pandas().to_parquet(self.output_file)
            elif self.output_file.endswith('.csv'):
                result.arrow().to_pandas().to_csv(self.output_file, index=False)
        
        return result.fetchall()
```

### Custom Sensors

```python
# airflow/dags/utils/sensors.py

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import os
import duckdb

class NewDataSensor(BaseSensorOperator):
    """
    Sensor to detect when new data is available
    """
    
    @apply_defaults
    def __init__(
        self,
        data_path,
        min_rows=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.min_rows = min_rows
        
    def poke(self, context):
        if not os.path.exists(self.data_path):
            return False
            
        # For parquet files, check row count
        if self.data_path.endswith('.parquet'):
            try:
                conn = duckdb.connect(':memory:')
                row_count = conn.execute(f"SELECT COUNT(*) FROM '{self.data_path}'").fetchone()[0]
                return row_count >= self.min_rows
            except:
                return False
                
        # For other files, just check if file exists and is not empty
        return os.path.getsize(self.data_path) > 0
```

## Best Practices for Airflow in This Project

1. **Modular DAG Design**: Each DAG has a single responsibility, making them easier to maintain.

2. **Task Idempotency**: Tasks are designed to be idempotent, allowing for retries and backfills.

3. **Effective XCom Usage**: XComs are used to pass small amounts of data between tasks.

4. **Error Handling**: Retry policies are defined for tasks that might fail temporarily.

5. **Resource Efficiency**: Tasks are grouped to minimize Airflow overhead.

6. **Code Reusability**: Common functions are extracted to modules to avoid duplication.

7. **Conditional Execution**: Branch operators are used to implement conditional logic.

8. **Parameter Support**: DAGs accept runtime parameters for flexibility.

9. **Monitoring and Logging**: Clear task organization for easier debugging.

10. **Data Verification**: Sensors verify data is available before processing begins.
