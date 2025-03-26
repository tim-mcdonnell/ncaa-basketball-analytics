# Tech Stack: Components and Integration

## Overview

This document outlines the technology stack for the NCAA Basketball Analytics project, detailing each component's role, benefits, and how they integrate with each other.

![Tech Stack Architecture](https://i.imgur.com/placeholder.png)

## Core Technologies

### Python 3.12

**Role**: Core programming language for all components of the system.

**Benefits**:
- Latest stable Python version with performance improvements
- Type annotation support for better code quality
- Pattern matching for cleaner data parsing
- New standard library features like `datetime` improvements

**Key Libraries**:
- `typing`: Advanced type hints
- `asyncio`: Asynchronous I/O for API requests
- `pathlib`: Object-oriented filesystem paths
- `contextlib`: Context management utilities

**Integration Points**:
- Base language for all custom code
- Runtime environment for Airflow, DuckDB, and Polars

### Apache Airflow

**Role**: Workflow orchestration platform for scheduling and monitoring data pipelines.

**Benefits**:
- Task dependency management
- Scheduling with cron-like expressions
- Robust error handling and retries
- Extensible with custom operators
- Web UI for monitoring and manual triggers

**Key Features Used**:
- DAGs (Directed Acyclic Graphs) for workflow definition
- TaskGroups for organizing related tasks
- Sensors for external dependency monitoring
- Executors for parallel task execution
- XComs for task-to-task communication

**Integration Points**:
- Orchestrates API data collection
- Triggers feature computation
- Schedules model training and evaluation
- Manages incremental vs. full recalculations

**Example Usage**:
```python
# Sample DAG for daily data collection
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'espn_data_collection',
    default_args=default_args,
    description='Collect ESPN API data',
    schedule=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    
    fetch_teams = PythonOperator(
        task_id='fetch_teams',
        python_callable=fetch_teams_data,
    )
    
    fetch_games = PythonOperator(
        task_id='fetch_games',
        python_callable=fetch_games_data,
    )
    
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_raw_data,
    )
    
    fetch_teams >> fetch_games >> process_data
```

### DuckDB

**Role**: Analytical database for storing, querying, and transforming sports data.

**Benefits**:
- Column-oriented storage optimized for analytics
- SQL interface for familiar query patterns
- Direct integration with Polars and Pandas
- In-process architecture with minimal setup
- Support for Parquet files as a storage format

**Key Features Used**:
- JSON functions for API data processing
- Window functions for time-series statistics
- Complex aggregations for feature engineering
- Efficient join operations for entity relationships
- Time-based partitioning for historical data

**Integration Points**:
- Primary storage for processed data
- Computation engine for complex queries
- Integration with Polars for data transformation
- Provides data for ML model training

**Example Usage**:
```python
import duckdb

# Connect to database
conn = duckdb.connect('data/ncaa_basketball.duckdb')

# Create table
conn.execute("""
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR PRIMARY KEY,
    date TIMESTAMP,
    home_team_id VARCHAR,
    away_team_id VARCHAR,
    home_score INTEGER,
    away_score INTEGER,
    season INTEGER,
    is_tournament BOOLEAN
)
""")

# Query with window functions for team performance
conn.execute("""
SELECT 
    team_id,
    game_date,
    points,
    AVG(points) OVER (
        PARTITION BY team_id 
        ORDER BY game_date 
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) as avg_points_last_5
FROM team_game_stats
ORDER BY team_id, game_date
""").fetchall()
```

### Polars

**Role**: Data manipulation library for efficient transformations and feature engineering.

**Benefits**:
- 10-100x faster than Pandas for many operations
- Memory efficient for large datasets
- Lazy evaluation for query optimization
- Native support for parallel execution
- Seamless integration with DuckDB

**Key Features Used**:
- Lazy execution for complex transformations
- Window functions for rolling statistics
- Predicate pushdown for efficient filtering
- GroupBy operations for aggregations
- Time-based operations for seasonal analysis

**Integration Points**:
- Primary data transformation library
- Feature engineering computations
- Preprocessing for ML models
- Data preparation for visualizations

**Example Usage**:
```python
import polars as pl

# Load data from DuckDB
df = pl.from_arrow(duckdb.sql("SELECT * FROM team_game_stats").arrow())

# Feature engineering with lazy evaluation
features = (
    df.lazy()
    .group_by("team_id")
    .sort("game_date")
    .select([
        pl.col("team_id"),
        pl.col("game_id"),
        pl.col("game_date"),
        pl.col("points"),
        pl.col("points").mean().over(
            "team_id", 
            pl.col("game_date").rolling_window(5)
        ).alias("rolling_avg_points_5"),
        pl.col("assists").mean().over(
            "team_id", 
            pl.col("game_date").rolling_window(5)
        ).alias("rolling_avg_assists_5"),
        # More features...
    ])
    .collect()
)
```

### Plotly Dash

**Role**: Interactive web dashboard for data visualization and exploration.

**Benefits**:
- Interactive charts and graphs
- React-based components with Python backend
- Callback system for dynamic updates
- Layout customization with Bootstrap
- Support for complex visualizations

**Key Features Used**:
- Interactive scatter plots and line charts
- Data tables with filtering
- Dropdown selectors for teams and players
- Date range selectors for time filtering
- Callback chains for drill-down analysis

**Integration Points**:
- Frontend for data exploration
- Visualization of model predictions
- Interactive team and player analysis
- Historical game analysis interface

**Example Usage**:
```python
import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import polars as pl
import duckdb

# Initialize app
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'])

# Layout
app.layout = html.Div([
    html.H1("NCAA Basketball Analytics"),
    html.Div([
        html.Label("Select Season:"),
        dcc.Dropdown(
            id='season-dropdown',
            options=[{'label': f"{year-1}-{year}", 'value': year} 
                     for year in range(2010, 2025)],
            value=2024
        ),
        html.Label("Select Team:"),
        dcc.Dropdown(id='team-dropdown')
    ]),
    dcc.Graph(id='team-performance-graph')
])

# Callbacks
@callback(
    Output('team-dropdown', 'options'),
    Input('season-dropdown', 'value')
)
def update_teams(season):
    conn = duckdb.connect('data/ncaa_basketball.duckdb')
    teams = conn.execute(f"""
        SELECT DISTINCT team_id, team_name 
        FROM teams 
        WHERE season = {season}
        ORDER BY team_name
    """).fetchall()
    return [{'label': name, 'value': id} for id, name in teams]

@callback(
    Output('team-performance-graph', 'figure'),
    Input('team-dropdown', 'value'),
    Input('season-dropdown', 'value')
)
def update_graph(team_id, season):
    if not team_id:
        return px.line()
    
    conn = duckdb.connect('data/ncaa_basketball.duckdb')
    data = conn.execute(f"""
        SELECT game_date, opponent_team_name, points, opponent_points
        FROM team_games
        WHERE team_id = '{team_id}' AND season = {season}
        ORDER BY game_date
    """).fetch_arrow_table()
    
    df = pl.from_arrow(data).to_pandas()
    
    fig = px.line(df, x='game_date', y=['points', 'opponent_points'],
                  title=f'Team Performance in {season-1}-{season} Season',
                  labels={'value': 'Points', 'game_date': 'Game Date'})
    
    return fig
```

### MLflow

**Role**: End-to-end machine learning lifecycle platform for experiment tracking, model management, and deployment.

**Benefits**:
- Experiment tracking with metrics and parameters
- Model versioning and lineage
- Model registry for deployment management
- Artifact storage for model files
- Language-agnostic design

**Key Features Used**:
- Experiment tracking for model comparison
- Hyperparameter tracking
- Model versioning
- Performance metric logging
- Model registry for production models

**Integration Points**:
- Track all model training experiments
- Store model artifacts and parameters
- Register production models
- Compare model versions
- Log feature importance and evaluation metrics

**Example Usage**:
```python
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set experiment
mlflow.set_experiment("ncaa-game-predictions")

# Start a run
with mlflow.start_run(run_name="lightgbm-model"):
    # Log parameters
    mlflow.log_param("model_type", "lightgbm")
    mlflow.log_param("num_leaves", 31)
    mlflow.log_param("learning_rate", 0.05)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log feature importance
    feature_importance = model.feature_importances_
    mlflow.log_dict(
        {feature: importance for feature, importance in zip(X_train.columns, feature_importance)},
        "feature_importance.json"
    )
```

### PyTorch

**Role**: Deep learning framework for building and training neural networks.

**Benefits**:
- Dynamic computational graph
- GPU acceleration
- Rich ecosystem of pre-built layers
- Efficient tensor operations
- Support for various neural network architectures

**Key Features Used**:
- LSTM networks for sequence modeling
- Custom datasets and data loaders
- Training loops with early stopping
- Optimizer selection and learning rate scheduling
- Model checkpointing and loading

**Integration Points**:
- Advanced predictive modeling
- Sequence modeling for game progressions
- Time series forecasting
- Feature extraction from raw data

**Example Usage**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define LSTM model for game prediction
class GamePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GamePredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'data/models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('data/models/best_model.pth'))
    return model
```

## Integration Architecture

The components are integrated into a cohesive system with the following data flow:

1. **Data Collection Pipeline**:
   - Python scripts use the ESPN API client to fetch data
   - Airflow orchestrates scheduled and on-demand fetches
   - Raw JSON data is stored as Parquet files
   - DuckDB loads and processes this raw data

2. **Data Processing Pipeline**:
   - DuckDB and Polars transform raw data into structured tables
   - Time-series relationships are established
   - Incremental processing tracks data changes
   - Processed data is stored in a medallion architecture

3. **Feature Engineering Pipeline**:
   - Feature registry defines all available features
   - Polars computes features efficiently using lazy evaluation
   - DuckDB handles complex SQL transformations
   - Features are stored as Parquet files with versioning

4. **Model Training Pipeline**:
   - MLflow tracks experiments and parameters
   - PyTorch models are trained on feature datasets
   - Model performance is evaluated and logged
   - Best models are registered for prediction

5. **Prediction Pipeline**:
   - Latest features are fed to registered models
   - Predictions are generated for upcoming games
   - Results are stored in DuckDB
   - Historical accuracy is tracked

6. **Visualization Layer**:
   - Dash app queries DuckDB for data
   - Interactive components enable data exploration
   - Predictions are presented with confidence intervals
   - Historical performance is visualized

## Deployment Considerations

For local development and hobby usage:

1. **Containerization**: Consider Docker for encapsulating dependencies
2. **Local Airflow**: Use Airflow's LocalExecutor for scheduling
3. **Version Control**: Use Git for tracking code changes
4. **Environment Management**: Use conda or venv for Python environment

The architecture is designed to scale easily if needed in the future, with minimal changes required to move to a more production-focused deployment.
