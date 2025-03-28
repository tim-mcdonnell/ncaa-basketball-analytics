# NCAA Basketball Analytics Airflow Pipelines

This directory contains the Airflow DAGs, operators, hooks, sensors, and utilities for orchestrating the NCAA basketball analytics pipeline.

## Overview

The pipeline consists of four main DAGs:

1. **Data Collection DAG**: Fetches team, game, player, and player statistics data from the ESPN API
2. **Feature Engineering DAG**: Calculates team, player, and game features for prediction models
3. **Model Training DAG**: Prepares training data, trains the model, evaluates performance, and registers the model
4. **Prediction DAG**: Fetches upcoming games, prepares prediction data, generates predictions, and stores results

## Setting Up Local Airflow Environment

### Prerequisites

- Python 3.8+
- Docker and Docker Compose

### Step 1: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Unix/MacOS
# or
venv\Scripts\activate  # Windows
```

### Step 2: Install requirements

```bash
pip install -r airflow/requirements.txt
```

### Step 3: Set up Airflow using Docker Compose

1. Create a `docker-compose.yml` file in the project root:

```yaml
version: '3'
services:
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    ports:
      - "5432:5432"
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  webserver:
    image: apache/airflow:2.5.1
    depends_on:
      - postgres
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__FERNET_KEY=''
      - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./airflow/logs:/opt/airflow/logs
      - ./airflow/config:/opt/airflow/config
      - ./airflow/hooks:/opt/airflow/hooks
      - ./airflow/operators:/opt/airflow/operators
      - ./airflow/sensors:/opt/airflow/sensors
      - ./airflow/utils:/opt/airflow/utils
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./models:/opt/airflow/models
      - ./predictions:/opt/airflow/predictions
    ports:
      - "8080:8080"
    command: webserver
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5

  scheduler:
    image: apache/airflow:2.5.1
    depends_on:
      - postgres
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__FERNET_KEY=''
      - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./airflow/logs:/opt/airflow/logs
      - ./airflow/config:/opt/airflow/config
      - ./airflow/hooks:/opt/airflow/hooks
      - ./airflow/operators:/opt/airflow/operators
      - ./airflow/sensors:/opt/airflow/sensors
      - ./airflow/utils:/opt/airflow/utils
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./models:/opt/airflow/models
      - ./predictions:/opt/airflow/predictions
    command: scheduler

volumes:
  postgres-db-volume:
```

2. Create required directories:

```bash
mkdir -p ./data ./models ./predictions ./airflow/logs
```

### Step 4: Initialize Airflow database and create admin user

```bash
docker-compose up -d postgres
docker-compose run webserver airflow db init
docker-compose run webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### Step 5: Start Airflow services

```bash
docker-compose up -d
```

The Airflow UI will be available at http://localhost:8080. Log in with the username `admin` and password `admin`.

### Step 6: Configure connections and variables

1. Go to Admin > Connections to set up the DuckDB connection:
   - Conn Id: `duckdb_default`
   - Conn Type: `Generic`
   - Description: `DuckDB Connection`
   - Host: leave empty
   - Schema: leave empty
   - Login: leave empty
   - Password: leave empty
   - Port: leave empty
   - Extra: leave empty

2. Go to Admin > Variables to set up the required variables from `airflow/config/variables.json`:
   - Import the variables from the JSON file or manually create each variable

### Step 7: Testing the pipeline

1. Create a DuckDB database at `./data/ncaa_basketball.duckdb`

2. Enable the DAGs in the Airflow UI:
   - Start with the `data_collection_dag` to collect data
   - Then enable the `feature_engineering_dag` to calculate features
   - Then enable the `model_training_dag` to train the model
   - Finally, enable the `prediction_dag` to generate predictions

3. Monitor the progress in the Airflow UI:
   - Check the task status
   - View logs for details on task execution
   - Verify that each DAG completes successfully

## Troubleshooting

- **Missing dependencies**: If you encounter missing Python dependencies, install them in the Airflow containers:
  ```bash
  docker-compose exec webserver pip install <package-name>
  docker-compose exec scheduler pip install <package-name>
  ```

- **DAG not appearing**: If a DAG is not appearing in the Airflow UI, check for syntax errors:
  ```bash
  docker-compose exec webserver python -c "import airflow.dags.<dag_module>"
  ```

- **Task failures**: If tasks fail, check the logs in the Airflow UI or in the `./airflow/logs` directory for detailed error messages.

## Production Deployment

For production deployment, follow these additional steps:

1. Update Airflow variables with production settings:
   ```bash
   # Upload production_variables.json to your Airflow instance
   # Then import them through the Airflow UI or use CLI:
   airflow variables import airflow/config/production_variables.json
   ```

2. Configure proper database connections:
   - Set up a PostgreSQL database for Airflow
   - Configure DuckDB paths for absolute locations on the production server
   - Ensure proper permissions on data directories

3. Set up MLflow:
   - Configure MLflow with a PostgreSQL backend for tracking
   - Set up artifact storage with S3, GCS, or local filesystem
   - Update `mlflow_tracking_uri` variable in Airflow

4. Configure security:
   - Use proper authentication for Airflow (LDAP, OAuth, etc.)
   - Secure API credentials in environment variables or secrets manager
   - Set up appropriate role-based access control

5. Set up monitoring and alerting:
   - Configure Slack and email alerts in Airflow
   - Set up dashboards for monitoring DAG performance
   - Implement proper logging and log rotation

## MLflow Integration

The NCAA Basketball Analytics project uses MLflow for model tracking and registry. Here's how to set it up:

### Local Development

1. Start the MLflow server using Docker Compose:
   ```bash
   docker compose up -d mlflow
   ```

2. Access the MLflow UI at http://localhost:5000

3. Test the connection from Airflow:
   ```python
   # In an Airflow task or Python script
   from src.models.mlflow.tracking import MLflowTracker

   tracker = MLflowTracker(tracking_uri="sqlite:///mlflow.db")
   experiment_id = tracker.create_experiment("my_experiment")
   ```

### Production Setup

1. Set up a dedicated PostgreSQL database for MLflow:
   ```sql
   CREATE DATABASE mlflow;
   CREATE USER mlflow WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
   ```

2. Configure storage for model artifacts (S3 example):
   ```bash
   export MLFLOW_S3_ENDPOINT_URL=https://your-s3-endpoint
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   ```

3. Start the MLflow tracking server:
   ```bash
   mlflow server \
     --host 0.0.0.0 \
     --port 5000 \
     --backend-store-uri postgresql://mlflow:secure_password@mlflow-db/mlflow \
     --default-artifact-root s3://your-bucket/mlflow-artifacts
   ```

4. Update Airflow variables:
   ```json
   {
     "mlflow_tracking_uri": "postgresql://mlflow:secure_password@mlflow-db/mlflow"
   }
   ```

## End-to-End Testing

To verify the complete workflow integration:

1. Set up the local testing environment:
   ```bash
   ./setup_airflow_test_env.sh
   ```

2. Configure connections in Airflow UI:
   - DuckDB connection
   - MLflow connection

3. Run the integration tests:
   ```bash
   python -m pytest tests/airflow/test_pipeline_integration.py -v
   ```

4. Test the full pipeline execution:
   - Enable and trigger the data_collection_dag
   - Verify data is loaded in DuckDB
   - Enable and trigger the feature_engineering_dag
   - Verify features are calculated
   - Enable and trigger the model_training_dag
   - Verify model is trained and registered in MLflow
   - Enable and trigger the prediction_dag
   - Verify predictions are generated

This completes the test verification process for the Airflow orchestration system.
