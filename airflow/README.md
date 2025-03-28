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

1. Update `airflow/config/variables.json` with production paths and settings
2. Set up a proper database backend (PostgreSQL recommended)
3. Configure proper authentication and security
4. Use a production-ready Airflow deployment method (Kubernetes, Celery, etc.)
5. Set up monitoring and alerting
