#!/bin/bash

# Create required directories
mkdir -p ./data ./models ./predictions ./airflow/logs

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available (newer Docker versions use 'docker compose')
if ! docker compose version &> /dev/null; then
    echo "Docker Compose functionality is not available. Please ensure you have Docker Compose installed."
    exit 1
fi

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
docker compose up -d postgres

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Initialize Airflow database
echo "Initializing Airflow database..."
docker compose run --rm webserver airflow db init

# Create Airflow admin user
echo "Creating Airflow admin user..."
docker compose run --rm webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start Airflow services
echo "Starting Airflow services..."
docker compose up -d

# Create DuckDB database file
echo "Creating DuckDB database file..."
touch ./data/ncaa_basketball.duckdb

echo "Setup complete! Airflow UI is available at http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "Next steps:"
echo "1. Go to Admin > Variables and import variables from airflow/config/variables.json"
echo "2. Go to Admin > Connections and create a new connection:"
echo "   - Conn Id: duckdb_default"
echo "   - Conn Type: Generic"
echo "   - Description: DuckDB Connection"
echo "3. Enable the data_collection_dag in the Airflow UI"
echo "4. Monitor the DAG execution and logs"
