# Getting Started

This guide will help you set up your development environment for the NCAA Basketball Analytics project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.12 or later
- Git
- Docker (for running Airflow locally)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/ncaa-basketball-analytics.git
cd ncaa-basketball-analytics
```

2. Set up a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
uv sync
```

## Configuration

1. Create environment configuration by copying the example:

```bash
cp .env.example .env
```

2. Edit `.env` with your settings (API keys, database paths, etc.)

## Running Tests

The project uses pytest for testing. To run the test suite:

```bash
python -m pytest
```

## Development Workflow

### Test-Driven Development

The project follows Test-Driven Development (TDD) practices:

1. Write a failing test that defines the desired behavior
2. Implement just enough code to make the test pass
3. Refactor the code for clarity and performance while keeping tests passing

### Project Structure

Familiarize yourself with the [project structure](../architecture/project-structure.md) to understand where different components are located.

### Running Components

#### Running Airflow Locally

To start Airflow for local development:

```bash
docker-compose -f airflow/docker-compose.yml up -d
```

Access the Airflow UI at http://localhost:8080

#### Running the Dashboard

To start the Plotly Dash dashboard:

```bash
python app.py
```

Access the dashboard at http://localhost:8050

## Next Steps

- Learn about [adding features](adding-features.md)
- Explore [training models](training-models.md)
- Review the [architecture documentation](../architecture/index.md)
