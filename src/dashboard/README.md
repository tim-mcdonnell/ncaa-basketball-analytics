# NCAA Basketball Analytics Dashboard

## Overview

This dashboard provides interactive visualizations for NCAA basketball data, including:

- Team analysis with performance trends and statistics
- Game predictions with win probability estimates
- Player statistics and comparisons

## Usage

### Running the Dashboard

To run the dashboard locally:

```bash
# Option 1: Run as a module
python -m src.dashboard

# Option 2: Run the server directly
cd src/dashboard
python __main__.py
```

The dashboard will be available at http://localhost:8050 by default.

### Environment Variables

The dashboard can be configured using the following environment variables:

- `DASHBOARD_HOST`: Host to bind to (default: 0.0.0.0)
- `DASHBOARD_PORT`: Port to bind to (default: 8050)
- `DASHBOARD_DEBUG`: Whether to run in debug mode (default: False)
- `DUCKDB_PATH`: Path to the DuckDB database file (default: data/processed/basketball.db)

## Features

### Team Analysis

- Performance trends over time
- Win/loss records
- Points scored and allowed
- Recent game results

### Game Prediction

- Win probability estimates
- Team comparison by key statistics
- Historical matchup analysis

### Player Statistics

- Performance by metric
- Comparison between players
- Statistical breakdowns

## Development

### Project Structure

- `__main__.py`: Entry point for running the dashboard
- `app.py`: Dashboard application configuration
- `server.py`: Server setup and initialization
- `callbacks/`: Interactive callback functions
- `components/`: Reusable UI components
- `data/`: Data access layer
- `figures/`: Visualization generation
- `layouts/`: Page layouts

### Adding New Features

1. Create appropriate layout in `layouts/`
2. Implement any needed components in `components/`
3. Add data access methods in `data/repository.py`
4. Implement visualization functions in `figures/`
5. Register callbacks in `callbacks/`
6. Update app.py to include the new feature
