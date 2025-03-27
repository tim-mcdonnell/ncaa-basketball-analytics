"""Example usage of the configuration system."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ApiConfig,
    DbConfig,
    FeatureConfig,
    ModelConfig,
    DashboardConfig,
    get_environment,
    load_environment_config,
)


def main():
    """Demonstrate how to use the configuration system."""
    # Set the environment (normally this would be done in your shell)
    # os.environ["ENV"] = "development"

    # Get the current environment
    env = get_environment()
    print(f"Current environment: {env}")

    # Load configuration for the API component
    config_dir = Path("config/examples")
    api_config = load_environment_config(ApiConfig, config_dir)

    # Access configuration values using attribute notation
    print("\nAPI Configuration:")
    print(f"  Host: {api_config.host}")
    print(f"  Port: {api_config.port}")
    print(f"  Debug: {api_config.debug}")
    print(f"  Timeout: {api_config.timeout}")

    # Load configuration for the database component
    db_config = load_environment_config(DbConfig, config_dir)

    print("\nDatabase Configuration:")
    print(f"  Path: {db_config.path}")
    print(f"  Read-only: {db_config.read_only}")
    print(f"  Threads: {db_config.threads}")
    print(f"  Extensions: {db_config.extensions}")

    # Load configuration for the feature component
    feature_config = load_environment_config(FeatureConfig, config_dir)

    print("\nFeature Configuration:")
    print(f"  Enabled: {feature_config.enabled}")
    print(f"  Cache Results: {feature_config.cache_results}")

    # Access nested parameters
    if feature_config.parameters and "window_size" in feature_config.parameters:
        window_size = feature_config.parameters["window_size"]
        print(f"  Window Size: {window_size.value}")

    # Load configuration for the model component
    model_config = load_environment_config(ModelConfig, config_dir)

    print("\nModel Configuration:")
    print(f"  Type: {model_config.type}")
    print(f"  Feature Set: {model_config.feature_set}")
    print(f"  Target: {model_config.target}")
    print(f"  Learning Rate: {model_config.hyperparameters.learning_rate}")

    # Load configuration for the dashboard component
    dashboard_config = load_environment_config(DashboardConfig, config_dir)

    print("\nDashboard Configuration:")
    print(f"  Title: {dashboard_config.title}")
    print(f"  Refresh Interval: {dashboard_config.refresh_interval}")
    print(f"  Primary Color: {dashboard_config.theme.primary_color}")

    # Demonstrate dot notation for convenient access
    dot_config = dashboard_config.dot_dict()
    print(f"  Default View (dot notation): {dot_config.default_view}")
    print(f"  Charts Per Row (dot notation): {dot_config.layout.charts_per_row}")


if __name__ == "__main__":
    main()
