"""Default configuration settings."""

from typing import Any, Dict


def get_default_settings() -> Dict[str, Any]:
    """Get the default configuration settings.

    This function returns the default configuration settings for all components.
    These settings are used as fallbacks when no configuration file is provided.

    Returns:
        Default configuration dictionary
    """
    return {
        "api": {
            "host": "localhost",
            "port": 8000,
            "debug": False,
            "timeout": 30,
            "rate_limit": 0,
            "endpoints": {"teams": "/api/teams", "games": "/api/games", "stats": "/api/stats"},
        },
        "db": {
            "path": "./data/basketball.duckdb",
            "read_only": False,
            "memory_map": True,
            "threads": 4,
            "extensions": ["json", "httpfs"],
            "allow_external_access": False,
        },
        "feature": {
            "enabled": True,
            "cache_results": True,
            "parameters": {},
            "dependencies": [],
            "priority": 100,
        },
        "model": {
            "type": "xgboost",
            "feature_set": "all_features",
            "target": "win_probability",
            "hyperparameters": {"learning_rate": 0.01, "max_depth": 5, "num_estimators": 100},
            "evaluation": {
                "metrics": ["accuracy", "f1_score"],
                "validation_split": 0.2,
                "cross_validation_folds": 5,
            },
            "experiment_tracking": {"enabled": True, "log_artifacts": True},
            "random_seed": 42,
        },
        "dashboard": {
            "title": "NCAA Basketball Analytics",
            "refresh_interval": 60,
            "theme": {
                "primary_color": "#0066cc",
                "secondary_color": "#f8f9fa",
                "text_color": "#212529",
                "background_color": "#ffffff",
                "font_family": "Arial, sans-serif",
            },
            "layout": {
                "sidebar_width": 250,
                "content_width": "fluid",
                "charts_per_row": 2,
                "default_chart_height": 400,
            },
            "default_view": "team_overview",
            "available_views": ["team_overview", "team_comparison", "player_stats", "predictions"],
            "cache_timeout": 300,
            "show_filters": True,
            "max_items_per_page": 25,
        },
    }
