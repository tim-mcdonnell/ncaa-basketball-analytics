"""
End-to-end integration tests for the NCAA Basketball Analytics Airflow pipeline.

This module contains tests that verify the complete pipeline execution,
including data collection, feature engineering, model training, and prediction.
"""

import os
import pytest
import shutil
import tempfile

from airflow.models import DagBag
from airflow.models import Variable


class TestPipelineIntegration:
    """Test class for pipeline integration tests."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Create temp directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_ncaa_basketball.duckdb")
        cls.data_dir = os.path.join(cls.temp_dir, "data")
        cls.models_dir = os.path.join(cls.temp_dir, "models")
        cls.predictions_dir = os.path.join(cls.temp_dir, "predictions")

        # Create directories
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)

        # Create empty database file
        open(cls.db_path, "a").close()

        # Set Airflow variables
        Variable.set("ncaa_basketball_db_path", cls.db_path)
        Variable.set("ncaa_basketball_data_dir", cls.data_dir)
        Variable.set("ncaa_basketball_models_dir", cls.models_dir)
        Variable.set("ncaa_basketball_predictions_dir", cls.predictions_dir)
        Variable.set("ncaa_basketball_current_season", "2023-24")
        Variable.set("data_collection_lookback_days", "7")
        Variable.set("feature_engineering_lookback_days", "30")
        Variable.set("model_training_lookback_days", "365")
        Variable.set("prediction_forecast_days", "7")
        Variable.set("mlflow_tracking_uri", "sqlite:///mlflow.db")

    @classmethod
    def teardown_class(cls):
        """Clean up after tests."""
        # Remove temp directory and contents
        shutil.rmtree(cls.temp_dir)

    def test_dag_loading(self):
        """Test that all DAGs load without errors."""
        dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
        assert not dag_bag.import_errors
        assert len(dag_bag.dags) == 4

        # Check that all required DAGs are present
        assert "data_collection_dag" in dag_bag.dags
        assert "feature_engineering_dag" in dag_bag.dags
        assert "model_training_dag" in dag_bag.dags
        assert "prediction_dag" in dag_bag.dags

    def test_dag_dependencies(self):
        """Test the task dependencies for all DAGs."""
        dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)

        # Data collection DAG
        data_dag = dag_bag.dags["data_collection_dag"]
        # Get task instances for checking task dependencies
        teams_available = data_dag.get_task("teams_available")
        fetch_games = data_dag.get_task("fetch_games")
        fetch_players = data_dag.get_task("fetch_players")

        # Check upstream/downstream relationships
        assert teams_available.upstream_task_ids == {"fetch_teams"}
        assert fetch_games.upstream_task_ids == {"teams_available"}
        assert fetch_players.upstream_task_ids == {"teams_available"}

        # Feature engineering DAG
        feature_dag = dag_bag.dags["feature_engineering_dag"]
        # Get task instances for feature engineering
        calc_game_features = feature_dag.get_task("calculate_game_features")

        # Check dependencies
        assert calc_game_features.upstream_task_ids.issuperset(
            {"calculate_team_features", "calculate_player_features"}
        )

        # Model training DAG
        training_dag = dag_bag.dags["model_training_dag"]
        # Get task instances for model training
        train_model = training_dag.get_task("train_model")
        evaluate_model = training_dag.get_task("evaluate_model")
        register_model = training_dag.get_task("register_model")

        # Check dependencies
        assert train_model.upstream_task_ids == {"prepare_training_data"}
        assert evaluate_model.upstream_task_ids == {"train_model"}
        assert register_model.upstream_task_ids == {"evaluate_model"}

        # Prediction DAG
        prediction_dag = dag_bag.dags["prediction_dag"]
        # Get task instances for prediction
        prepare_pred_data = prediction_dag.get_task("prepare_prediction_data")
        generate_predictions = prediction_dag.get_task("generate_predictions")
        store_predictions = prediction_dag.get_task("store_predictions")

        # Check dependencies
        assert prepare_pred_data.upstream_task_ids == {"fetch_upcoming_games"}
        assert generate_predictions.upstream_task_ids == {"prepare_prediction_data"}
        assert store_predictions.upstream_task_ids == {"generate_predictions"}

    @pytest.mark.integration
    def test_espn_client_adapter_integration(self):
        """Test integration with the ESPN API client adapter."""
        from src.data.api.espn_client.adapter import ESPNApiClient

        # Initialize client
        client = ESPNApiClient()

        # Test basic functionality
        teams = client.get_teams()
        assert isinstance(teams, list)

        # Test getting games (if available)
        games = client.get_games(limit=5)
        assert isinstance(games, list)

    @pytest.mark.integration
    def test_mlflow_integration(self):
        """Test integration with MLflow tracking and registry."""
        from src.models.mlflow.tracking import MLflowTracker
        from src.models.mlflow.registry import MLflowModelRegistry

        # Initialize tracking and registry
        tracker = MLflowTracker(tracking_uri="sqlite:///mlflow.db")
        registry = MLflowModelRegistry(tracking_uri="sqlite:///mlflow.db")

        # Test basic tracking functionality
        experiment_id = tracker.create_experiment("test_experiment")
        assert experiment_id is not None

        # Start a run
        run_id = tracker.start_run("test_run")
        assert run_id is not None

        # Log some metrics
        tracker.log_metric("accuracy", 0.85)
        tracker.log_param("model_type", "test_model")

        # End the run
        tracker.end_run()

        # Test basic registry functionality
        model_name = "test_model"
        model_version = registry.register_model(
            model_name=model_name,
            model_uri=f"runs:/{run_id}/model",
            description="Test model registration",
        )
        assert model_version is not None
