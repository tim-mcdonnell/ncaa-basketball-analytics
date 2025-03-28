from airflow.models import DagBag
import os


class TestDagStructure:
    """Tests for DAG structure validation."""

    def setup_method(self):
        """Setup for the test - load the DAG bag."""
        self.dagbag = DagBag(
            dag_folder=os.path.join(os.path.dirname(__file__), "../../../airflow/dags"),
            include_examples=False,
        )

    def test_dag_loading_no_errors(self):
        """Test that all DAGs load successfully without import errors."""
        # Assert
        assert self.dagbag.import_errors == {}, f"DAG import errors: {self.dagbag.import_errors}"
        assert len(self.dagbag.dags) > 0, "No DAGs were loaded"

    def test_data_collection_dag_exists(self):
        """Test that the data collection DAG exists and has the correct structure."""
        # Assert
        assert "data_collection_dag" in self.dagbag.dags, "Data collection DAG not found"
        dag = self.dagbag.dags["data_collection_dag"]

        # Verify required tasks exist
        expected_tasks = ["fetch_teams", "fetch_games", "fetch_players", "fetch_player_stats"]
        for task_id in expected_tasks:
            assert task_id in dag.task_dict, f"Task {task_id} missing from data collection DAG"

        # Verify dependencies
        # All tasks should depend on fetch_teams
        for task_id in ["fetch_games", "fetch_players"]:
            assert dag.has_task_dependency(
                "fetch_teams", task_id
            ), f"Missing dependency from fetch_teams to {task_id}"

        # fetch_player_stats should depend on fetch_players
        assert dag.has_task_dependency(
            "fetch_players", "fetch_player_stats"
        ), "Missing dependency from fetch_players to fetch_player_stats"

    def test_feature_engineering_dag_exists(self):
        """Test that the feature engineering DAG exists and has the correct structure."""
        # Assert
        assert "feature_engineering_dag" in self.dagbag.dags, "Feature engineering DAG not found"
        dag = self.dagbag.dags["feature_engineering_dag"]

        # Verify required tasks exist
        expected_tasks = [
            "calculate_team_features",
            "calculate_player_features",
            "calculate_game_features",
        ]
        for task_id in expected_tasks:
            assert task_id in dag.task_dict, f"Task {task_id} missing from feature engineering DAG"

        # Verify dependencies - features can be calculated in parallel
        assert len(dag.task_dict) >= len(expected_tasks), "Feature engineering DAG missing tasks"

    def test_model_training_dag_exists(self):
        """Test that the model training DAG exists and has the correct structure."""
        # Assert
        assert "model_training_dag" in self.dagbag.dags, "Model training DAG not found"
        dag = self.dagbag.dags["model_training_dag"]

        # Verify required tasks exist
        expected_tasks = [
            "prepare_training_data",
            "train_model",
            "evaluate_model",
            "register_model",
        ]
        for task_id in expected_tasks:
            assert task_id in dag.task_dict, f"Task {task_id} missing from model training DAG"

        # Verify dependencies
        for i in range(1, len(expected_tasks)):
            # Each task should depend on the previous one in a linear flow
            assert dag.has_task_dependency(
                expected_tasks[i - 1], expected_tasks[i]
            ), f"Missing dependency from {expected_tasks[i-1]} to {expected_tasks[i]}"

    def test_prediction_dag_exists(self):
        """Test that the prediction DAG exists and has the correct structure."""
        # Assert
        assert "prediction_dag" in self.dagbag.dags, "Prediction DAG not found"
        dag = self.dagbag.dags["prediction_dag"]

        # Verify required tasks exist
        expected_tasks = [
            "fetch_upcoming_games",
            "prepare_prediction_data",
            "generate_predictions",
            "store_predictions",
        ]
        for task_id in expected_tasks:
            assert task_id in dag.task_dict, f"Task {task_id} missing from prediction DAG"

        # Verify dependencies - should be a linear flow
        for i in range(1, len(expected_tasks)):
            # Each task should depend on the previous one
            assert dag.has_task_dependency(
                expected_tasks[i - 1], expected_tasks[i]
            ), f"Missing dependency from {expected_tasks[i-1]} to {expected_tasks[i]}"

    def test_dags_have_default_args(self):
        """Test that all DAGs have default_args defined."""
        for dag_id, dag in self.dagbag.dags.items():
            assert dag.default_args, f"DAG {dag_id} does not have default_args"
            assert (
                "owner" in dag.default_args
            ), f"DAG {dag_id} does not have 'owner' in default_args"
            assert (
                "retries" in dag.default_args
            ), f"DAG {dag_id} does not have 'retries' in default_args"
            assert (
                "retry_delay" in dag.default_args
            ), f"DAG {dag_id} does not have 'retry_delay' in default_args"

    def test_dags_have_description(self):
        """Test that all DAGs have a description."""
        for dag_id, dag in self.dagbag.dags.items():
            assert dag.description, f"DAG {dag_id} does not have a description"

    def test_dags_have_tags(self):
        """Test that all DAGs have tags."""
        for dag_id, dag in self.dagbag.dags.items():
            assert dag.tags, f"DAG {dag_id} does not have tags"
            assert len(dag.tags) > 0, f"DAG {dag_id} has empty tags"

    def test_task_has_retry_policy(self):
        """Test that critical tasks have retry policy configured."""
        critical_tasks = [
            "fetch_teams",
            "fetch_games",
            "fetch_players",
            "fetch_player_stats",
            "calculate_team_features",
            "train_model",
            "generate_predictions",
        ]

        for dag_id, dag in self.dagbag.dags.items():
            for task_id, task in dag.task_dict.items():
                if task_id in critical_tasks:
                    assert (
                        task.retries > 0
                    ), f"Task {task_id} in DAG {dag_id} has no retries configured"
