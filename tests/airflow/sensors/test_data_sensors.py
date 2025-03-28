from unittest.mock import patch

from airflow.sensors.data_sensors import DuckDBTableSensor, NewDataSensor


class TestDataSensors:
    """Tests for data availability sensors."""

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_duckdb_table_sensor_table_exists(self, mock_duckdb_hook):
        """Test that the DuckDBTableSensor correctly detects when a table exists."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock table existence check
        mock_hook_instance.get_records.return_value = [
            ("1",)
        ]  # Non-empty result means table exists

        # Act
        sensor = DuckDBTableSensor(
            task_id="test_table_sensor",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="teams",
        )
        result = sensor.poke(context={})

        # Assert
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        mock_hook_instance.get_records.assert_called_once()
        assert result is True, "Sensor should return True when table exists"

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_duckdb_table_sensor_table_does_not_exist(self, mock_duckdb_hook):
        """Test that the DuckDBTableSensor correctly detects when a table doesn't exist."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock table existence check - empty result means table doesn't exist
        mock_hook_instance.get_records.return_value = []

        # Act
        sensor = DuckDBTableSensor(
            task_id="test_table_sensor",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="teams",
        )
        result = sensor.poke(context={})

        # Assert
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        mock_hook_instance.get_records.assert_called_once()
        assert result is False, "Sensor should return False when table doesn't exist"

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_duckdb_table_sensor_with_row_count(self, mock_duckdb_hook):
        """Test that the DuckDBTableSensor correctly checks for minimum row count."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock row count check
        mock_hook_instance.get_records.return_value = [("50",)]  # 50 rows in the table

        # Act & Assert - Should return True when row count meets minimum
        sensor = DuckDBTableSensor(
            task_id="test_table_sensor_with_rows",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="teams",
            min_rows=50,
        )
        result = sensor.poke(context={})
        assert result is True, "Sensor should return True when row count meets minimum"

        # Act & Assert - Should return False when row count below minimum
        sensor = DuckDBTableSensor(
            task_id="test_table_sensor_with_rows",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="teams",
            min_rows=100,
        )
        result = sensor.poke(context={})
        assert result is False, "Sensor should return False when row count below minimum"

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_new_data_sensor_with_new_data(self, mock_duckdb_hook):
        """Test that the NewDataSensor correctly detects when new data is available."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock check for new data since last execution
        mock_hook_instance.get_records.return_value = [("10",)]  # 10 new records

        # Act
        sensor = NewDataSensor(
            task_id="test_new_data_sensor",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="games",
            date_column="date",
            execution_date="{{ execution_date }}",
        )
        # Create mock context with execution_date
        mock_context = {"execution_date": "2023-01-01"}
        result = sensor.poke(context=mock_context)

        # Assert
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        mock_hook_instance.get_records.assert_called_once()
        assert result is True, "Sensor should return True when new data is available"

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_new_data_sensor_without_new_data(self, mock_duckdb_hook):
        """Test that the NewDataSensor correctly detects when no new data is available."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock check for new data since last execution
        mock_hook_instance.get_records.return_value = [("0",)]  # 0 new records

        # Act
        sensor = NewDataSensor(
            task_id="test_new_data_sensor",
            conn_id="duckdb_default",
            database="test.duckdb",
            table="games",
            date_column="date",
            execution_date="{{ execution_date }}",
        )
        # Create mock context with execution_date
        mock_context = {"execution_date": "2023-01-01"}
        result = sensor.poke(context=mock_context)

        # Assert
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        mock_hook_instance.get_records.assert_called_once()
        assert result is False, "Sensor should return False when no new data is available"

    @patch("airflow.sensors.data_sensors.DuckDBHook")
    def test_new_data_sensor_with_custom_sql(self, mock_duckdb_hook):
        """Test that the NewDataSensor correctly uses custom SQL when provided."""
        # Arrange
        mock_hook_instance = mock_duckdb_hook.return_value
        mock_hook_instance.get_records.return_value = [("5",)]  # 5 records matching custom SQL

        # Act
        sensor = NewDataSensor(
            task_id="test_new_data_sensor_custom_sql",
            conn_id="duckdb_default",
            database="test.duckdb",
            sql="SELECT COUNT(*) FROM games WHERE date > '{{ execution_date }}' AND home_score > away_score",
            execution_date="{{ execution_date }}",
        )
        # Create mock context with execution_date
        mock_context = {"execution_date": "2023-01-01"}
        result = sensor.poke(context=mock_context)

        # Assert
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        mock_hook_instance.get_records.assert_called_once()
        assert result is True, "Sensor should return True when custom SQL finds matching records"
