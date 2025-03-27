"""
Feature storage reader module.

This module provides the FeatureReader class for retrieving feature values
from a DuckDB database.
"""

import json
from typing import Dict, Any, Optional
import duckdb
import polars as pl


class FeatureReader:
    """
    Reader for retrieving feature values from a DuckDB database.

    This class provides methods to retrieve feature values and metadata
    in a structured format.

    Attributes:
        db_path: Path to the DuckDB database file
    """

    def __init__(self, db_path: str):
        """
        Initialize a feature reader.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path

    def get_values(
        self, feature_name: str, feature_version: str, entity_type: str, as_of: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Get feature values for a specific feature.

        Args:
            feature_name: Name of the feature
            feature_version: Version of the feature
            entity_type: Type of entities to retrieve values for
            as_of: Optional timestamp to retrieve values as of a certain time

        Returns:
            DataFrame containing entity IDs and feature values
        """
        conn = duckdb.connect(self.db_path)

        if as_of:
            # Get values as of a specific time
            query = """
                WITH latest_values AS (
                    SELECT
                        entity_id,
                        value,
                        MAX(timestamp) as max_timestamp
                    FROM feature_values
                    WHERE feature_name = ?
                    AND feature_version = ?
                    AND entity_type = ?
                    AND timestamp <= ?
                    GROUP BY entity_id, value
                )
                SELECT entity_id, value
                FROM latest_values
            """
            result = conn.execute(query, (feature_name, feature_version, entity_type, as_of)).pl()
        else:
            # Get latest values
            query = """
                WITH latest_timestamps AS (
                    SELECT
                        entity_id,
                        MAX(timestamp) as max_timestamp
                    FROM feature_values
                    WHERE feature_name = ?
                    AND feature_version = ?
                    AND entity_type = ?
                    GROUP BY entity_id
                )
                SELECT fv.entity_id, fv.value
                FROM feature_values fv
                JOIN latest_timestamps lt
                ON fv.entity_id = lt.entity_id AND fv.timestamp = lt.max_timestamp
                WHERE fv.feature_name = ?
                AND fv.feature_version = ?
                AND fv.entity_type = ?
            """
            result = conn.execute(
                query,
                (
                    feature_name,
                    feature_version,
                    entity_type,
                    feature_name,
                    feature_version,
                    entity_type,
                ),
            ).pl()

        conn.close()

        # Convert values from JSON string or numeric string if needed
        if not result.is_empty():
            result = result.with_columns(
                pl.col("value").map_elements(self._convert_value, return_dtype=pl.Object)
            )

        return result

    def _convert_value(self, value_str: str) -> Any:
        """Convert a string value to the appropriate type."""
        if not isinstance(value_str, str):
            return value_str

        # Try to convert JSON
        if value_str.startswith("[") or value_str.startswith("{"):
            return json.loads(value_str)

        # Try to convert to numeric
        try:
            # Check if it's an integer
            if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
                return int(value_str)
            # Check if it's a float (contains a decimal point)
            elif "." in value_str:
                return float(value_str)
        except (ValueError, TypeError):
            pass

        # Return as is if no conversion applies
        return value_str

    def get_value(
        self,
        feature_name: str,
        feature_version: str,
        entity_type: str,
        entity_id: str,
        as_of: Optional[str] = None,
    ) -> Any:
        """
        Get a single feature value for a specific entity.

        Args:
            feature_name: Name of the feature
            feature_version: Version of the feature
            entity_type: Type of entity
            entity_id: ID of the entity
            as_of: Optional timestamp to retrieve value as of a certain time

        Returns:
            Feature value for the entity
        """
        conn = duckdb.connect(self.db_path)

        if as_of:
            # Get value as of a specific time
            query = """
                SELECT value
                FROM feature_values
                WHERE feature_name = ?
                AND feature_version = ?
                AND entity_type = ?
                AND entity_id = ?
                AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(
                query, (feature_name, feature_version, entity_type, entity_id, as_of)
            ).fetchone()
        else:
            # Get latest value
            query = """
                SELECT value
                FROM feature_values
                WHERE feature_name = ?
                AND feature_version = ?
                AND entity_type = ?
                AND entity_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(
                query, (feature_name, feature_version, entity_type, entity_id)
            ).fetchone()

        conn.close()

        if result is None:
            return None

        value = result[0]

        # Convert to appropriate type
        return self._convert_value(value)

    def get_time_series(
        self,
        feature_name: str,
        feature_version: str,
        entity_type: str,
        entity_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get a time series of feature values for a specific entity.

        Args:
            feature_name: Name of the feature
            feature_version: Version of the feature
            entity_type: Type of entity
            entity_id: ID of the entity
            start_time: Optional start time for the time series
            end_time: Optional end time for the time series

        Returns:
            DataFrame containing timestamps and feature values
        """
        conn = duckdb.connect(self.db_path)

        # Build query based on time range
        query = """
            SELECT timestamp, value
            FROM feature_values
            WHERE feature_name = ?
            AND feature_version = ?
            AND entity_type = ?
            AND entity_id = ?
        """

        params = [feature_name, feature_version, entity_type, entity_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        result = conn.execute(query, params).pl()
        conn.close()

        # Convert values to appropriate types
        if not result.is_empty():
            result = result.with_columns(
                pl.col("value").map_elements(self._convert_value, return_dtype=pl.Object)
            )

        return result

    def get_feature_metadata(
        self, feature_name: str, feature_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific feature.

        Args:
            feature_name: Name of the feature
            feature_version: Version of the feature

        Returns:
            Dictionary containing feature metadata
        """
        conn = duckdb.connect(self.db_path)

        query = """
            SELECT metadata
            FROM feature_metadata
            WHERE feature_name = ?
            AND feature_version = ?
        """

        result = conn.execute(query, (feature_name, feature_version)).fetchone()
        conn.close()

        if result is None:
            return None

        return json.loads(result[0])
