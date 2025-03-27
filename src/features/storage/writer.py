"""
Feature storage writer module.

This module provides the FeatureWriter class for persisting feature values
to a DuckDB database.
"""

import json
from typing import Optional
import duckdb
import polars as pl

from src.features.registry import Feature


class FeatureWriter:
    """
    Writer for storing feature values in a DuckDB database.

    This class provides methods to store feature values and metadata
    in a structured format for later retrieval.

    Attributes:
        db_path: Path to the DuckDB database file
    """

    def __init__(self, db_path: str):
        """
        Initialize a feature writer.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """
        Initialize the database schema if it doesn't exist.

        Creates the necessary tables for storing feature values and metadata.
        """
        conn = duckdb.connect(self.db_path)

        # Create feature values table
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS feature_values_id_seq;
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                id INTEGER PRIMARY KEY DEFAULT nextval('feature_values_id_seq'),
                feature_name VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                entity_type VARCHAR NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                value VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL
            )
        """)

        # Create feature metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL,
                metadata JSON NOT NULL,
                PRIMARY KEY (feature_name, feature_version)
            )
        """)

        # Create index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_values_lookup
            ON feature_values (feature_name, feature_version, entity_type, entity_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp
            ON feature_values (feature_name, feature_version, entity_id, timestamp)
        """)

        conn.close()

    def store(
        self,
        feature: Feature,
        data: pl.DataFrame,
        entity_type: str,
        entity_id_column: str,
        value_column: str,
        timestamp_column: Optional[str] = None,
        update_existing: bool = False,
    ) -> None:
        """
        Store feature values in the database.

        Args:
            feature: The feature to store values for
            data: DataFrame containing feature values
            entity_type: Type of entities (e.g., "team", "player")
            entity_id_column: Name of the column containing entity IDs
            value_column: Name of the column containing feature values
            timestamp_column: Optional name of the column containing timestamps
            update_existing: Whether to update existing values or append
        """
        # Make sure required columns exist
        if entity_id_column not in data.columns:
            raise ValueError(f"Entity ID column '{entity_id_column}' not in DataFrame")

        if value_column not in data.columns:
            raise ValueError(f"Value column '{value_column}' not in DataFrame")

        if timestamp_column and timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not in DataFrame")

        # Convert to records for insertion
        records = []
        for row in data.rows(named=True):
            entity_id = row[entity_id_column]
            value = row[value_column]

            record = {
                "feature_name": feature.name,
                "feature_version": feature.version,
                "entity_id": str(entity_id),
                "entity_type": entity_type,
                "value": json.dumps(value) if not isinstance(value, str) else value,
            }

            # Add timestamp if provided
            if timestamp_column:
                record["timestamp"] = row[timestamp_column]

            records.append(record)

        # Insert or update records
        conn = duckdb.connect(self.db_path)

        if update_existing:
            # First delete existing values for this feature/entity combination
            conn.execute(
                """
                DELETE FROM feature_values
                WHERE feature_name = ? AND feature_version = ? AND entity_type = ?
                AND entity_id IN (SELECT DISTINCT entity_id FROM feature_values
                                 WHERE feature_name = ? AND feature_version = ? AND entity_type = ?)
            """,
                (
                    feature.name,
                    feature.version,
                    entity_type,
                    feature.name,
                    feature.version,
                    entity_type,
                ),
            )

        # Insert new values
        for record in records:
            if timestamp_column:
                conn.execute(
                    """
                    INSERT INTO feature_values (feature_name, feature_version, entity_id, entity_type, timestamp, value)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        record["feature_name"],
                        record["feature_version"],
                        record["entity_id"],
                        record["entity_type"],
                        record["timestamp"],
                        record["value"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO feature_values (feature_name, feature_version, entity_id, entity_type, value)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        record["feature_name"],
                        record["feature_version"],
                        record["entity_id"],
                        record["entity_type"],
                        record["value"],
                    ),
                )

        conn.close()

    def store_metadata(self, feature: Feature) -> None:
        """
        Store feature metadata in the database.

        Args:
            feature: The feature whose metadata should be stored
        """
        # Convert metadata to JSON
        metadata_dict = {
            "description": feature.metadata.description,
            "category": feature.metadata.category,
            "tags": feature.metadata.tags,
            "dependencies": feature.dependencies,
        }

        metadata_json = json.dumps(metadata_dict)

        # Store in database
        conn = duckdb.connect(self.db_path)

        # Use upsert to handle existing metadata
        conn.execute(
            """
            DELETE FROM feature_metadata
            WHERE feature_name = ? AND feature_version = ?
        """,
            (feature.name, feature.version),
        )

        conn.execute(
            """
            INSERT INTO feature_metadata (feature_name, feature_version, metadata)
            VALUES (?, ?, ?)
        """,
            (feature.name, feature.version, metadata_json),
        )

        conn.close()
