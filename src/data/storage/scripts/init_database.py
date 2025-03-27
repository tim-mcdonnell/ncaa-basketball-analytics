#!/usr/bin/env python3
"""
Database initialization script.

This script initializes the NCAA Basketball Analytics database with all required tables.
It's meant to be run directly to set up a fresh database.

Usage:
    python -m src.data.storage.scripts.init_database [--db-path PATH]
"""

import argparse
import logging
import sys

from src.data.storage.db import DatabaseManager


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Initialize NCAA Basketball Analytics database")
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the database file (optional, uses default if not provided)",
    )
    return parser.parse_args()


def main():
    """Initialize the database with schema."""
    logger = setup_logging()
    args = parse_args()

    try:
        # Get database path from args or use default
        db_path = args.db_path if args.db_path else None

        logger.info(
            "Initializing database%s", f" at {db_path}" if db_path else " at default location"
        )

        # Initialize database manager
        db_manager = DatabaseManager(db_path=db_path)

        # Initialize schema
        db_manager.initialize_schema()

        logger.info("Database initialization completed successfully")

        # Show the location of the database file
        actual_path = db_manager.db_path
        logger.info(f"Database created at: {actual_path}")

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        sys.exit(1)
    finally:
        # Ensure we close any open connections
        if "db_manager" in locals():
            db_manager.close_connection()


if __name__ == "__main__":
    main()
