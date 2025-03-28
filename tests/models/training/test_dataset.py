import polars as pl
import torch
from torch.utils.data import DataLoader

from src.models.training.dataset import GameDataset, create_train_val_test_split


class TestGameDataset:
    """Test suite for the GameDataset class."""

    def test_dataset_creation(self):
        """Verify training datasets are correctly created."""
        # Create mock data
        data = pl.DataFrame(
            {
                "game_id": ["1", "2", "3", "4", "5"],
                "home_team_id": ["A", "B", "C", "D", "E"],
                "away_team_id": ["F", "G", "H", "I", "J"],
                "home_score": [70, 65, 80, 75, 90],
                "away_score": [65, 70, 75, 80, 85],
                "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
                "feature3": [0.3, 0.3, 0.3, 0.3, 0.3],
            }
        )

        feature_cols = ["feature1", "feature2", "feature3"]
        target_col = "home_score_margin"

        # Test dataset creation
        dataset = GameDataset(data, feature_cols, target_col)

        # Verify dataset length
        assert len(dataset) == 5, "Dataset length does not match input data length"

        # Test getting an item
        features, target = dataset[0]

        # Verify item types
        assert isinstance(features, torch.Tensor), "Features should be a torch Tensor"
        assert isinstance(target, torch.Tensor), "Target should be a torch Tensor"

        # Verify feature dimensions
        assert features.shape == (
            len(feature_cols),
        ), f"Expected shape ({len(feature_cols)},), got {features.shape}"

    def test_dataset_dataloader(self):
        """Test creating a DataLoader from the dataset."""
        # Create mock data
        data = pl.DataFrame(
            {
                "game_id": ["1", "2", "3", "4", "5"],
                "home_team_id": ["A", "B", "C", "D", "E"],
                "away_team_id": ["F", "G", "H", "I", "J"],
                "home_score": [70, 65, 80, 75, 90],
                "away_score": [65, 70, 75, 80, 85],
                "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature2": [0.5, 0.4, 0.3, 0.2, 0.1],
                "feature3": [0.3, 0.3, 0.3, 0.3, 0.3],
            }
        )

        feature_cols = ["feature1", "feature2", "feature3"]
        target_col = "home_score_margin"

        # Create dataset
        dataset = GameDataset(data, feature_cols, target_col)

        # Create DataLoader
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Get first batch
        features_batch, targets_batch = next(iter(dataloader))

        # Verify batch shapes
        assert (
            features_batch.shape == (batch_size, len(feature_cols))
        ), f"Expected features shape ({batch_size}, {len(feature_cols)}), got {features_batch.shape}"
        assert targets_batch.shape == (
            batch_size,
        ), f"Expected targets shape ({batch_size},), got {targets_batch.shape}"


class TestDataSplitting:
    """Test suite for data splitting functionality."""

    def test_train_val_test_split(self):
        """Test creating train/validation/test splits."""
        # Create mock data - using timestamps to test time-based splitting
        data = pl.DataFrame(
            {
                "game_id": [f"{i}" for i in range(1, 101)],
                "game_date": [
                    i for i in range(1, 101)
                ],  # Use integers for dates to simplify testing
                "feature1": [float(i) / 100 for i in range(1, 101)],
                "target": [float(i % 10) for i in range(1, 101)],
            }
        )

        # Test standard split (random)
        train_data, val_data, test_data = create_train_val_test_split(
            data,
            split_type="random",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        # Verify split sizes
        assert len(train_data) == 70, f"Expected 70 train samples, got {len(train_data)}"
        assert len(val_data) == 15, f"Expected 15 validation samples, got {len(val_data)}"
        assert len(test_data) == 15, f"Expected 15 test samples, got {len(test_data)}"

        # Verify no overlap between splits
        train_ids = set(train_data["game_id"].to_list())
        val_ids = set(val_data["game_id"].to_list())
        test_ids = set(test_data["game_id"].to_list())

        assert len(train_ids.intersection(val_ids)) == 0, "Train and validation sets overlap"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and test sets overlap"
        assert len(val_ids.intersection(test_ids)) == 0, "Validation and test sets overlap"

        # Test time-based split
        train_data, val_data, test_data = create_train_val_test_split(
            data,
            split_type="time",
            date_column="game_date",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # For time-based split, check that indices are sequential and non-overlapping
        # Get the last element of each split to verify they are in order
        last_train_idx = train_data["game_date"].max()
        first_val_idx = val_data["game_date"].min()
        last_val_idx = val_data["game_date"].max()
        first_test_idx = test_data["game_date"].min()

        # Verify splits are in order
        assert last_train_idx < first_val_idx, "Train should end before validation starts"
        assert last_val_idx < first_test_idx, "Validation should end before test starts"
