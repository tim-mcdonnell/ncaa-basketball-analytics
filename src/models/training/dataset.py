import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional


class GameDataset(Dataset):
    """
    Dataset class for basketball game data.

    This class converts game data from Polars DataFrame to PyTorch tensors
    for model training.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        feature_columns: List[str],
        target_column: str,
        calculate_score_margin: bool = True,
    ):
        """
        Initialize the dataset with game data.

        Args:
            data: Polars DataFrame containing game data
            feature_columns: List of column names to use as features
            target_column: Column name for the target variable
            calculate_score_margin: Whether to calculate home_score_margin if target_column is not in data
        """
        self.data = data.clone()  # Clone to avoid modifying original
        self.feature_columns = feature_columns
        self.target_column = target_column

        # If target column doesn't exist and is home_score_margin, calculate it
        if (
            target_column == "home_score_margin"
            and target_column not in self.data.columns
            and calculate_score_margin
        ):
            if "home_score" in self.data.columns and "away_score" in self.data.columns:
                self.data = self.data.with_columns(
                    (pl.col("home_score") - pl.col("away_score")).alias("home_score_margin")
                )
            else:
                raise ValueError(
                    "Cannot calculate home_score_margin: home_score or away_score missing"
                )

        # Verify target column exists
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Convert features and target to numpy arrays for faster indexing
        self.features = self.data.select(feature_columns).to_numpy()
        self.targets = self.data.select(target_column).to_numpy().flatten()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target) as PyTorch tensors
        """
        # Convert numpy arrays to PyTorch tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return features, target


def create_train_val_test_split(
    data: pl.DataFrame,
    split_type: str = "random",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: Optional[int] = None,
    date_column: Optional[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into training, validation, and test sets.

    Args:
        data: Data to split
        split_type: Type of split ('random' or 'time')
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        date_column: Column name containing dates (for time-based split)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get number of samples for each split
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    if split_type.lower() == "random":
        # Random split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        # Create splits
        train_data = data.filter(pl.Series(np.isin(range(n_samples), train_indices)))
        val_data = data.filter(pl.Series(np.isin(range(n_samples), val_indices)))
        test_data = data.filter(pl.Series(np.isin(range(n_samples), test_indices)))

    elif split_type.lower() == "time":
        # Time-based split
        if date_column is None or date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found for time-based split")

        # Sort by date
        sorted_data = data.sort(date_column)

        # Get indices for each split
        train_end = n_train
        val_end = n_train + n_val

        # Split sequentially
        train_data = sorted_data.slice(0, train_end)
        val_data = sorted_data.slice(train_end, n_val)
        test_data = sorted_data.slice(val_end, n_samples - val_end)

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    return train_data, val_data, test_data


def create_data_loaders(
    train_data: pl.DataFrame,
    val_data: pl.DataFrame,
    test_data: Optional[pl.DataFrame] = None,
    feature_columns: List[str] = None,
    target_column: str = "home_score_margin",
    batch_size: int = 32,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and test datasets.

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data (optional)
        feature_columns: List of feature column names
        target_column: Target column name
        batch_size: Batch size for DataLoaders
        shuffle_train: Whether to shuffle training data
        shuffle_val: Whether to shuffle validation data
        shuffle_test: Whether to shuffle test data
        num_workers: Number of workers for DataLoader

    Returns:
        Dictionary containing DataLoaders for train, val, and test (if provided)
    """
    # Validate feature columns
    if feature_columns is None:
        raise ValueError("Feature columns must be provided")

    # Create datasets
    train_dataset = GameDataset(train_data, feature_columns, target_column)
    val_dataset = GameDataset(val_data, feature_columns, target_column)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers
    )

    # Create dict to return
    loaders = {"train": train_loader, "val": val_loader}

    # Add test loader if test data provided
    if test_data is not None:
        test_dataset = GameDataset(test_data, feature_columns, target_column)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers
        )
        loaders["test"] = test_loader

    return loaders
