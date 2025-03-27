"""
Tests for the feature registry.

This module tests the functionality of the feature registry, which is responsible
for managing feature definitions, metadata, and dependencies.
"""

from src.features.registry import FeatureRegistry, Feature, FeatureMetadata


class TestFeatureRegistry:
    """Tests for the feature registry functionality."""

    def test_feature_registration(self):
        """Test that features can be registered with metadata."""
        # Arrange
        registry = FeatureRegistry()
        feature = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        # Act
        registry.register(feature)

        # Assert
        assert registry.get("win_percentage") == feature
        assert registry.get("win_percentage").name == "win_percentage"
        assert registry.get("win_percentage").version == "1.0.0"
        assert registry.get("win_percentage").metadata.description == "Team win percentage"

    def test_feature_registration_with_dependencies(self):
        """Test that features can be registered with dependencies."""
        # Arrange
        registry = FeatureRegistry()
        wins_feature = Feature(
            name="wins",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Total team wins",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        games_feature = Feature(
            name="games_played",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Total games played",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        win_pct_feature = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
            dependencies=["wins", "games_played"],
        )

        # Act
        registry.register(wins_feature)
        registry.register(games_feature)
        registry.register(win_pct_feature)

        # Assert
        assert registry.get("win_percentage").dependencies == ["wins", "games_played"]

    def test_get_nonexistent_feature(self):
        """Test that getting a nonexistent feature returns None."""
        # Arrange
        registry = FeatureRegistry()

        # Act
        feature = registry.get("nonexistent_feature")

        # Assert
        assert feature is None

    def test_feature_versioning(self):
        """Test that features can be registered with different versions."""
        # Arrange
        registry = FeatureRegistry()
        feature_v1 = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        feature_v2 = Feature(
            name="win_percentage",
            version="2.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage (improved)",
                category="team_performance",
                tags=["team", "performance", "advanced"],
            ),
        )

        # Act
        registry.register(feature_v1)
        registry.register(feature_v2)

        # Assert
        assert registry.get("win_percentage").version == "2.0.0"  # Latest by default
        assert registry.get("win_percentage", version="1.0.0").version == "1.0.0"
        assert registry.get("win_percentage", version="2.0.0").version == "2.0.0"

    def test_list_features(self):
        """Test that all registered features can be listed."""
        # Arrange
        registry = FeatureRegistry()
        feature1 = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        feature2 = Feature(
            name="points_per_game",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team points per game",
                category="team_performance",
                tags=["team", "performance", "offense"],
            ),
        )

        # Act
        registry.register(feature1)
        registry.register(feature2)
        features = registry.list_features()

        # Assert
        assert len(features) == 2
        assert "win_percentage" in [f.name for f in features]
        assert "points_per_game" in [f.name for f in features]

    def test_list_features_by_category(self):
        """Test that features can be filtered by category."""
        # Arrange
        registry = FeatureRegistry()
        team_feature = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        player_feature = Feature(
            name="player_efficiency",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Player efficiency rating",
                category="player_performance",
                tags=["player", "performance", "advanced"],
            ),
        )

        # Act
        registry.register(team_feature)
        registry.register(player_feature)
        team_features = registry.list_features(category="team_performance")
        player_features = registry.list_features(category="player_performance")

        # Assert
        assert len(team_features) == 1
        assert team_features[0].name == "win_percentage"
        assert len(player_features) == 1
        assert player_features[0].name == "player_efficiency"

    def test_list_features_by_tag(self):
        """Test that features can be filtered by tag."""
        # Arrange
        registry = FeatureRegistry()
        basic_feature = Feature(
            name="win_percentage",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team win percentage",
                category="team_performance",
                tags=["team", "performance", "basic"],
            ),
        )

        advanced_feature = Feature(
            name="offensive_rating",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Team offensive rating",
                category="team_performance",
                tags=["team", "performance", "advanced"],
            ),
        )

        # Act
        registry.register(basic_feature)
        registry.register(advanced_feature)
        basic_features = registry.list_features(tag="basic")
        advanced_features = registry.list_features(tag="advanced")

        # Assert
        assert len(basic_features) == 1
        assert basic_features[0].name == "win_percentage"
        assert len(advanced_features) == 1
        assert advanced_features[0].name == "offensive_rating"
