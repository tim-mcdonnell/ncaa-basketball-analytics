"""Tests for configuration validation functionality."""

import pytest
from pydantic import ValidationError

from src.config.base import BaseConfig
from src.config.validation import validate_config


def test_config_validation_basic():
    """Verify that basic validation works for configuration values."""

    # This test will pass if validate_config correctly identifies valid configurations
    class TestConfig(BaseConfig):
        name: str
        value: int

    # Valid configuration should pass validation
    valid_config = {"name": "test", "value": 42}
    result = validate_config(TestConfig, valid_config)
    assert result.name == "test"
    assert result.value == 42


def test_config_validation_types():
    """Verify that type validation works correctly for configuration values."""

    # This test will pass if validate_config correctly identifies type mismatches
    class TypeTestConfig(BaseConfig):
        name: str
        value: int

    # Invalid type for 'value' should raise ValidationError
    invalid_config = {"name": "test", "value": "not_an_int"}
    with pytest.raises(ValidationError):
        validate_config(TypeTestConfig, invalid_config)


def test_config_validation_constraints():
    """Verify that constraints like ranges are validated correctly."""
    from pydantic import Field

    class ConstraintTestConfig(BaseConfig):
        count: int = Field(ge=0, le=100)  # Must be between 0 and 100
        name: str = Field(min_length=3)  # Must be at least 3 characters

    # Valid constraints should pass
    valid_config = {"count": 50, "name": "test"}
    result = validate_config(ConstraintTestConfig, valid_config)
    assert result.count == 50
    assert result.name == "test"

    # Invalid constraints should fail
    with pytest.raises(ValidationError):
        validate_config(ConstraintTestConfig, {"count": 101, "name": "test"})

    with pytest.raises(ValidationError):
        validate_config(ConstraintTestConfig, {"count": 50, "name": "ab"})


def test_config_error_messages():
    """Verify that validation errors provide clear, helpful error messages."""
    from pydantic import Field

    class ErrorTestConfig(BaseConfig):
        port: int = Field(ge=1024, le=65535, description="Server port number")
        host: str = Field(min_length=3, description="Server hostname")

    # Test error message for range constraint
    try:
        validate_config(ErrorTestConfig, {"port": 80, "host": "localhost"})
        pytest.fail("Validation should have failed for port < 1024")
    except ValidationError as e:
        error_text = str(e)
        assert "port" in error_text
        assert "1024" in error_text
        assert "greater than or equal to 1024" in error_text

    # Test error message for string length
    try:
        validate_config(ErrorTestConfig, {"port": 8080, "host": "a"})
        pytest.fail("Validation should have failed for host < 3 chars")
    except ValidationError as e:
        error_text = str(e)
        assert "host" in error_text
        assert "at least 3 characters" in error_text or "shorter than 3 characters" in error_text


def test_config_dot_notation():
    """Verify that config can be accessed using dot notation."""

    class DotTestConfig(BaseConfig):
        server: dict
        database: dict

    config_data = {
        "server": {"host": "localhost", "port": 8080},
        "database": {"url": "postgresql://user:pass@localhost/db"},
    }

    config = validate_config(DotTestConfig, config_data)

    # Convert to dot notation dictionary
    dot_config = config.dot_dict()

    # Test dot notation access
    assert dot_config.server.host == "localhost"
    assert dot_config.server.port == 8080
    assert dot_config.database.url == "postgresql://user:pass@localhost/db"
