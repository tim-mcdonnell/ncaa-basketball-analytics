from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    AuthenticationError,
    ResourceNotFoundError,
    ValidationError,
    ServiceUnavailableError,
    ParseError,
    TimeoutError,
)


def test_base_api_error():
    """Test base APIError class."""
    message = "Test error message"
    status_code = 500
    error = APIError(message, status_code)

    assert error.message == message
    assert error.status_code == status_code
    assert str(error) == message


def test_rate_limit_error():
    """Test RateLimitError class."""
    message = "Rate limited"
    retry_after = 30
    error = RateLimitError(message, retry_after)

    assert error.message == message
    assert error.status_code == 429
    assert error.retry_after == retry_after
    assert str(error) == message


def test_authentication_error():
    """Test AuthenticationError class."""
    message = "Auth failed"
    error = AuthenticationError(message)

    assert error.message == message
    assert error.status_code == 401
    assert str(error) == message


def test_resource_not_found_error():
    """Test ResourceNotFoundError class."""
    resource_type = "Team"
    resource_id = "123"
    error = ResourceNotFoundError(resource_type, resource_id)

    assert error.status_code == 404
    assert f"{resource_type}" in error.message
    assert resource_id in error.message
    assert str(error) == error.message


def test_validation_error():
    """Test ValidationError class."""
    message = "Validation failed"
    errors = {"field": "error"}
    error = ValidationError(message, errors)

    assert error.message == message
    assert error.status_code == 400
    assert error.errors == errors
    assert str(error) == message


def test_service_unavailable_error():
    """Test ServiceUnavailableError class."""
    message = "Service down"
    error = ServiceUnavailableError(message)

    assert error.message == message
    assert error.status_code == 503
    assert str(error) == message


def test_parse_error():
    """Test ParseError class."""
    message = "Parse failed"
    error = ParseError(message)

    assert error.message == message
    assert error.status_code is None
    assert str(error) == message


def test_timeout_error():
    """Test TimeoutError class."""
    message = "Request timed out"
    error = TimeoutError(message)

    assert error.message == message
    assert error.status_code is None
    assert str(error) == message
