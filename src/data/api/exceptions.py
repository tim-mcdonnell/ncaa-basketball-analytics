"""Exception classes for the API module."""


class APIError(Exception):
    """Base exception class for API errors."""

    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, message: str = "API rate limit exceeded", retry_after: int = None):
        self.retry_after = retry_after
        super().__init__(message, status_code=429)


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class ResourceNotFoundError(APIError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: str, message: str = None):
        if message is None:
            message = f"{resource_type} with ID {resource_id} not found"
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message, status_code=404)


class ValidationError(APIError):
    """Raised when API request validation fails."""

    def __init__(self, message: str = "Validation failed", errors: dict = None):
        self.errors = errors or {}
        super().__init__(message, status_code=400)


class ServiceUnavailableError(APIError):
    """Raised when the API service is unavailable."""

    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, status_code=503)


class ConnectionResetError(APIError):
    """Raised when a connection is reset by the server or network issues."""

    def __init__(self, message: str = "Connection reset by peer"):
        super().__init__(message, status_code=None)


class ParseError(APIError):
    """Raised when parsing API responses fails."""

    def __init__(self, message: str = "Failed to parse API response"):
        super().__init__(message, status_code=None)


class TimeoutError(APIError):
    """Raised when an API request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, status_code=None)
