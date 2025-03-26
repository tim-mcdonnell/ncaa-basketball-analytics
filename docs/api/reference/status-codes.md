# Response Status Codes

The ESPN API returns standard HTTP status codes to indicate the success or failure of API requests.

## Success Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | The request was successful. This is the most common success response for GET requests. |
| 201 | Created | The request was successful and a new resource was created. This is typically used for POST requests. |
| 204 | No Content | The request was successful but there is no content to return. Often used for DELETE operations. |

## Client Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 400 | Bad Request | The request was malformed or contains invalid parameters. Check the request syntax and parameters. |
| 401 | Unauthorized | Authentication is required. This typically means that you need to include valid credentials. |
| 403 | Forbidden | The server understood the request but refuses to authorize it. This typically means you don't have permission to access the requested resource. |
| 404 | Not Found | The requested resource does not exist. This could be due to an incorrect URL or a resource that has been removed. |
| 429 | Too Many Requests | Rate limit has been exceeded. The API limits the number of requests that can be made in a given time period. |

## Server Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 500 | Internal Server Error | An unexpected error occurred on the server. If this persists, contact ESPN API support. |
| 502 | Bad Gateway | The server received an invalid response from an upstream server. |
| 503 | Service Unavailable | The server is currently unable to handle the request due to temporary overloading or maintenance. |
| 504 | Gateway Timeout | The server did not receive a timely response from an upstream server. |

## Error Response Format

When the API returns an error response, it typically includes a JSON body with additional details about the error:

```json
{
  "status": 400,
  "message": "Invalid parameter: 'date' must be in YYYYMMDD format",
  "code": "INVALID_PARAMETER",
  "detail": "The date parameter '20240101' is not in the correct format. Use 'YYYYMMDD' format (e.g., '20240101')."
}
```

### Common Error Fields

| Field | Description |
|-------|-------------|
| `status` | The HTTP status code associated with the error. |
| `message` | A brief description of the error that occurred. |
| `code` | A unique code that identifies the specific error type. |
| `detail` | A more detailed explanation of the error, often including suggestions for resolving the issue. |

## Error Handling Best Practices

When working with the ESPN API, consider these error handling best practices:

1. **Handle Rate Limiting**: Implement exponential backoff when receiving 429 errors.
2. **Graceful Degradation**: Design your application to handle temporary service disruptions (503 errors).
3. **Validate Input Parameters**: Verify that your request parameters are correctly formatted to avoid 400 errors.
4. **Check Resource Existence**: Implement fallback behavior when resources are not found (404 errors).
5. **Log Errors**: Maintain detailed logs of API errors to help diagnose issues.

## Example Error Scenarios

### Invalid Date Format

```http
GET /v2/sports/basketball/leagues/mens-college-basketball/events?dates=01-01-2024
```

```json
{
  "status": 400,
  "message": "Invalid date format",
  "code": "INVALID_DATE_FORMAT",
  "detail": "The 'dates' parameter must be in YYYYMMDD format (e.g., 20240101)"
}
```

### Resource Not Found

```http
GET /v2/sports/basketball/leagues/mens-college-basketball/teams/999999
```

```json
{
  "status": 404,
  "message": "Team not found",
  "code": "RESOURCE_NOT_FOUND",
  "detail": "The requested team with ID '999999' does not exist"
}
```

### Rate Limit Exceeded

```json
{
  "status": 429,
  "message": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "detail": "You have exceeded the rate limit of 100 requests per minute. Please wait and try again later."
}
``` 