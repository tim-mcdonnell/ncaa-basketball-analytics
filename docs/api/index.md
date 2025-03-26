# ESPN API Documentation - NCAA Men's Basketball

Welcome to the comprehensive documentation for ESPN's NCAA Men's Basketball API. This documentation will help you understand and utilize the various endpoints available for accessing game data, team information, player statistics, and more.

## Getting Started

!!! info "API Base URL"
    The base URL for all endpoints is:
    ```
    https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
    ```

### Authentication

The documented endpoints are publicly accessible and don't require authentication for basic usage. However, rate limiting may apply.

### Request Format

All endpoints accept standard HTTP GET requests and return JSON responses.

## Documentation Structure

This documentation is organized into the following sections:

### Core Endpoints

- [Season Endpoints](endpoints/seasons.md) - Season, week, and calendar information
- [Team Endpoints](endpoints/teams.md) - Team data, rosters, and statistics
- [Athlete Endpoints](endpoints/athletes.md) - Player information and statistics
- [Event Endpoints](endpoints/events.md) - Game data, play-by-play, and competitions
- [Ranking Endpoints](endpoints/rankings.md) - AP polls and other ranking systems
- [Award Endpoints](endpoints/awards.md) - Player and coach awards and honors

### Reference

- [Data Structures](reference/data-structures.md) - Common data structures and patterns
- [Status Codes](reference/status-codes.md) - API response codes and error handling
- [Query Parameters](reference/query-parameters.md) - Common query parameters

## Common Query Parameters

Many endpoints support the following parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lang` | String | Language code for localized content | `?lang=en` |
| `region` | String | Regional localization | `?region=us` |
| `limit` | Integer | Number of items per page | `?limit=100` |
| `page` | Integer | Page number for paginated results | `?page=2` |

## Feedback and Support

If you have feedback about this documentation or questions about the API, please reach out to the ESPN developer team.
