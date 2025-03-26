# ESPN API Endpoints - NCAA Men's Basketball

This section contains detailed documentation for all available endpoints in the ESPN API for NCAA Men's Basketball. These endpoints provide access to a wide range of data, from team and player information to game results, statistics, and more.

## Base URL

All API endpoints use the following base URL:

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## Available Endpoints

The API is organized into several logical categories of endpoints:

### [Athlete Endpoints](athletes.md)

Athlete endpoints provide access to information about individual players, including:

- Basic player information and profiles
- Career and season statistics
- Athlete performance in specific games
- Player projections and historical data

### [Event Endpoints](events.md)

Event endpoints retrieve information about games and matchups, including:

- Game details and summaries
- Play-by-play data
- Box scores and statistics
- Game calendars and schedules

### [Team Endpoints](teams.md)

Team endpoints provide information about college basketball teams, including:

- Team rosters and details
- Team statistics and records
- Team schedules and results
- Historical team data

### [Season Endpoints](seasons.md)

Season endpoints access season-specific information, including:

- Season details and dates
- Season types (regular season, postseason)
- Weekly schedules
- Season statistics

### [Rankings Endpoints](rankings.md)

Rankings endpoints provide access to various poll and ranking data:

- AP Top 25 poll
- Coaches poll
- Selection committee rankings
- Historical ranking data

### [Awards Endpoints](awards.md)

Awards endpoints retrieve information about individual and team accolades:

- Player of the Year awards
- All-American teams
- Conference awards
- Historical award winners

## Common Response Patterns

All endpoints follow consistent patterns for responses:

1. **Direct data endpoints** return the requested resource directly
2. **Collection endpoints** return paginated lists of items
3. **Reference endpoints** return links to related resources

## Authentication

Most endpoints in this API are publicly accessible without authentication. However, some data may be limited or require specific permissions.

## Rate Limiting

To ensure service stability, API requests are subject to rate limiting. For more information, see the [Rate Limiting](../guides/rate-limiting.md) guide.

## Next Steps

For detailed information about specific endpoints, select a category from the navigation menu or links above. Each endpoint documentation includes:

- URL structure and parameters
- Response format and field descriptions
- Example requests and responses
- Usage notes and limitations 