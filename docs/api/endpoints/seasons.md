# Season Endpoints

This document provides detailed information about the NCAA Men's Basketball season-related endpoints in the ESPN v2 API.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## Season List Endpoint

### `/seasons`

Returns a paginated list of all available seasons.

**Query Parameters:**
- `pageSize` (optional): Number of items per page (default: 25)
- `pageIndex` (optional): Page number (default: 1)

**Response Structure:**
- `count`: Total number of seasons available
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of season references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons" | jq
```

**Example Response:**
```json
{
  "count": 21,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024?lang=en&region=us"
    }
  ]
}
```

## Specific Season Endpoint

### `/seasons/{year}`

Returns detailed information about a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season (e.g., 2025)

**Response Fields:**
- `$ref`: Reference URL to the season resource
- `year`: Four-digit year identifier
- `startDate`: ISO 8601 formatted season start date
- `endDate`: ISO 8601 formatted season end date
- `displayName`: Human-readable season name (e.g., "2024-25")
- `type`: Object containing information about the current season type
- `types`: Reference to all season types for this season

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025?lang=en&region=us",
  "year": 2025,
  "startDate": "2024-07-13T07:00Z",
  "endDate": "2025-04-09T06:59Z",
  "displayName": "2024-25",
  "type": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2?lang=en&region=us"
  },
  "types": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types?lang=en&region=us"
  }
}
```

## Season Types Endpoint

### `/seasons/{year}/types`

Returns a list of all season types (preseason, regular season, postseason, etc.) for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `count`: Total number of season types
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of season type references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types" | jq
```

**Example Response:**
```json
{
  "count": 4,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/3?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/4?lang=en&region=us"
    }
  ]
}
```

## Specific Season Type Endpoint

### `/seasons/{year}/types/{type}`

Returns detailed information about a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason, 4=Off Season)

**Response Fields:**
- `$ref`: Reference URL to the season type resource
- `id`: Unique identifier for the season type
- `type`: Numeric type identifier
- `name`: Human-readable name of the season type
- `abbreviation`: Short abbreviation for the season type
- `startDate`: ISO 8601 formatted start date for this season type
- `endDate`: ISO 8601 formatted end date for this season type
- `hasGroups`: Boolean indicating if this season type has group information
- `hasStandings`: Boolean indicating if this season type has standings
- `hasLegs`: Boolean indicating if this season type has "legs" (rarely used in basketball)
- `slug`: URL-friendly identifier

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2?lang=en&region=us",
  "id": "2",
  "type": 2,
  "name": "Regular Season",
  "abbreviation": "reg",
  "year": 2025,
  "startDate": "2024-11-04T08:00Z",
  "endDate": "2025-03-18T06:59Z",
  "hasGroups": true,
  "hasStandings": true,
  "hasLegs": false,
  "slug": "regular-season",
  "weeks": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks?lang=en&region=us"
  },
  "groups": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups?lang=en&region=us"
  }
}
```

## Season Type Weeks Endpoint

### `/seasons/{year}/types/{type}/weeks`

Returns a list of all weeks for a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID

**Response Structure:**
- `count`: Total number of weeks
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of week references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks" | jq
```

**Example Response:**
```json
{
  "count": 19,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/2?lang=en&region=us"
    }
  ]
}
```

## Specific Week Endpoint

### `/seasons/{year}/types/{type}/weeks/{week}`

Returns detailed information about a specific week.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `week` (required): The week number

**Response Fields:**
- `$ref`: Reference URL to the week resource
- `id`: Unique identifier for the week
- `number`: Week number
- `startDate`: ISO 8601 formatted start date for the week
- `endDate`: ISO 8601 formatted end date for the week
- `text`: Display text for the week (e.g., "Week 1")
- `rankings`: Reference to rankings for this week

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/1" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/1?lang=en&region=us",
  "id": "1",
  "number": 1,
  "startDate": "2024-11-04T08:00Z",
  "endDate": "2024-11-11T07:59Z",
  "text": "Week 1",
  "rankings": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/1/rankings?lang=en&region=us"
  }
}
```

## Week Rankings Endpoint

### `/seasons/{year}/types/{type}/weeks/{week}/rankings`

Returns a list of rankings (polls) for a specific week.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `week` (required): The week number

**Response Structure:**
- `count`: Total number of ranking systems
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of ranking references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/16/rankings" | jq
```

**Example Response:**
```json
{
  "count": 2,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/16/rankings/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/16/rankings/2?lang=en&region=us"
    }
  ]
}
```

## Week Events Endpoint

### `/seasons/{year}/types/{type}/weeks/{week}/events`

Returns a list of events (games) scheduled for a specific week.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `week` (required): The week number

**Query Parameters:**
- `limit` (optional): Number of events to return (default: 25)
- `groups` (optional): Filter by group/conference ID

**Response Structure:**
- `count`: Total number of events
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of event references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/1/events" | jq
```

**Example Response:**
```json
{
  "count": 124,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 5,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401474516?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401474517?lang=en&region=us"
    }
  ]
}
```

## Season Athletes Endpoint

### `/seasons/{year}/athletes`

Returns a paginated list of athletes for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Query Parameters:**
- `limit` (optional): Number of athletes to return per page (default: 25)
- `page` (optional): Page number (default: 1)
- `groups` (optional): Filter by group/conference ID
- `teams` (optional): Filter by team ID

**Response Structure:**
- `count`: Total number of athletes
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of athlete references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes?limit=10" | jq
```

**Example Response:**
```json
{
  "count": 5647,
  "pageIndex": 1,
  "pageSize": 10,
  "pageCount": 565,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes/4433261?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes/4433262?lang=en&region=us"
    }
  ]
}
```

## Season Teams Endpoint

### `/seasons/{year}/teams`

Returns a paginated list of teams for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Query Parameters:**
- `limit` (optional): Number of teams to return per page (default: 25)
- `page` (optional): Page number (default: 1)
- `groups` (optional): Filter by group/conference ID
- `active` (optional): Filter for active teams (true/false)

**Response Structure:**
- `count`: Total number of teams
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of team references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams?limit=10" | jq
```

**Example Response:**
```json
{
  "count": 925,
  "pageIndex": 1,
  "pageSize": 10,
  "pageCount": 93,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/2?lang=en&region=us"
    }
  ]
}
```

## Season Power Index Endpoint

### `/seasons/{year}/powerindex`

Returns power index rankings for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Fields:**
- `count`: Total number of teams with power index rankings
- `pageIndex`: Current page number
- `pageSize`: Number of teams per page
- `pageCount`: Total number of pages
- `items`: Array of team power index entries

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/powerindex" | jq
```

**Example Response:**
```json
{
  "count": 363,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 15,
  "items": [
    {
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/127?lang=en&region=us"
      },
      "stats": [
        {
          "name": "bpi",
          "displayName": "Basketball Power Index",
          "shortDisplayName": "BPI",
          "description": "Basketball Power Index",
          "abbreviation": "BPI",
          "value": 13.9,
          "displayValue": "13.9"
        },
        {
          "name": "rank",
          "displayName": "Rank",
          "shortDisplayName": "Rank",
          "description": "Rank",
          "abbreviation": "Rank",
          "value": 1.0,
          "displayValue": "1"
        },
        {
          "name": "rankChange",
          "displayName": "Rank Change",
          "shortDisplayName": "Rank Chg",
          "description": "Rank Change",
          "abbreviation": "Rank Chg",
          "value": 0.0,
          "displayValue": "--"
        },
        {
          "name": "trend",
          "displayName": "Trend",
          "shortDisplayName": "TREND",
          "description": "Trend",
          "abbreviation": "TREND",
          "value": 0.0,
          "displayValue": "EVEN"
        }
      ]
    }
  ]
}
```

## Season Power Index Leaders Endpoint

### `/seasons/{year}/powerIndex/leaders`

Returns statistical leaders based on power index categories for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `id`: Unique identifier for the leader category
- `name`: Name of the statistic
- `displayName`: Display name for the statistic
- `shortDisplayName`: Abbreviated display name
- `description`: Description of the statistic
- `abbreviation`: Standard abbreviation
- `leaders`: Array of top athletes for this statistic

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/powerIndex/leaders" | jq
```

## Season Awards Endpoint

### `/seasons/{year}/awards`

Returns awards given for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `count`: Total number of award categories
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/awards" | jq
```

## Calendar Endpoint

### `/calendar`

Returns calendar information for the league including key dates and schedule details.

**Response Fields:**
- `count`: Total number of calendar items
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of calendar type references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar" | jq
```

**Example Response:**
```json
{
  "count": 4,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar/ondays?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar/offdays?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar/blacklist?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar/whitelist?lang=en&region=us"
    }
  ]
}
```

## Season Futures Endpoint

### `/seasons/{year}/futures`

Returns a list of futures/betting markets for the specified season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `count`: Total number of futures markets
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of futures market objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/futures" | jq
```

## Season Rankings Endpoint

### `/seasons/{year}/rankings`

Returns a list of available rankings for the specified season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `count`: Total number of ranking sets
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of ranking references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/rankings" | jq
``` 