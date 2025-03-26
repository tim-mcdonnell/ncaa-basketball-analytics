# Ranking Endpoints

This document provides detailed information about the ranking-related endpoints in the ESPN API for NCAA Men's Basketball.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## League Rankings Endpoint

### `/rankings`

Returns references to the latest available rankings for the current season.

**Response Structure:**
- `count`: Total number of ranking systems
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of references to specific ranking resources

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/rankings" | jq
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
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/20/rankings/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/20/rankings/2?lang=en&region=us"
    }
  ]
}
```

## Season Rankings Endpoint

### `/seasons/{year}/rankings`

Returns a list of all available rankings for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season (e.g., 2025)

**Response Structure:**
- Similar to the league rankings endpoint, but filters to the specified season

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/rankings" | jq
```

## Week Rankings Endpoint

### `/seasons/{year}/types/{type}/weeks/{week}/rankings`

Returns a list of all ranking systems available for a specific week.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason)
- `week` (required): The week number

**Response Structure:**
- `count`: Total number of ranking systems
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of references to specific ranking resources

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

## Specific Ranking Endpoint

### `/seasons/{year}/types/{type}/weeks/{week}/rankings/{ranking_id}`

Returns detailed poll rankings for a specific ranking system, week, and season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason)
- `week` (required): The week number
- `ranking_id` (required): The ranking system ID

**Common Ranking IDs:**
- 1: Associated Press Top 25 (AP Poll)
- 2: USA Today Coaches Poll

**Response Fields:**
- `id`: Unique identifier for the ranking system
- `name`: Full name of the ranking system (e.g., "AP Top 25")
- `shortName`: Abbreviated name (e.g., "AP")
- `type`: Type of poll (e.g., "student")
- `headline`: Display headline
- `shortHeadline`: Short display headline
- `pollDateTitle`: How poll timeframes are referenced
- `teams`: Array of ranked teams
- `others`: Array of teams receiving votes but not ranked in the top 25
- `droppedOut`: Array of teams that dropped out of the rankings from the previous poll

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/weeks/16/rankings/1" | jq
```

**Example Response:**
```json
{
  "id": "1",
  "name": "AP Top 25",
  "shortName": "AP",
  "type": "student",
  "headline": "AP Poll",
  "shortHeadline": "AP",
  "pollDateTitle": "week",
  "teams": [
    {
      "current": 1,
      "previous": 1,
      "points": 1571.0,
      "firstPlaceVotes": 61,
      "trend": "-",
      "record": {
        "summary": "32-2",
        "stats": [
          {
            "name": "wins",
            "displayName": "Wins",
            "shortDisplayName": "W",
            "description": "Wins",
            "abbreviation": "W",
            "type": "wins",
            "value": 32.0,
            "displayValue": "32"
          },
          {
            "name": "losses",
            "displayName": "Losses",
            "shortDisplayName": "L",
            "description": "Losses",
            "abbreviation": "L",
            "type": "losses",
            "value": 2.0,
            "displayValue": "2"
          }
        ]
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/2?lang=en&region=us"
      },
      "date": "2025-03-17T07:00Z",
      "lastUpdated": "2025-03-17T21:06Z"
    },
    {
      "current": 2,
      "previous": 2,
      "points": 1502.0,
      "firstPlaceVotes": 2,
      "trend": "-",
      "record": {
        "summary": "30-3",
        "stats": [
          {
            "name": "wins",
            "displayName": "Wins",
            "shortDisplayName": "W",
            "description": "Wins",
            "abbreviation": "W",
            "type": "wins",
            "value": 30.0,
            "displayValue": "30"
          },
          {
            "name": "losses",
            "displayName": "Losses",
            "shortDisplayName": "L",
            "description": "Losses",
            "abbreviation": "L",
            "type": "losses",
            "value": 3.0,
            "displayValue": "3"
          }
        ]
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/41?lang=en&region=us"
      },
      "date": "2025-03-17T07:00Z",
      "lastUpdated": "2025-03-17T21:07Z"
    }
  ],
  "others": [
    {
      "current": 0,
      "previous": 0,
      "points": 107.0,
      "firstPlaceVotes": 0,
      "trend": "+26",
      "record": {
        "summary": "0-0",
        "stats": []
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/2181?lang=en&region=us"
      }
    }
  ],
  "droppedOut": [
    {
      "current": 0,
      "previous": 24,
      "points": 55.0,
      "firstPlaceVotes": 0,
      "trend": "+24",
      "record": {
        "summary": "0-0",
        "stats": []
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/356?lang=en&region=us"
      }
    }
  ]
}
```

## Rankings Data Structure Details

### Ranked Teams

For each team in the `teams` array:

- `current`: Current rank position (1-25 for ranked teams)
- `previous`: Previous rank position
- `points`: Poll points received
- `firstPlaceVotes`: Number of first-place votes received
- `trend`: Trend indicator showing movement
  - `-`: No change
  - `+n`: Moved up n positions
  - `-n`: Moved down n positions
- `record`: Team record information
  - `summary`: Abbreviated record (e.g., "32-2")
  - `stats`: Detailed breakdown of record statistics
- `team`: Reference to detailed team information
- `date`: ISO timestamp of the ranking
- `lastUpdated`: ISO timestamp when the ranking was last updated

### Other Teams Receiving Votes

The `others` array contains teams that received votes but did not make the top 25. These teams have the same structure as ranked teams except their `current` value is 0.

### Teams That Dropped Out

The `droppedOut` array contains teams that were ranked in the previous poll but fell out of the current rankings. These teams have the same structure as ranked teams with `current` value of 0 and `previous` showing their former ranking. 