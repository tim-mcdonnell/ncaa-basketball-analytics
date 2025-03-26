# Team Endpoints

This document provides detailed information about team-related endpoints in the ESPN API for NCAA Men's Basketball.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## Team List Endpoint

### `/seasons/{year}/teams`

Returns a paginated list of teams for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season (e.g., 2025)

**Query Parameters:**
- `limit` (optional): Number of teams to return per page (default: 25)
- `page` (optional): Page number (default: 1)
- `active` (optional): Filter for active teams only (true/false)
- `groups` (optional): Filter by group/conference ID
- `lang` (optional): Language for the response (default: en)
- `region` (optional): Region for the response (default: us)

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

## Team Detail Endpoint

### `/seasons/{year}/teams/{team_id}`

Returns detailed information about a specific team.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Fields:**
- `id`: Unique identifier for the team
- `uid`: Universal identifier with format "s:40~l:41~t:{team_id}"
- `slug`: URL-friendly identifier
- `location`: Geographic location (city/state)
- `name`: Team name (mascot)
- `nickname`: Alternative short name
- `abbreviation`: Official abbreviation
- `displayName`: Full display name (location + team name)
- `shortDisplayName`: Shortened display name
- `color`: Primary team color (hexadecimal)
- `alternateColor`: Secondary team color (hexadecimal)
- `isActive`: Boolean indicating if the team is currently active
- `isAllStar`: Boolean indicating if the team is an all-star team
- `logos`: Array of team logo images
- `record`: Reference to team record information
- `groups`: Reference to conferences/divisions the team belongs to
- `ranks`: Reference to current team rankings
- `statistics`: Reference to team statistics
- `links`: Array of related web links
- `franchise`: Reference to franchise information

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52?lang=en&region=us",
  "id": "52",
  "uid": "s:40~l:41~t:52",
  "slug": "north-carolina-tar-heels",
  "location": "North Carolina",
  "name": "Tar Heels",
  "nickname": "UNC",
  "abbreviation": "UNC",
  "displayName": "North Carolina Tar Heels",
  "shortDisplayName": "UNC",
  "color": "7BAFD4",
  "alternateColor": "13294B",
  "isActive": true,
  "isAllStar": false,
  "logos": [
    {
      "href": "https://a.espncdn.com/i/teamlogos/ncaa/500/153.png",
      "width": 500,
      "height": 500,
      "alt": "North Carolina Tar Heels",
      "rel": ["full", "default"],
      "lastUpdated": "2018-06-05T12:11Z"
    }
  ],
  "record": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/teams/52/record?lang=en&region=us"
  },
  "groups": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/groups?lang=en&region=us"
  },
  "ranks": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/ranks?lang=en&region=us"
  },
  "statistics": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/statistics?lang=en&region=us"
  },
  "links": [
    {
      "language": "en-US",
      "rel": ["clubhouse", "desktop", "team"],
      "href": "https://www.espn.com/mens-college-basketball/team/_/id/153/north-carolina-tar-heels",
      "text": "Clubhouse",
      "shortText": "Clubhouse",
      "isExternal": false,
      "isPremium": false
    }
  ],
  "franchise": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/franchises/153?lang=en&region=us"
  }
}
```

## Team Athletes Endpoint

### `/seasons/{year}/teams/{team_id}/athletes`

Returns a paginated list of athletes on a team's roster.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Query Parameters:**
- `limit` (optional): Number of athletes to return per page (default: 25)
- `page` (optional): Page number (default: 1)

**Response Structure:**
- `count`: Total number of athletes on the team
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of athlete references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/athletes" | jq
```

**Example Response:**
```json
{
  "count": 15,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes/4432857?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes/4432858?lang=en&region=us"
    }
  ]
}
```

## Team Events Endpoint

### `/seasons/{year}/teams/{team_id}/events`

Returns a paginated list of events (games) for a specific team's schedule/results.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Query Parameters:**
- `limit` (optional): Number of events to return per page (default: 25)
- `page` (optional): Page number (default: 1)
- `seasontype` (optional): Filter by season type (1=Preseason, 2=Regular Season, 3=Postseason)

**Response Structure:**
- `count`: Total number of events
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of event references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/events?seasontype=2" | jq
```

**Example Response:**
```json
{
  "count": 31,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 2,
  "items": [
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401474610?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401474611?lang=en&region=us"
    }
  ]
}
```

## Team Coaches Endpoint

### `/seasons/{year}/teams/{team_id}/coaches`

Returns a list of coaches for a specific team.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of coaches
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of coach objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/coaches" | jq
```

**Example Response:**
```json
{
  "count": 1,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 1,
  "items": [
    {
      "id": "305",
      "firstName": "Hubert",
      "lastName": "Davis",
      "experience": 3,
      "status": "active",
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52?lang=en&region=us"
      }
    }
  ]
}
```

## Team Statistics Endpoint

### `/seasons/{year}/teams/{team_id}/statistics`

Returns comprehensive statistical data for a specific team.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Fields:**
- `id`: Unique identifier for the statistics record
- `name`: Name of the statistic category
- `abbreviation`: Standard abbreviation
- `categories`: Array of statistical categories
  - `name`: Category name (e.g., "offense", "defense")
  - `displayName`: Display name for the category
  - `stats`: Array of specific statistics in this category

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/statistics" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/statistics?lang=en&region=us",
  "id": "52",
  "name": "Team Statistics",
  "abbreviation": "TEAM",
  "categories": [
    {
      "name": "offense",
      "displayName": "Offense",
      "stats": [
        {
          "name": "pts",
          "displayName": "Points",
          "shortDisplayName": "Pts",
          "description": "Points",
          "abbreviation": "Pts",
          "value": 81.4,
          "displayValue": "81.4"
        },
        {
          "name": "fg%",
          "displayName": "Field Goal Percentage",
          "shortDisplayName": "FG%",
          "description": "Field Goal Percentage",
          "abbreviation": "FG%",
          "value": 45.7,
          "displayValue": "45.7"
        }
      ]
    },
    {
      "name": "defense",
      "displayName": "Defense",
      "stats": [
        {
          "name": "oppPts",
          "displayName": "Opponent Points",
          "shortDisplayName": "Opp Pts",
          "description": "Opponent Points",
          "abbreviation": "Opp Pts",
          "value": 72.3,
          "displayValue": "72.3"
        }
      ]
    }
  ]
}
```

## Season Type Team Statistics Endpoint

### `/seasons/{year}/types/{type}/teams/{team_id}/statistics`

Returns statistical data for a team filtered by a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason)
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- Similar to the team statistics endpoint, but filtered to the specified season type

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/teams/52/statistics" | jq
```

## Team Record Endpoint

### `/seasons/{year}/types/{type}/teams/{team_id}/record`

Returns detailed record information for a team in a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `team_id` (required): The unique identifier for the team

**Response Fields:**
- `summary`: Abbreviated record (e.g., "20-10")
- `items`: Array of record type objects
  - `summary`: Record summary
  - `description`: Description of the record type
  - `type`: Type identifier
  - `stats`: Detailed statistics for this record type

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/teams/52/record" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/teams/52/record?lang=en&region=us",
  "summary": "20-10",
  "stats": [
    {
      "name": "wins",
      "displayName": "Wins",
      "shortDisplayName": "W",
      "description": "Wins",
      "abbreviation": "W",
      "type": "wins",
      "value": 20.0,
      "displayValue": "20"
    },
    {
      "name": "losses",
      "displayName": "Losses",
      "shortDisplayName": "L",
      "description": "Losses",
      "abbreviation": "L",
      "type": "losses",
      "value": 10.0,
      "displayValue": "10"
    }
  ],
  "items": [
    {
      "summary": "20-10",
      "description": "Total",
      "type": {
        "id": "0",
        "name": "total",
        "displayName": "Total",
        "abbreviation": "Total"
      },
      "stats": [
        {
          "name": "wins",
          "displayName": "Wins",
          "shortDisplayName": "W",
          "description": "Wins",
          "abbreviation": "W",
          "type": "wins",
          "value": 20.0,
          "displayValue": "20"
        },
        {
          "name": "losses",
          "displayName": "Losses",
          "shortDisplayName": "L",
          "description": "Losses",
          "abbreviation": "L",
          "type": "losses",
          "value": 10.0,
          "displayValue": "10"
        }
      ]
    },
    {
      "summary": "12-8",
      "description": "Conference",
      "type": {
        "id": "1",
        "name": "vsconf",
        "displayName": "Conference",
        "abbreviation": "CONF"
      },
      "stats": [
        {
          "name": "wins",
          "displayName": "Wins",
          "shortDisplayName": "W",
          "description": "Wins",
          "abbreviation": "W",
          "type": "wins",
          "value": 12.0,
          "displayValue": "12"
        },
        {
          "name": "losses",
          "displayName": "Losses",
          "shortDisplayName": "L",
          "description": "Losses",
          "abbreviation": "L",
          "type": "losses",
          "value": 8.0,
          "displayValue": "8"
        }
      ]
    },
    {
      "summary": "12-2",
      "description": "Home",
      "type": {
        "id": "2",
        "name": "home",
        "displayName": "Home",
        "abbreviation": "HOME"
      },
      "stats": [
        {
          "name": "wins",
          "displayName": "Wins",
          "shortDisplayName": "W",
          "description": "Wins",
          "abbreviation": "W",
          "type": "wins",
          "value": 12.0,
          "displayValue": "12"
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
    {
      "summary": "6-6",
      "description": "Away",
      "type": {
        "id": "3",
        "name": "road",
        "displayName": "Away",
        "abbreviation": "AWAY"
      },
      "stats": [
        {
          "name": "wins",
          "displayName": "Wins",
          "shortDisplayName": "W",
          "description": "Wins",
          "abbreviation": "W",
          "type": "wins",
          "value": 6.0,
          "displayValue": "6"
        },
        {
          "name": "losses",
          "displayName": "Losses",
          "shortDisplayName": "L",
          "description": "Losses",
          "abbreviation": "L",
          "type": "losses",
          "value": 6.0,
          "displayValue": "6"
        }
      ]
    },
    {
      "summary": "2-2",
      "description": "Neutral",
      "type": {
        "id": "4",
        "name": "neutral",
        "displayName": "Neutral",
        "abbreviation": "NEUT"
      },
      "stats": [
        {
          "name": "wins",
          "displayName": "Wins",
          "shortDisplayName": "W",
          "description": "Wins",
          "abbreviation": "W",
          "type": "wins",
          "value": 2.0,
          "displayValue": "2"
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
    }
  ]
}
```

## Team Ranks Endpoint

### `/seasons/{year}/teams/{team_id}/ranks`

Returns current rankings for a specific team across various ranking systems.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of rankings
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of ranking objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/ranks" | jq
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
      "type": "AP",
      "rank": 7
    },
    {
      "type": "Coaches",
      "rank": 8
    }
  ]
}
```

## Against the Spread (ATS) Endpoint

### `/seasons/{year}/types/{type}/teams/{team_id}/ats`

Returns against-the-spread statistics for a specific team.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `records`: Array of ATS record objects
  - `type`: Record type (e.g., "overall", "home", "away")
  - `summary`: Abbreviated record (e.g., "15-12-3")
  - `winsATS`: Wins against the spread
  - `lossesATS`: Losses against the spread
  - `pushesATS`: Pushes against the spread

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/teams/52/ats" | jq
```

**Example Response:**
```json
{
  "records": [
    {
      "type": "overall",
      "summary": "15-12-3",
      "winsATS": 15,
      "lossesATS": 12,
      "pushesATS": 3
    },
    {
      "type": "home",
      "summary": "8-5-1",
      "winsATS": 8,
      "lossesATS": 5,
      "pushesATS": 1
    },
    {
      "type": "away",
      "summary": "5-6-1",
      "winsATS": 5,
      "lossesATS": 6,
      "pushesATS": 1
    },
    {
      "type": "neutral",
      "summary": "2-1-1",
      "winsATS": 2,
      "lossesATS": 1,
      "pushesATS": 1
    }
  ]
}
```

## Team Awards Endpoint

### `/seasons/{year}/teams/{team_id}/awards`

Returns awards received by a specific team in a given season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of awards
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/awards" | jq
```

## Team Notes Endpoint

### `/teams/{team_id}/notes`

Returns notes and important information related to a specific team.

**Path Parameters:**
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of notes
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of note objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/teams/52/notes" | jq
```

## Team Groups Endpoint

### `/seasons/{year}/teams/{team_id}/groups`

Returns information about the conferences and divisions a team belongs to.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of groups
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of group references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/groups" | jq
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
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/90?lang=en&region=us"
    }
  ]
}
``` 