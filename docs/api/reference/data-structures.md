# ESPN API - Common Data Structures

This document provides detailed information about common data structures and patterns used throughout the ESPN v2 API for NCAA Men's Basketball.

## Basic Response Pattern

Most ESPN API responses follow a consistent pattern with these common elements:

1. Reference Objects with `$ref` fields
2. Pagination structure for list endpoints
3. Standardized field naming conventions
4. Nested objects with descriptive properties

## Reference Objects

### Ref Structure

Many responses contain reference objects that point to more detailed information. These references use the `$ref` field with a URL to the detailed resource.

```json
{
  "$ref": "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025"
}
```

Typically, the client would need to make additional HTTP requests to resolve these references. The API is designed to be deeply navigable through these references.

## Pagination Structure

List endpoints return paginated results with a consistent structure. The pagination information includes count, page size, current page index, and total number of pages.

```json
{
  "count": 363,         // Total number of items in the full dataset
  "pageIndex": 1,       // Current page (1-based indexing)
  "pageSize": 25,       // Number of items per page
  "pageCount": 15,      // Total number of pages
  "items": [...]        // Array of items for the current page
}
```

!!! tip "Navigation"
    To navigate through pages, use the `page` query parameter, for example: `?page=2`
    
    To change the number of items per page, use the `limit` query parameter, for example: `?limit=50`

## Team Object

Example of basic team information:

```json
{
  "id": "52",
  "uid": "s:40~l:41~t:52",
  "slug": "north-carolina-tar-heels",
  "location": "North Carolina",
  "name": "Tar Heels",
  "nickname": "UNC",
  "abbreviation": "UNC",
  "color": "7BAFD4",
  "alternateColor": "13294B",
  "logos": [
    {
      "href": "https://a.espncdn.com/i/teamlogos/ncaa/500/153.png",
      "width": 500,
      "height": 500,
      "alt": "North Carolina Tar Heels",
      "rel": ["full", "default"]
    }
  ]
}
```

## Event Object

Example of basic event information:

```json
{
  "id": "401478604",
  "uid": "s:40~l:41~e:401478604",
  "date": "2024-03-09T23:00Z",
  "name": "Duke at North Carolina",
  "shortName": "DUKE @ UNC",
  "season": {
    "year": 2025,
    "type": 2,
    "slug": "regular-season"
  },
  "competitions": [
    {
      "id": "401478604",
      "competitors": [
        {
          "id": "52",
          "homeAway": "home"
        },
        {
          "id": "150",
          "homeAway": "away"
        }
      ]
    }
  ]
}
```

## Rankings Data Structure

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

### Rankings Data Field Descriptions

- `id`: Unique identifier for the ranking system
- `name`: Full name of the ranking system (e.g., "AP Top 25")
- `shortName`: Abbreviated name (e.g., "AP")
- `type`: Type of poll (e.g., "student")
- `headline`: Display headline
- `shortHeadline`: Short display headline
- `pollDateTitle`: How poll timeframes are referenced
- `teams`: Array of ranked teams
    - `current`: Current rank position
    - `previous`: Previous rank position
    - `points`: Poll points received
    - `firstPlaceVotes`: Number of first-place votes received
    - `trend`: Trend indicator showing movement ("+", "-", or numeric value)
    - `record`: Team record information
        - `summary`: Abbreviated record (e.g., "32-2")
        - `stats`: Detailed breakdown of record statistics
    - `team`: Reference to detailed team information
    - `date`: ISO timestamp of the ranking
    - `lastUpdated`: ISO timestamp when the ranking was last updated
- `others`: Array of teams receiving votes but not ranked in the top 25
- `droppedOut`: Array of teams that dropped out of the rankings from the previous poll

## Franchise Data Structure

```json
{
  "id": "1",
  "uid": "s:40~l:41~f:1",
  "slug": "alaska-anchorage-seawolves",
  "location": "Alaska Anchorage",
  "name": "Seawolves",
  "nickname": "AK-Anchorage",
  "abbreviation": "UAA",
  "displayName": "Alaska Anchorage Seawolves",
  "shortDisplayName": "AK-Anchorage",
  "color": "000000",
  "isActive": true,
  "logos": [
    {
      "href": "https://a.espncdn.com/i/teamlogos/ncaa/500/1.png",
      "width": 500,
      "height": 500,
      "alt": "",
      "rel": ["full", "default"],
      "lastUpdated": "2019-11-05T17:36Z"
    }
  ],
  "team": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/1?lang=en&region=us"
  },
  "venue": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/venues/4823?lang=en&region=us",
    "id": "4823",
    "fullName": "Alaska Airlines Center",
    "address": {
      "city": "Anchorage",
      "state": "AK"
    },
    "indoor": true,
    "grass": false,
    "images": []
  }
}
```

### Franchise Data Field Descriptions

- `id`: Unique identifier for the franchise
- `uid`: Universal identifier with format "s:40~l:41~f:{id}"
- `slug`: URL-friendly identifier
- `location`: Geographic location (usually city and/or state)
- `name`: Team name (mascot)
- `nickname`: Alternative short name
- `abbreviation`: Official abbreviation
- `displayName`: Full display name (location + team name)
- `shortDisplayName`: Shortened display name
- `color`: Primary team color (hexadecimal)
- `isActive`: Boolean indicating if the franchise is currently active
- `logos`: Array of team logo images
- `team`: Reference to the current season's team record
- `venue`: Information about the team's home venue

## League Root Data Structure

```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball?lang=en&region=us",
  "id": "41",
  "guid": "11965179-4504-3b99-983e-83ea0e36d5f3",
  "uid": "s:40~l:41",
  "name": "NCAA Men's Basketball",
  "displayName": "NCAA Men's Basketball",
  "abbreviation": "NCAAM",
  "shortName": "NCAAM",
  "midsizeName": "NCAAM Basketball",
  "slug": "mens-college-basketball",
  "isTournament": false,
  "season": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025?lang=en&region=us",
    "year": 2025,
    "startDate": "2024-07-13T07:00Z",
    "endDate": "2025-04-09T06:59Z",
    "displayName": "2024-25",
    "type": {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/3?lang=en&region=us"
    }
  },
  "seasons": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons?lang=en&region=us"
  },
  "franchises": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/franchises?lang=en&region=us"
  },
  "teams": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams?lang=en&region=us"
  },
  "events": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events?lang=en&region=us"
  },
  "notes": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/notes?lang=en&region=us"
  },
  "rankings": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/rankings?lang=en&region=us"
  },
  "links": [
    {
      "language": "en-US",
      "rel": [
        "index",
        "desktop",
        "league"
      ],
      "href": "https://www.espn.com/mens-college-basketball/",
      "text": "Index",
      "shortText": "Index",
      "isExternal": false,
      "isPremium": false
    }
  ],
  "logos": [
    {
      "href": "https://a.espncdn.com/redesign/assets/img/icons/ESPN-icon-basketball.png",
      "width": 500,
      "height": 500,
      "alt": "",
      "rel": [
        "full",
        "default"
      ],
      "lastUpdated": "2021-08-03T15:56Z"
    }
  ],
  "gender": "MALE"
}
```

### League Root Data Field Descriptions

- `id`: Unique identifier for the league
- `guid`: Global unique identifier
- `uid`: Universal identifier with format "s:40~l:41"
- `name`: Full name of the league
- `displayName`: Display name used for UI presentation
- `abbreviation`: Standard abbreviation (e.g., "NCAAM")
- `shortName`: Shortened display name
- `midsizeName`: Medium-length display name
- `slug`: URL-friendly identifier
- `isTournament`: Boolean indicating if the league is a tournament format
- `season`: Information about the current season
- `seasons`: Reference to all available seasons
- `franchises`: Reference to all franchises in the league
- `teams`: Reference to teams in the current season
- `events`: Reference to events/games
- `notes`: Reference to league notes
- `rankings`: Reference to rankings data
- `links`: Array of related web links
- `logos`: Array of league logo images
- `gender`: Gender of the league (always "MALE" for men's basketball)

## Group Data Structure

```json
{
  "id": "1",
  "name": "Atlantic Coast",
  "abbreviation": "ACC",
  "shortName": "ACC",
  "midsizeName": "ACC",
  "season": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025?lang=en&region=us"
  },
  "children": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/1/children?lang=en&region=us"
  },
  "parent": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/90?lang=en&region=us"
  },
  "standings": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/1/standings?lang=en&region=us"
  },
  "isConference": true,
  "slug": "atlantic-coast",
  "teams": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/1/teams?lang=en&region=us"
  }
}
```

### Group Data Field Descriptions

- `id`: Unique identifier for the group
- `name`: Full name of the group (e.g., "Atlantic Coast", "NCAA Division I", etc.)
- `abbreviation`: Standard abbreviation (e.g., "ACC", "NCAA")
- `shortName`: Shortened display name
- `midsizeName`: Medium-length display name
- `season`: Reference to the season
- `children`: Reference to child groups
- `parent`: Reference to the parent group
- `standings`: Reference to group standings
- `isConference`: Boolean indicating if the group is a conference
- `slug`: URL-friendly identifier
- `teams`: Reference to teams in the group 