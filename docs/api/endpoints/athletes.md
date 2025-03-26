# Athlete Endpoints

This document provides detailed information about athlete-related endpoints in the ESPN API for NCAA Men's Basketball.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## Basic Athlete Information

### Get Athlete Details

```
GET /athletes/{athlete_id}
```

Retrieves detailed information about a specific athlete.

**Path Parameters:**
- `athlete_id` (required): The unique identifier for the athlete.

**Response Fields:**
- `id`: Unique identifier for the athlete
- `uid`: Universal identifier with format "s:40~l:41~a:{athlete_id}"
- `guid`: Global unique identifier
- `type`: Sport type (e.g., "basketball")
- `alternateIds`: Object containing alternate identifiers
- `firstName`: Athlete's first name
- `lastName`: Athlete's last name
- `fullName`: Athlete's full name
- `displayName`: Name displayed in UI
- `shortName`: Shortened name (typically "F. Lastname")
- `weight`: Athlete's weight in pounds
- `height`: Athlete's height in inches
- `dateOfBirth`: ISO 8601 formatted date of birth
- `age`: Athlete's age in years
- `slug`: URL-friendly identifier
- `jersey`: Jersey number
- `position`: Position information
- `linked`: Boolean indicating if the athlete is linked to other data
- `college`: Reference to college information
- `headshot`: Reference to headshot image
- `teams`: List of teams the athlete has been on
- `injuries`: List of injuries the athlete has had
- `statistics`: Reference to statistics information
- `status`: Current status (active, injured, etc.)

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137?lang=en&region=us",
  "id": "4433137",
  "uid": "s:40~l:41~a:4433137",
  "guid": "9a0fb84c-85b5-47b2-b877-d44ff3a5c93f",
  "type": "basketball",
  "alternateIds": {
    "sdr": "1145881"
  },
  "firstName": "Zach",
  "lastName": "Edey",
  "fullName": "Zach Edey",
  "displayName": "Zach Edey",
  "shortName": "Z. Edey",
  "weight": 305,
  "height": 87,
  "displayHeight": "7' 3\"",
  "dateOfBirth": "2002-05-14T04:00Z",
  "jersey": "15",
  "position": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/positions/1?lang=en&region=us",
    "id": "1",
    "name": "Center",
    "displayName": "Center",
    "abbreviation": "C",
    "leaf": true
  },
  "headshot": {
    "href": "https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/4433137.png",
    "alt": "Zach Edey"
  },
  "teams": [
    {
      "id": "2509",
      "uid": "s:40~l:41~t:2509",
      "slug": "purdue-boilermakers",
      "location": "Purdue",
      "name": "Boilermakers",
      "abbreviation": "PUR",
      "displayName": "Purdue Boilermakers",
      "shortDisplayName": "Boilermakers",
      "color": "9E928E",
      "alternateColor": "000000",
      "href": "https://api.espn.com/v1/sports/basketball/mens-college-basketball/teams/2509"
    }
  ],
  "links": [
    {
      "language": "en-US",
      "rel": ["playercard", "desktop", "athlete"],
      "href": "https://www.espn.com/mens-college-basketball/player/_/id/4433137/zach-edey",
      "text": "Player Card",
      "shortText": "Player Card",
      "isExternal": false,
      "isPremium": false
    }
  ],
  "injuries": []
}
```

## Athlete Statistics

### Get Athlete Statistics

```
GET /athletes/{athlete_id}/statistics/0
```

Retrieves comprehensive statistics for a specific athlete.

**Path Parameters:**
- `athlete_id` (required): The unique identifier for the athlete.

**Response Fields:**
- `categories`: Array of statistical categories
  - `name`: Name of the category (e.g., "offensive", "defensive")
  - `displayName`: Display name for the category
  - `abbreviation`: Standard abbreviation
  - `stats`: Array of statistics within this category

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137/statistics/0" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137/statistics/0?lang=en&region=us",
  "id": "0",
  "name": "Career",
  "abbreviation": "Career",
  "categories": [
    {
      "name": "offensive",
      "displayName": "Offensive",
      "abbreviation": "OFF",
      "stats": [
        {
          "name": "pts",
          "displayName": "Points",
          "shortDisplayName": "Pts",
          "description": "Points",
          "abbreviation": "Pts",
          "type": "points",
          "value": 23.2,
          "displayValue": "23.2"
        },
        {
          "name": "fg%",
          "displayName": "Field Goal Percentage",
          "shortDisplayName": "FG%",
          "description": "Field Goal Percentage",
          "abbreviation": "FG%",
          "type": "fieldGoalPercentage",
          "value": 62.1,
          "displayValue": "62.1"
        },
        {
          "name": "orpg",
          "displayName": "Offensive Rebounds Per Game",
          "shortDisplayName": "ORPG",
          "description": "Offensive Rebounds Per Game",
          "abbreviation": "ORPG",
          "type": "reboundsOffensive",
          "value": 4.0,
          "displayValue": "4.0"
        }
      ]
    },
    {
      "name": "defensive",
      "displayName": "Defensive",
      "abbreviation": "DEF",
      "stats": [
        {
          "name": "drpg",
          "displayName": "Defensive Rebounds Per Game",
          "shortDisplayName": "DRPG",
          "description": "Defensive Rebounds Per Game",
          "abbreviation": "DRPG",
          "type": "reboundsDefensive",
          "value": 8.8,
          "displayValue": "8.8"
        },
        {
          "name": "rpg",
          "displayName": "Rebounds Per Game",
          "shortDisplayName": "RPG",
          "description": "Rebounds Per Game",
          "abbreviation": "RPG",
          "type": "rebounds",
          "value": 12.8,
          "displayValue": "12.8"
        },
        {
          "name": "bpg",
          "displayName": "Blocks Per Game",
          "shortDisplayName": "BPG",
          "description": "Blocks Per Game",
          "abbreviation": "BPG",
          "type": "blocks",
          "value": 2.2,
          "displayValue": "2.2"
        }
      ]
    }
  ]
}
```

### Get Athlete Statistics Log

```
GET /athletes/{athlete_id}/statisticslog
```

Returns a history of statistics for a specific athlete across multiple seasons.

**Path Parameters:**
- `athlete_id` (required): The unique identifier for the athlete.

**Response Structure:**
- `count`: Total number of seasons with statistics
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of season statistics references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137/statisticslog" | jq
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
      "season": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024?lang=en&region=us",
        "year": 2024,
        "displayName": "2023-24"
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/2509?lang=en&region=us"
      },
      "statistics": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/2/athletes/4433137/statistics?lang=en&region=us"
      }
    },
    {
      "season": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2023?lang=en&region=us",
        "year": 2023,
        "displayName": "2022-23"
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2023/teams/2509?lang=en&region=us"
      },
      "statistics": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2023/types/2/athletes/4433137/statistics?lang=en&region=us"
      }
    }
  ]
}
```

## Season-Specific Athlete Information

### Get Athlete Details for a Season

```
GET /seasons/{year}/athletes/{athlete_id}
```

Retrieves information about a specific athlete for a particular season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `athlete_id` (required): The unique identifier for the athlete

**Response Structure:**
Similar to the basic athlete endpoint, but with season-specific data.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/athletes/4433137" | jq
```

### Get Season-Specific Athlete Statistics

```
GET /seasons/{year}/types/{type}/athletes/{athlete_id}/statistics
```

Retrieves statistics for a specific athlete in a particular season and season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason)
- `athlete_id` (required): The unique identifier for the athlete

**Response Structure:**
Similar to the athlete statistics endpoint, but filtered to the specified season and season type.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/2/athletes/4433137/statistics" | jq
```

### Get Athlete Projections

```
GET /seasons/{year}/types/{type}/athletes/{athlete_id}/projections
```

Retrieves statistical projections for an athlete in a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `athlete_id` (required): The unique identifier for the athlete

**Response Structure:**
Similar to the statistics endpoint, but contains projected values rather than actual statistics.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/athletes/4433137/projections" | jq
```

### Get Athlete Event Log

```
GET /seasons/{year}/athletes/{athlete_id}/eventlog
```

Returns a list of events (games) that an athlete has participated in during a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `athlete_id` (required): The unique identifier for the athlete

**Response Structure:**
- `count`: Total number of events
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of event references with athlete-specific performance data

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/athletes/4433137/eventlog" | jq
```

**Example Response:**
```json
{
  "count": 33,
  "pageIndex": 1,
  "pageSize": 25,
  "pageCount": 2,
  "items": [
    {
      "event": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661?lang=en&region=us"
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/2509?lang=en&region=us"
      },
      "stats": [
        {
          "name": "MIN",
          "displayValue": "30"
        },
        {
          "name": "FG",
          "displayValue": "14-18"
        },
        {
          "name": "3PT",
          "displayValue": "0-0"
        },
        {
          "name": "FT",
          "displayValue": "5-7"
        },
        {
          "name": "OREB",
          "displayValue": "5"
        },
        {
          "name": "DREB",
          "displayValue": "10"
        },
        {
          "name": "REB",
          "displayValue": "15"
        },
        {
          "name": "AST",
          "displayValue": "0"
        },
        {
          "name": "STL",
          "displayValue": "0"
        },
        {
          "name": "BLK",
          "displayValue": "3"
        },
        {
          "name": "TO",
          "displayValue": "5"
        },
        {
          "name": "PF",
          "displayValue": "2"
        },
        {
          "name": "PTS",
          "displayValue": "33"
        }
      ]
    }
  ]
}
```

### Get Athlete Notes

```
GET /seasons/{year}/athletes/{athlete_id}/notes
```

Returns notes and important information about an athlete for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `athlete_id` (required): The unique identifier for the athlete

**Response Structure:**
- `count`: Total number of notes
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of note objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/athletes/4433137/notes" | jq
``` 