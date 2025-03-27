# Award Endpoints

This document provides detailed information about the NCAA Men's Basketball award-related endpoints in the ESPN v2 API.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## League Awards Endpoint

### `/awards`

Returns a list of all award categories defined for the league.

**Response Structure:**
- `count`: Total number of award categories
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award category references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards" | jq
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
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/1?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/2?lang=en&region=us"
    },
    {
      "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/3?lang=en&region=us"
    }
  ]
}
```

## Award Detail Endpoint

### `/awards/{award_id}`

Returns information about a specific award category.

**Path Parameters:**
- `award_id` (required): The unique identifier for the award category

**Response Fields:**
- `id`: Unique identifier for the award
- `name`: Full name of the award (e.g., "Player of the Year")
- `displayName`: Display name used for UI presentation
- `shortDisplayName`: Shortened display name
- `description`: Description of the award criteria
- `recipients`: Information about historical recipients of the award

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/1" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/1?lang=en&region=us",
  "id": "1",
  "name": "Player of the Year",
  "displayName": "Player of the Year",
  "shortDisplayName": "POY",
  "description": "Annual award given to the top player in NCAA Men's Basketball",
  "recipients": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/1/recipients?lang=en&region=us"
  }
}
```

## Award Recipients Endpoint

### `/awards/{award_id}/recipients`

Returns a list of athletes who have received a specific award across multiple seasons.

**Path Parameters:**
- `award_id` (required): The unique identifier for the award category

**Response Structure:**
- `count`: Total number of recipients
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award recipient objects

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/awards/1/recipients" | jq
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
      "athlete": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4395628?lang=en&region=us"
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/teams/258?lang=en&region=us"
      },
      "year": 2024,
      "seasonType": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/2?lang=en&region=us"
      }
    },
    {
      "athlete": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137?lang=en&region=us"
      },
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/teams/2305?lang=en&region=us"
      },
      "year": 2023,
      "seasonType": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2023/types/2?lang=en&region=us"
      }
    }
  ]
}
```

## Season Awards Endpoint

### `/seasons/{year}/awards`

Returns a list of awards given for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Response Structure:**
- `count`: Total number of award categories
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award references for the specified season

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/awards" | jq
```

## Team Awards Endpoint

### `/seasons/{year}/teams/{team_id}/awards`

Returns awards received by a specific team in a given season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Response Structure:**
- `count`: Total number of awards received by the team
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of award references received by the team

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/awards" | jq
```

## Common Award IDs

The following are common award IDs in NCAA Men's Basketball:

| ID | Award Name |
|----|------------|
| 1 | Player of the Year |
| 2 | Coach of the Year |
| 3 | All-American First Team |
| 4 | All-American Second Team |
| 5 | All-American Third Team |
| 6 | Freshman of the Year |
| 7 | Defensive Player of the Year |
| 8 | Most Improved Player |
| 9 | Sixth Man of the Year |
| 10 | All-Conference First Team |
| 11 | All-Conference Second Team |
| 12 | All-Conference Third Team |
| 13 | All-Conference Defensive Team |
| 14 | All-Conference Freshman Team |
| 15 | Conference Player of the Year |
| 16 | Conference Coach of the Year |
| 17 | Conference Defensive Player of the Year |
| 18 | Conference Freshman of the Year |
| 19 | Conference Sixth Man of the Year |

## Data Availability

Award data may not be available until officially announced for a given season. The league awards structure defines the categories, while the season-specific awards endpoints contain the actual winners for each year.

## Awards Data Structure

**League Awards Response:**
- `count`: Total number of available award categories
- `pageIndex`: Current page index (when paginated)
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of references to specific award categories

**Award Detail Response:**
- `id`: Unique identifier for the award
- `name`: Full name of the award (e.g., "Player of the Year")
- `displayName`: Display name used for UI presentation
- `shortDisplayName`: Shortened display name
- `description`: Description of the award criteria
- `recipients`: Reference to award recipients across seasons

**Award Recipients Response:**
- `count`: Total number of recipients
- `pageIndex`: Current page index
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of recipient objects
  - `athlete`: Reference to the athlete who received the award
  - `team`: Reference to the team the athlete represented
  - `year`: Season year when the award was given
  - `seasonType`: Season type (typically Regular Season)
