# Event Endpoints

This document provides detailed information about event-related endpoints in the ESPN API for NCAA Men's Basketball. Events represent games and related data.

## Base URL

```
https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
```

## Event Information

### Get Event Details

```
GET /events/{event_id}
```

Retrieves detailed information about a specific event (game).

**Path Parameters:**
- `event_id` (required): The unique identifier for the event.

**Response Fields:**
- `id`: Unique identifier for the event
- `uid`: Universal identifier with format "s:40~l:41~e:{event_id}"
- `date`: ISO 8601 formatted date and time of the event
- `name`: Event name (typically "{away_team} at {home_team}")
- `shortName`: Shortened event name
- `season`: Reference to season information
- `seasonType`: Reference to season type information
- `week`: Reference to week information (if applicable)
- `competitions`: Array of competition objects
  - `id`: Competition identifier
  - `date`: Competition date and time
  - `attendance`: Number of attendees
  - `venue`: Venue information
  - `broadcasts`: Broadcast information
  - `competitors`: Array of team information
  - `status`: Game status information
  - `notes`: Array of notes about the game
  - `headlines`: Game headlines
  - `leaders`: Statistical leaders for the game
  - `format`: Game format information
  - `startDate`: ISO 8601 formatted start date and time
  - `geoBroadcasts`: Geographic broadcast information
  - `headlines`: News headlines related to the game
  - `series`: Series information (if part of a series)
  - `tickets`: Ticket availability information
  - `odds`: Betting odds
  - `conference`: Conference information

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661" | jq
```

**Example Response:**
```json
{
  "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661?lang=en&region=us",
  "id": "401524661",
  "uid": "s:40~l:41~e:401524661",
  "date": "2024-01-10T00:00Z",
  "name": "Nebraska Cornhuskers at Purdue Boilermakers",
  "shortName": "NEB @ PUR",
  "season": {
    "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024?lang=en&region=us",
    "year": 2024,
    "type": 2,
    "slug": "regular-season"
  },
  "competitions": [
    {
      "id": "401524661",
      "uid": "s:40~l:41~e:401524661~c:401524661",
      "date": "2024-01-10T00:00Z",
      "attendance": 14876,
      "venue": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/venues/4103?lang=en&region=us",
        "id": "4103",
        "fullName": "Mackey Arena",
        "address": {
          "city": "West Lafayette",
          "state": "IN"
        },
        "capacity": 14804,
        "indoor": true
      },
      "broadcasts": [
        {
          "type": {
            "id": "1",
            "shortName": "TV"
          },
          "market": {
            "id": "1",
            "type": "National"
          },
          "media": {
            "shortName": "BTN"
          },
          "lang": "en",
          "region": "us"
        }
      ],
      "competitors": [
        {
          "id": "2509",
          "uid": "s:40~l:41~t:2509",
          "type": "team",
          "order": 0,
          "homeAway": "home",
          "winner": true,
          "team": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/2509?lang=en&region=us",
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
            "logo": "https://a.espncdn.com/i/teamlogos/ncaa/500/2509.png"
          },
          "score": "87",
          "records": [
            {
              "type": "total",
              "summary": "15-2",
              "displayValue": "15-2"
            },
            {
              "type": "home",
              "summary": "9-0",
              "displayValue": "9-0"
            },
            {
              "type": "road",
              "summary": "4-1",
              "displayValue": "4-1"
            }
          ],
          "statistics": []
        },
        {
          "id": "158",
          "uid": "s:40~l:41~t:158",
          "type": "team",
          "order": 1,
          "homeAway": "away",
          "winner": false,
          "team": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/158?lang=en&region=us",
            "id": "158",
            "uid": "s:40~l:41~t:158",
            "slug": "nebraska-cornhuskers",
            "location": "Nebraska",
            "name": "Cornhuskers",
            "abbreviation": "NEB",
            "displayName": "Nebraska Cornhuskers",
            "shortDisplayName": "Cornhuskers",
            "color": "E41C38",
            "alternateColor": "FFFFFF",
            "logo": "https://a.espncdn.com/i/teamlogos/ncaa/500/158.png"
          },
          "score": "80",
          "records": [
            {
              "type": "total",
              "summary": "13-4",
              "displayValue": "13-4"
            },
            {
              "type": "home",
              "summary": "12-1",
              "displayValue": "12-1"
            },
            {
              "type": "road",
              "summary": "1-3",
              "displayValue": "1-3"
            }
          ],
          "statistics": []
        }
      ],
      "notes": [],
      "status": {
        "type": {
          "id": "3",
          "name": "STATUS_FINAL",
          "description": "Final",
          "detail": "Final",
          "shortDetail": "Final"
        },
        "displayClock": "0:00",
        "period": 2,
        "clock": 0
      },
      "headlines": [
        {
          "description": "— Zach Edey scored 15 of his 33 points in the first half, Fletcher Loyer finished with 13 points and No. 1 Purdue held off Nebraska for an 87-80 victory Tuesday night.",
          "type": "Game Recap",
          "shortLinkText": "Zach Edey scored 15 of his 33 points in the first half, Fletcher Loyer finished with 13 points and No. 1 Purdue held off Nebraska"
        }
      ]
    }
  ],
  "links": [
    {
      "language": "en-US",
      "rel": ["summary", "desktop", "event"],
      "href": "https://www.espn.com/mens-college-basketball/game/_/gameId/401524661",
      "text": "Gamecast",
      "shortText": "Gamecast",
      "isExternal": false,
      "isPremium": false
    }
  ]
}
```

### Get Event Summary

```
GET /events/{event_id}/summary
```

Retrieves a summary of the event including scoring, play-by-play, and other key details.

**Path Parameters:**
- `event_id` (required): The unique identifier for the event.

**Response Fields:**
Includes various references to data structures that provide a summary of the event, such as:
- `plays`: Play-by-play data
- `boxscore`: Box score statistics
- `leaders`: Statistical leaders for the game
- `format`: Game format information
- `gameInfo`: General information about the game
- `scoring`: Scoring by period
- `standings`: Current team standings
- `videos`: Associated video content

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661/summary" | jq
```

## Event Box Score

### Get Event Box Score

```
GET /events/{event_id}/competitions/{competition_id}/competitors/{team_id}/statistics
```

Retrieves detailed box score statistics for a specific team in an event.

**Path Parameters:**
- `event_id` (required): The unique identifier for the event
- `competition_id` (required): The competition identifier (typically the same as the event_id)
- `team_id` (required): The identifier for the team

**Response Structure:**
Contains detailed statistics grouped by category (e.g., scoring, rebounding, etc.) for both teams or a specific team.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661/competitions/401524661/competitors/2509/statistics" | jq
```

## Play-by-Play Data

### Get Event Play-by-Play

```
GET /events/{event_id}/playbyplay
```

Retrieves detailed play-by-play information for a specific event.

**Path Parameters:**
- `event_id` (required): The unique identifier for the event.

**Query Parameters:**
- `limit` (optional): Number of plays to return per page
- `offset` (optional): Number of plays to skip

**Response Structure:**
- `count`: Total number of plays
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of play objects
  - `id`: Play identifier
  - `sequenceNumber`: Order of the play
  - `type`: Type of play
  - `text`: Description of the play
  - `scoreValue`: Points scored on the play
  - `team`: Reference to the team that performed the action
  - `period`: Period in which the play occurred
  - `clock`: Game clock at the time of the play
  - `participants`: Array of athletes involved in the play

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661/playbyplay?limit=10" | jq
```

**Example Response:**
```json
{
  "count": 304,
  "pageIndex": 1,
  "pageSize": 10,
  "pageCount": 31,
  "items": [
    {
      "id": "401524661:1",
      "type": {
        "id": "129",
        "text": "Tip Off",
        "abbreviation": "TIP"
      },
      "text": "Tipoff: Zach Edey vs. Rienk Mast (Brice Williams gains possession)",
      "period": {
        "number": 1
      },
      "clock": {
        "displayValue": "20:00"
      },
      "participants": [
        {
          "athlete": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4433137?lang=en&region=us"
          }
        },
        {
          "athlete": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4535675?lang=en&region=us"
          }
        },
        {
          "athlete": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4588720?lang=en&region=us"
          }
        }
      ],
      "wallclock": "2024-01-10T00:00:00Z",
      "teamId": 158,
      "sequenceNumber": "1"
    },
    {
      "id": "401524661:2",
      "type": {
        "id": "404",
        "text": "Missed Jump Shot",
        "abbreviation": "MISSJUMP"
      },
      "text": "Brice Williams missed Three Point Jumper",
      "scoreValue": 0,
      "team": {
        "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/158?lang=en&region=us"
      },
      "period": {
        "number": 1
      },
      "clock": {
        "displayValue": "19:36"
      },
      "participants": [
        {
          "athlete": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/athletes/4588720?lang=en&region=us"
          }
        }
      ],
      "wallclock": "2024-01-10T00:00:24Z",
      "teamId": 158,
      "sequenceNumber": "2"
    }
  ]
}
```

### Get Event Line Score

```
GET /events/{event_id}/competitions/{competition_id}/linescore
```

Retrieves the score by period for an event.

**Path Parameters:**
- `event_id` (required): The unique identifier for the event
- `competition_id` (required): The competition identifier (typically the same as the event_id)

**Response Structure:**
Contains score information broken down by period for each team.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/401524661/competitions/401524661/linescore" | jq
```

## Season Event Lists

### Get Season Events

```
GET /seasons/{year}/events
```

Retrieves a list of events for a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season

**Query Parameters:**
- `limit` (optional): Number of events to return per page
- `offset` (optional): Number of events to skip
- `dates` (optional): Date or date range in ISO format (e.g., 2024-01-10, 2024-01-10T00:00Z-2024-01-17T23:59Z)

**Response Structure:**
- `count`: Total number of events
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of event references

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/events?dates=20240110&limit=10" | jq
```

### Get Season Type Events

```
GET /seasons/{year}/types/{type}/events
```

Retrieves a list of events for a specific season type (regular season, postseason, etc.).

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID (1=Preseason, 2=Regular Season, 3=Postseason)

**Query Parameters:**
Same as the season events endpoint.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/2/events?limit=10" | jq
```

### Get Team Events

```
GET /seasons/{year}/teams/{team_id}/events
```

Retrieves a list of events for a specific team in a specific season.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `team_id` (required): The unique identifier for the team

**Query Parameters:**
Same as the season events endpoint.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/teams/2509/events?limit=10" | jq
```

## Weekly Schedule

### Get Week Events

```
GET /seasons/{year}/types/{type}/weeks/{week}/events
```

Retrieves a list of events for a specific week in a specific season type.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- `type` (required): The season type ID
- `week` (required): The week number or identifier

**Query Parameters:**
Same as the season events endpoint.

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/2/weeks/10/events?limit=10" | jq
```

## Event Groups

### Get League Calendar

```
GET /calendar
```

Retrieves a calendar of events grouped by date.

**Response Structure:**
- `count`: Total number of calendar entries
- `pageIndex`: Current page number
- `pageSize`: Number of items per page
- `pageCount`: Total number of pages
- `items`: Array of calendar entry objects that group events by date

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/calendar" | jq
```

### Get Conference Tournament Events

```
GET /seasons/{year}/types/3/groups/50/events
```

Retrieves events specific to conference tournaments during the postseason.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- Group 50 represents conference tournaments

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/3/groups/50/events?limit=10" | jq
```

### Get NCAA Tournament Events

```
GET /seasons/{year}/types/3/groups/100/events
```

Retrieves events specific to the NCAA tournament during the postseason.

**Path Parameters:**
- `year` (required): The four-digit year identifier for the season
- Group 100 represents the NCAA tournament

**Example Request:**
```bash
curl -s "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2024/types/3/groups/100/events?limit=10" | jq
``` 