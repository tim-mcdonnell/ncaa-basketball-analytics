# Common Query Parameters

Many ESPN API endpoints support common query parameters that allow you to filter, paginate, and customize responses. This document outlines the most frequently used query parameters across the API.

## Pagination Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `limit` | Integer | Number of items to return per page. Default is typically 25. Max value varies by endpoint. | `?limit=100` |
| `page` | Integer | Page number to retrieve (1-based indexing). Default is 1. | `?page=2` |

## Localization Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lang` | String | Language code for localized content. Default is 'en' (English). | `?lang=en` |
| `region` | String | Regional localization. Default is 'us' (United States). | `?region=us` |

## Time and Date Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dates` | String | Filter events by specific date(s) in YYYYMMDD format. | `?dates=20240301` |
| `dateRange` | String | Date range in format YYYYMMDD-YYYYMMDD. | `?dateRange=20240301-20240315` |

## Season Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `season` | Integer | Season year to retrieve data for. | `?season=2025` |
| `seasontype` | Integer | Filter by season type (1=Preseason, 2=Regular Season, 3=Postseason) | `?seasontype=2` |

## Team Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `teams` | Integer | Filter by team ID(s). Can be comma-separated for multiple teams. | `?teams=52,150` |
| `groups` | Integer | Filter by group (conference/division) ID(s). Can be comma-separated. | `?groups=1,5` |

## Filtering Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `active` | Boolean | Filter for active items only. | `?active=true` |
| `home` | Boolean | Filter for home teams. | `?home=true` |
| `away` | Boolean | Filter for away teams. | `?away=true` |

## Metadata Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `enable` | String | Enable specific features or data in the response. Comma-separated values. | `?enable=odds,logos` |
| `disable` | String | Disable specific features or data in the response. Comma-separated values. | `?disable=stats,records` |

## Sorting Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `sort` | String | Field to sort results by. | `?sort=name` |
| `order` | String | Sort order (asc or desc). Default varies by endpoint. | `?sort=name&order=asc` |

## Examples

### Fetching a Specific Page of Teams with Increased Limit

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams?limit=50&page=2
```

### Getting Events for a Specific Date Range

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events?dateRange=20250301-20250315
```

### Getting Regular Season Standings for a Conference

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/types/2/groups/1/standings
```

### Getting Team Schedule

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2025/teams/52/events?seasontype=2
```

## Parameter Combinations

Parameters can be combined to create more specific queries. For example:

```
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events?dates=20250301&groups=1&limit=100
```

This would retrieve up to 100 events on March 1, 2025, involving teams from group ID 1.

## Notes on Usage

- Not all parameters are supported by every endpoint. Consult the specific endpoint documentation for supported parameters.
- Some parameters may have different effects depending on the endpoint.
- Invalid parameter values typically result in a 400 Bad Request response.
- Parameter names are case-sensitive.
- Boolean parameters accept `true` or `false` values (lowercase). 