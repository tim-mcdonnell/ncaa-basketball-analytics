---
title: ESPN API Integration Refactoring
description: Refactoring the ESPN API integration component to improve code organization and reliability
---

# ESPN API Integration Refactoring

## 🎯 Overview

**Background:** The initial ESPN API integration has been implemented, but there are code organization issues and opportunities to improve resilience and test coverage based on the specifications in the improvement document.

**Objective:** Refactor the ESPN API client implementation to resolve duplication, improve error handling, and enhance test coverage.

**Scope:** This task focuses on refactoring the existing ESPN API integration code without introducing new major features. It includes resolving the duplicate client implementation, improving error recovery mechanisms, enhancing test coverage, and completing missing functionality.

## 📐 Technical Requirements

### Client Implementation Refactoring

```
src/data/api/
├── __init__.py          # Export clean public API
├── async_client.py      # Base async client (keep existing)
├── rate_limiter.py      # Rate limiter (keep existing)
├── espn_client/         # ESPN-specific implementation
│   ├── __init__.py      # Export AsyncESPNClient, ESPNClient
│   ├── client.py        # Client implementation
│   ├── config.py        # Configuration handling
│   ├── teams.py         # Team-specific client methods
│   ├── games.py         # Game-specific client methods
│   └── players.py       # Player-specific client methods
└── espn_client.py       # File to be removed after refactoring
```

### Enhanced Error Recovery Implementation

```python
async def get_with_enhanced_recovery(self, endpoint, params=None, max_recovery_attempts=2):
    """
    Make GET request with enhanced recovery for intermittent API issues.

    Args:
        endpoint: API endpoint
        params: Request parameters
        max_recovery_attempts: Maximum recovery attempts for intermittent issues

    Returns:
        Response data
    """
    recovery_attempt = 0
    last_exception = None

    # Regular retry mechanism will handle standard retries
    # This adds an additional recovery layer for intermittent issues
    while recovery_attempt <= max_recovery_attempts:
        try:
            return await self.get(endpoint, params)
        except (ServiceUnavailableError, ConnectionResetError) as e:
            last_exception = e
            recovery_attempt += 1

            # Use longer waits for recovery attempts
            backoff_time = min(30, 5 * 2 ** recovery_attempt)
            logger.warning(
                f"Recovery attempt {recovery_attempt}/{max_recovery_attempts} "
                f"for {endpoint}. Waiting {backoff_time}s before retry."
            )
            await asyncio.sleep(backoff_time)

    # If we exhausted recovery attempts, raise the last exception
    logger.error(f"Enhanced recovery failed after {max_recovery_attempts} attempts")
    raise last_exception
```

### Expanded Player Stats Endpoint

The player stats endpoint is currently a placeholder. It should be implemented to fetch and return actual player statistics from the ESPN API.

```python
async def get_player_stats(
    self, player_id: str, season: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get statistics for a specific player.

    Args:
        player_id: Player ID
        season: Optional season year (e.g., "2023-24")

    Returns:
        Player statistics data

    Raises:
        ResourceNotFoundError: If player not found
        APIError: If API request fails
    """
    path = f"/athletes/{player_id}/statistics"
    params = {}
    if season:
        params["season"] = season

    return await self.get(path, params)
```

### Enhanced Model Validation

Add improved validation to the model classes to handle ESPN API inconsistencies:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class TeamResponse(BaseModel):
    """Pydantic model for ESPN team API response validation."""
    id: str
    name: str
    abbreviation: Optional[str] = ""
    location: Optional[str] = ""
    logo: Optional[str] = None
    record: Optional[Dict[str, Any]] = None

    @validator('record', pre=True)
    def extract_record_summary(cls, v):
        """Extract record summary from nested structure."""
        if isinstance(v, dict) and 'items' in v and v['items']:
            return v['items'][0].get('summary', '0-0')
        return v or '0-0'
```

## 🧪 Testing Requirements

### Test-Driven Development Process

1. **RED Phase**:
   - Write failing tests for refactored client organization
   - Create tests for enhanced error recovery
   - Implement tests for player stats endpoint
   - Add tests for model validations with inconsistent data

2. **GREEN Phase**:
   - Refactor client implementation to pass organization tests
   - Implement enhanced error recovery to make tests pass
   - Complete player stats endpoint implementation
   - Enhance model validation to handle inconsistent data

3. **REFACTOR Phase**:
   - Ensure clean separation of concerns in the client implementation
   - Optimize error recovery for different types of errors
   - Ensure code reuse across endpoints where applicable

### Test Cases

- [ ] Test `test_client_initialization_with_config`: Verify client initializes correctly with config
- [ ] Test `test_enhanced_error_recovery`: Test the enhanced recovery mechanism
- [ ] Test `test_service_unavailable_recovery`: Test recovery from 503 errors
- [ ] Test `test_connection_reset_recovery`: Test recovery from connection reset errors
- [ ] Test `test_get_player_stats`: Verify player stats retrieval works correctly
- [ ] Test `test_team_model_inconsistent_data`: Test team model with various inconsistent API response formats
- [ ] Test `test_game_model_inconsistent_data`: Test game model with various inconsistent API response formats
- [ ] Test `test_player_model_inconsistent_data`: Test player model with various inconsistent API response formats

### Integration Testing

- [ ] Test end-to-end flow from client to models with enhanced error recovery
- [ ] Test using actual API responses to verify parsing and validation correctness

## 🛠️ Implementation Process

1. **Prepare Directory Structure**
   - Create or update the modular structure for the ESPN client

2. **Refactor Client Implementation**
   - Move team-specific methods to teams.py
   - Move game-specific methods to games.py
   - Move player-specific methods to players.py
   - Update client.py to import and expose these methods

3. **Implement Enhanced Error Recovery**
   - Add the enhanced recovery mechanism to the client
   - Ensure proper logging and backoff for different error types

4. **Complete Player Stats Implementation**
   - Implement the player stats endpoint
   - Add proper error handling and validation

5. **Enhance Model Validation**
   - Add validators to handle inconsistent data
   - Ensure all edge cases are handled appropriately

6. **Test Implementation**
   - Run all tests to verify functionality
   - Add any missing test cases discovered during implementation

7. **Clean Up Legacy Implementation**
   - Once all tests pass with the new implementation, safely remove the duplicate code
   - Update any imports in dependent modules

## ✅ Acceptance Criteria

- [ ] All existing tests continue to pass
- [ ] New tests for refactored components pass
- [ ] Client implementation is properly organized into modules
- [ ] No duplicate implementation of AsyncESPNClient exists
- [ ] Enhanced error recovery successfully handles intermittent API issues
- [ ] Player stats endpoint is properly implemented
- [ ] Model validation correctly handles inconsistent API response formats
- [ ] Imports in dependent modules are updated to use the refactored client
- [ ] Code meets project quality standards (passes linting and typing)
