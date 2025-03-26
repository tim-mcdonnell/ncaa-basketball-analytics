from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class GameStatus(str, Enum):
    """Game status enum."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELED = "canceled"


class TeamScore(BaseModel):
    """Team with score in a game."""
    id: str = Field(..., description="Team ID")
    name: str = Field(..., description="Team name")
    score: str = Field(default="0", description="Team score")
    
    @validator('score')
    def score_must_be_numeric(cls, v):
        """Ensure score is numeric."""
        try:
            int(v)
            return v
        except ValueError:
            return "0"


class Game(BaseModel):
    """Game data model for ESPN API responses."""
    id: str = Field(..., description="Unique game identifier")
    date: str = Field(..., description="Game date and time (ISO format)")
    name: str = Field(..., description="Game name (e.g., 'Team A vs Team B')")
    short_name: Optional[str] = Field(default=None, description="Short game name")
    status: GameStatus = Field(default=GameStatus.SCHEDULED, description="Game status")
    teams: List[TeamScore] = Field(default_factory=list, description="Teams and scores")
    
    @validator('date')
    def validate_date_format(cls, v):
        """Validate date is in ISO format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except (ValueError, AttributeError):
            raise ValueError("Invalid date format, expected ISO format")


class GameList(BaseModel):
    """List of games."""
    games: List[Game] = Field(default_factory=list, description="List of games") 