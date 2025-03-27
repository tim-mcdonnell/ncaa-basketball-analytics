from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class GameStatusType(BaseModel):
    """Game status type information."""

    completed: bool = False
    description: str = "Scheduled"
    state: Optional[str] = None
    detail: Optional[str] = None


class GameStatus(BaseModel):
    """Game status information."""

    type: GameStatusType = Field(default_factory=GameStatusType)


class TeamCompetitor(BaseModel):
    """Team competitor in a game."""

    id: str
    team: Dict[str, Any]
    homeAway: str
    score: Optional[str] = None


class Competition(BaseModel):
    """Game competition details."""

    id: str
    competitors: List[TeamCompetitor]
    status: GameStatus
    venue: Optional[Dict[str, Any]] = None


class GameResponse(BaseModel):
    """
    Pydantic model for ESPN game API response validation.

    Includes validation for handling inconsistent formats.
    """

    id: str
    date: str
    name: str
    competitions: List[Competition]

    @validator("competitions", each_item=True, pre=True)
    def validate_competitors(cls, v):
        """Validate competitors data."""
        if "competitors" in v:
            for comp in v["competitors"]:
                if "score" not in comp:
                    comp["score"] = None
        return v


class GameStatus(BaseModel):
    """Game status information."""

    is_completed: bool = Field(default=False, description="Whether the game is completed")
    is_in_progress: bool = Field(default=False, description="Whether the game is in progress")
    is_scheduled: bool = Field(default=True, description="Whether the game is scheduled")
    status_text: str = Field(default="", description="Status description (e.g., 'Final')")

    # For backward compatibility
    @property
    def completed(self) -> bool:
        """For backward compatibility."""
        return self.is_completed

    @property
    def description(self) -> str:
        """For backward compatibility."""
        return self.status_text


class TeamScore(BaseModel):
    """Team with score in a game."""

    team_id: str = Field(..., description="Team ID")
    team_name: str = Field(..., description="Team name")
    score: int = Field(default=0, description="Team score")
    is_home: bool = Field(default=False, description="Whether this is the home team")


class Game(BaseModel):
    """Game data model for ESPN API responses."""

    id: str = Field(..., description="Unique game identifier")
    date: datetime = Field(..., description="Game date and time")
    name: Optional[str] = Field(default=None, description="Game name/description")
    status: GameStatus = Field(default_factory=GameStatus, description="Game status")
    home_team: TeamScore = Field(..., description="Home team information")
    away_team: TeamScore = Field(..., description="Away team information")
    venue: Optional[str] = Field(default=None, description="Venue name")
    attendance: Optional[int] = Field(default=None, description="Attendance count")


class GameList(BaseModel):
    """List of games."""

    games: List[Game] = Field(default_factory=list, description="List of games")
