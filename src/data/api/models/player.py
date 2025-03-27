from typing import List, Optional
from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    """Player statistics data model."""

    points_per_game: float = Field(default=0.0, description="Points per game")
    rebounds_per_game: float = Field(default=0.0, description="Rebounds per game")
    assists_per_game: float = Field(default=0.0, description="Assists per game")


class Player(BaseModel):
    """Player data model for ESPN API responses."""

    id: str = Field(..., description="Unique player identifier")
    full_name: str = Field(..., description="Player full name")
    first_name: Optional[str] = Field(default=None, description="Player first name")
    last_name: Optional[str] = Field(default=None, description="Player last name")
    jersey: Optional[str] = Field(default=None, description="Jersey number")
    position: Optional[str] = Field(default=None, description="Player position")
    team_id: Optional[str] = Field(default=None, description="Team identifier")
    team_name: Optional[str] = Field(default=None, description="Team name")
    headshot: Optional[str] = Field(default=None, description="URL to player headshot")
    stats: Optional[PlayerStats] = Field(default=None, description="Player statistics")

    # For backward compatibility
    @property
    def name(self) -> str:
        """For backward compatibility."""
        return self.full_name


class PlayerList(BaseModel):
    """List of players."""

    players: List[Player] = Field(default_factory=list, description="List of players")
