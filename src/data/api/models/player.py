from typing import List, Optional
from pydantic import BaseModel, Field


class Player(BaseModel):
    """Player data model for ESPN API responses."""
    id: str = Field(..., description="Unique player identifier")
    name: str = Field(..., description="Player full name")
    jersey: Optional[str] = Field(default=None, description="Jersey number")
    position: str = Field(default="Unknown", description="Player position")
    height: Optional[str] = Field(default=None, description="Player height")
    weight: Optional[str] = Field(default=None, description="Player weight")
    year: Optional[str] = Field(default=None, description="Academic year/class")


class PlayerList(BaseModel):
    """List of players."""
    players: List[Player] = Field(default_factory=list, description="List of players") 