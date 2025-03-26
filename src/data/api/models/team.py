from typing import List, Optional
from pydantic import BaseModel, Field


class TeamRecord(BaseModel):
    """Team win-loss record."""
    summary: str = Field(default="0-0", description="Record summary (e.g., '10-5')")
    wins: int = Field(default=0, description="Number of wins")
    losses: int = Field(default=0, description="Number of losses")


class Team(BaseModel):
    """Team data model for ESPN API responses."""
    id: str = Field(..., description="Unique team identifier")
    name: str = Field(..., description="Team name")
    abbreviation: str = Field(default="", description="Team abbreviation")
    location: Optional[str] = Field(default=None, description="Team location/city")
    logo: Optional[str] = Field(default=None, description="URL to team logo")
    record: TeamRecord = Field(default_factory=TeamRecord, description="Team record")


class TeamList(BaseModel):
    """List of teams."""
    teams: List[Team] = Field(default_factory=list, description="List of teams") 