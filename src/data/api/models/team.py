from typing import List, Optional
from pydantic import BaseModel, Field, validator


class TeamResponse(BaseModel):
    """
    Pydantic model for ESPN team API response validation.

    Includes validation for handling inconsistent record formats.
    """

    id: str
    name: str
    abbreviation: Optional[str] = ""
    location: Optional[str] = ""
    logo: Optional[str] = None
    record: Optional[str] = "0-0"

    @validator("record", pre=True)
    def extract_record_summary(cls, v):
        """Extract record summary from nested structure."""
        if v is None:
            return "0-0"
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and "items" in v and v["items"]:
            return v["items"][0].get("summary", "0-0")
        return "0-0"


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
