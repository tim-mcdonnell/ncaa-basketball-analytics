"""
Dimensional team data model.

This module defines the data model for dimensional team data.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DimTeam(BaseModel):
    """
    Dimensional team data model representing data stored in the dim_teams table.

    This model represents the cleaned and transformed team data for analytical use.

    Attributes:
        team_id: Unique team identifier
        name: Team name
        conference: Conference name
        division: Division within conference
        logo_url: URL to team logo image
        mascot: Team mascot name
        primary_color: Primary team color (hex code)
        secondary_color: Secondary team color (hex code)
        venue_name: Home venue name
        venue_capacity: Home venue capacity
        city: Team's home city
        state: Team's home state
        created_at: Record creation timestamp
        updated_at: Record last update timestamp
    """

    team_id: str = Field(..., description="Unique team identifier")
    name: str = Field(..., description="Team name")
    conference: Optional[str] = Field(None, description="Conference name")
    division: Optional[str] = Field(None, description="Division within conference")
    logo_url: Optional[str] = Field(None, description="URL to team logo image")
    mascot: Optional[str] = Field(None, description="Team mascot name")
    primary_color: Optional[str] = Field(None, description="Primary team color (hex code)")
    secondary_color: Optional[str] = Field(None, description="Secondary team color (hex code)")
    venue_name: Optional[str] = Field(None, description="Home venue name")
    venue_capacity: Optional[int] = Field(None, description="Home venue capacity")
    city: Optional[str] = Field(None, description="Team's home city")
    state: Optional[str] = Field(None, description="Team's home state")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Record last update timestamp"
    )
