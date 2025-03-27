"""
Raw team data model.

This module defines the data model for raw team data from the ESPN API.
"""

from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RawTeam(BaseModel):
    """
    Raw team data model representing data stored in the raw_teams table.

    This model stores the raw team data exactly as received from the API,
    along with metadata about the collection.

    Attributes:
        id: Unique identifier for the record
        team_id: Team identifier from the source system
        raw_data: Raw JSON response from the API
        source_url: URL or endpoint that provided the data
        collected_at: Timestamp when the data was collected
        processing_version: Version of the processing code
    """

    id: Optional[UUID] = Field(None, description="Unique identifier for the record")
    team_id: str = Field(..., description="Team identifier from the source system")
    raw_data: Dict = Field(..., description="Raw JSON response from the API")
    source_url: str = Field(..., description="URL or endpoint that provided the data")
    collected_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when the data was collected"
    )
    processing_version: str = Field(..., description="Version of the processing code")
