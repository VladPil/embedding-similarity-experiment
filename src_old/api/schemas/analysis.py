"""
Pydantic schemas for book analysis API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class AnalysisTypeEnum(str, Enum):
    """Available analysis types."""
    GENRE = "genre"
    CHARACTER = "character"
    TENSION = "tension"
    PACE = "pace"
    WATER = "water"
    THEME = "theme"


class AnalysisPresetEnum(str, Enum):
    """Analysis presets."""
    ALL = "all"
    FAST = "fast"
    ESSENTIAL = "essential"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class BookAnalysisRequest(BaseModel):
    """Request for book analysis."""

    text: str = Field(..., min_length=100, description="Book text to analyze")

    selected_analyses: Optional[List[AnalysisTypeEnum]] = Field(
        None,
        description="List of analysis types to run (null = all)"
    )

    preset: Optional[AnalysisPresetEnum] = Field(
        None,
        description="Use preset configuration (overrides selected_analyses)"
    )

    chunk_size: int = Field(
        1000,
        ge=100,
        le=5000,
        description="Chunk size for text splitting"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Once upon a time...",
                "selected_analyses": ["genre", "character", "pace"],
                "chunk_size": 1000
            }
        }


class EstimateTimeRequest(BaseModel):
    """Request for time estimation."""

    text_length: int = Field(..., ge=0, description="Text length in characters")
    chunk_size: int = Field(1000, ge=100, le=5000)
    selected_analyses: Optional[List[AnalysisTypeEnum]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text_length": 400000,
                "chunk_size": 1000,
                "selected_analyses": ["genre", "character"]
            }
        }


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class AnalysisStatistics(BaseModel):
    """Statistics for analysis execution."""

    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    total_execution_time: float
    chunks_analyzed: int
    text_length: int


class GenreAnalysisResult(BaseModel):
    """Genre analysis result."""

    main_genre: str
    sub_genres: List[str]
    confidence: float
    reasoning: Optional[str] = None
    interpretation: Optional[str] = None


class CharacterInfo(BaseModel):
    """Character information."""

    name: str
    role: str
    traits: List[Dict[str, Any]]
    appearances: List[Dict[str, Any]]
    first_appearance: float


class CharacterAnalysisResult(BaseModel):
    """Character analysis result."""

    characters: List[CharacterInfo]
    total_characters: int
    chunks_analyzed: int
    coverage: float
    interpretation: Optional[str] = None


class TensionPoint(BaseModel):
    """Tension timeline point."""

    position: float
    score: float
    source: str
    description: str
    excerpt: Optional[str] = None


class TensionAnalysisResult(BaseModel):
    """Tension analysis result."""

    average_tension: float
    timeline: List[TensionPoint]
    peaks: List[float]
    peak_count: int
    interpretation: Optional[str] = None


class PaceAnalysisResult(BaseModel):
    """Pace analysis result."""

    overall_pace: str
    pace_score: float
    timeline: List[Dict[str, Any]]
    statistics: Dict[str, float]
    interpretation: Optional[str] = None


class WaterAnalysisResult(BaseModel):
    """Water level analysis result."""

    water_percentage: float
    info_density: float
    rating: str
    verbose_chunks: List[int]
    interpretation: Optional[str] = None


class ThemeInfo(BaseModel):
    """Theme information."""

    name: str
    weight: float
    confidence: float
    frequency: int
    examples: List[Dict[str, Any]]


class ThemeAnalysisResult(BaseModel):
    """Theme analysis result."""

    themes: List[ThemeInfo]
    total_themes: int
    interpretation: Optional[str] = None


class BookAnalysisResponse(BaseModel):
    """Response for book analysis."""

    success: bool
    results: Dict[str, Any]
    interpretations: Dict[str, str]
    statistics: AnalysisStatistics
    metadata: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": {
                    "genre": {
                        "main_genre": "fantasy",
                        "sub_genres": ["epic", "adventure"],
                        "confidence": 0.95
                    }
                },
                "interpretations": {
                    "genre": "Высокая уверенность в классификации жанра как фэнтези..."
                },
                "statistics": {
                    "total_analyses": 3,
                    "successful_analyses": 3,
                    "failed_analyses": 0,
                    "total_execution_time": 45.2,
                    "chunks_analyzed": 400,
                    "text_length": 400000
                }
            }
        }


class TimeEstimateResponse(BaseModel):
    """Response for time estimation."""

    estimates: Dict[str, float]
    total_time: float

    class Config:
        json_schema_extra = {
            "example": {
                "estimates": {
                    "genre": 5.0,
                    "character": 30.0,
                    "total": 35.0
                },
                "total_time": 35.0
            }
        }


class AvailableAnalysesResponse(BaseModel):
    """Response with available analyses."""

    analyses: List[str]
    presets: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "analyses": ["genre", "character", "tension", "pace", "water", "theme"],
                "presets": ["all", "fast", "essential"]
            }
        }
