"""
Pydantic schemas for similarity analysis API.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class SimilarityMethodEnum(str, Enum):
    """Available similarity methods."""
    COSINE = "cosine"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SimilarityScopeEnum(str, Enum):
    """Similarity scopes."""
    FULL_TEXT = "full_text"
    CHUNK = "chunk"
    THEMATIC = "thematic"


class SimilarityPresetEnum(str, Enum):
    """Similarity presets."""
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class SimilarityMethodConfig(BaseModel):
    """Configuration for a similarity method."""

    method: SimilarityMethodEnum
    scope: SimilarityScopeEnum

    class Config:
        json_schema_extra = {
            "example": {
                "method": "cosine",
                "scope": "full_text"
            }
        }


class SimilarityCalculationRequest(BaseModel):
    """Request for similarity calculation."""

    text1: str = Field(..., min_length=50, description="First text")
    text2: str = Field(..., min_length=50, description="Second text")

    selected_methods: Optional[List[SimilarityMethodConfig]] = Field(
        None,
        description="List of methods to use (null = default)"
    )

    preset: Optional[SimilarityPresetEnum] = Field(
        None,
        description="Use preset configuration (overrides selected_methods)"
    )

    aggregate: bool = Field(
        True,
        description="Whether to aggregate results into final score"
    )

    chunk_size: int = Field(
        1000,
        ge=100,
        le=5000,
        description="Chunk size for text splitting"
    )

    metadata1: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata for text1"
    )

    metadata2: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata for text2"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text1": "The quick brown fox jumps over the lazy dog...",
                "text2": "A fast auburn fox leaps above the sleepy canine...",
                "preset": "comprehensive",
                "aggregate": True,
                "chunk_size": 1000
            }
        }


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class SimilarityMethodResult(BaseModel):
    """Result from a single similarity method."""

    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any]


class SimilarityStatistics(BaseModel):
    """Statistics for similarity calculation."""

    total_methods: int
    average_score: float
    min_score: float
    max_score: float
    average_confidence: float
    total_execution_time: float
    text1_length: int
    text2_length: int


class SimilarityCalculationResponse(BaseModel):
    """Response for similarity calculation."""

    success: bool
    results: Dict[str, SimilarityMethodResult]
    interpretations: Dict[str, str]
    statistics: SimilarityStatistics
    final_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    final_interpretation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": {
                    "cosine_full_text": {
                        "score": 0.85,
                        "confidence": 0.9,
                        "details": {"method": "average_embeddings"}
                    },
                    "semantic_thematic": {
                        "score": 0.78,
                        "confidence": 0.85,
                        "details": {"topics_text1": 10, "topics_text2": 10}
                    }
                },
                "interpretations": {
                    "cosine_full_text": "Высокая схожесть - тексты значительно похожи",
                    "semantic_thematic": "Высокая семантическая схожесть - тексты имеют много общих тем"
                },
                "statistics": {
                    "total_methods": 2,
                    "average_score": 0.815,
                    "min_score": 0.78,
                    "max_score": 0.85,
                    "average_confidence": 0.875,
                    "total_execution_time": 6.5,
                    "text1_length": 50000,
                    "text2_length": 48000
                },
                "final_score": 0.82,
                "final_interpretation": "Высокая общая схожесть (агрегированная оценка)"
            }
        }


class AvailableSimilarityMethodsResponse(BaseModel):
    """Response with available similarity methods."""

    methods: List[str]
    scopes: Dict[str, List[str]]
    presets: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "methods": ["cosine", "semantic", "hybrid"],
                "scopes": {
                    "cosine": ["full_text", "chunk"],
                    "semantic": ["thematic"],
                    "hybrid": ["full_text", "chunk"]
                },
                "presets": ["fast", "comprehensive", "semantic", "hybrid"]
            }
        }


# =============================================================================
# BATCH PROCESSING SCHEMAS
# =============================================================================

class BatchSimilarityItem(BaseModel):
    """Single item in batch similarity calculation."""

    id: str = Field(..., description="Unique identifier for this comparison")
    text1: str = Field(..., min_length=50)
    text2: str = Field(..., min_length=50)
    metadata1: Optional[Dict[str, Any]] = None
    metadata2: Optional[Dict[str, Any]] = None


class BatchSimilarityRequest(BaseModel):
    """Request for batch similarity calculation."""

    items: List[BatchSimilarityItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text pairs to compare"
    )

    preset: SimilarityPresetEnum = Field(
        SimilarityPresetEnum.FAST,
        description="Preset to use for all comparisons"
    )

    aggregate: bool = Field(True)

    @validator('items')
    def validate_items_count(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 items per batch request")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "comp_1",
                        "text1": "First text...",
                        "text2": "Second text..."
                    },
                    {
                        "id": "comp_2",
                        "text1": "Third text...",
                        "text2": "Fourth text..."
                    }
                ],
                "preset": "fast",
                "aggregate": True
            }
        }


class BatchSimilarityItemResult(BaseModel):
    """Result for single batch item."""

    id: str
    success: bool
    final_score: Optional[float] = None
    error: Optional[str] = None


class BatchSimilarityResponse(BaseModel):
    """Response for batch similarity calculation."""

    success: bool
    results: List[BatchSimilarityItemResult]
    total_items: int
    successful_items: int
    failed_items: int
    total_execution_time: float

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {"id": "comp_1", "success": True, "final_score": 0.85},
                    {"id": "comp_2", "success": True, "final_score": 0.42}
                ],
                "total_items": 2,
                "successful_items": 2,
                "failed_items": 0,
                "total_execution_time": 12.5
            }
        }
