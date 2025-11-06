"""
API schemas module.
"""

from server.api.schemas.analysis import (
    AnalysisTypeEnum,
    AnalysisPresetEnum,
    BookAnalysisRequest,
    BookAnalysisResponse,
    EstimateTimeRequest,
    TimeEstimateResponse,
    AvailableAnalysesResponse,
    AnalysisStatistics,
    GenreAnalysisResult,
    CharacterAnalysisResult,
    TensionAnalysisResult,
    PaceAnalysisResult,
    WaterAnalysisResult,
    ThemeAnalysisResult
)

from server.api.schemas.similarity import (
    SimilarityMethodEnum,
    SimilarityScopeEnum,
    SimilarityPresetEnum,
    SimilarityMethodConfig,
    SimilarityCalculationRequest,
    SimilarityCalculationResponse,
    AvailableSimilarityMethodsResponse,
    SimilarityStatistics,
    SimilarityMethodResult,
    BatchSimilarityRequest,
    BatchSimilarityResponse
)

__all__ = [
    # Analysis schemas
    'AnalysisTypeEnum',
    'AnalysisPresetEnum',
    'BookAnalysisRequest',
    'BookAnalysisResponse',
    'EstimateTimeRequest',
    'TimeEstimateResponse',
    'AvailableAnalysesResponse',
    'AnalysisStatistics',
    'GenreAnalysisResult',
    'CharacterAnalysisResult',
    'TensionAnalysisResult',
    'PaceAnalysisResult',
    'WaterAnalysisResult',
    'ThemeAnalysisResult',

    # Similarity schemas
    'SimilarityMethodEnum',
    'SimilarityScopeEnum',
    'SimilarityPresetEnum',
    'SimilarityMethodConfig',
    'SimilarityCalculationRequest',
    'SimilarityCalculationResponse',
    'AvailableSimilarityMethodsResponse',
    'SimilarityStatistics',
    'SimilarityMethodResult',
    'BatchSimilarityRequest',
    'BatchSimilarityResponse',
]
