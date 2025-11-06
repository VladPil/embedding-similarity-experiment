"""
Similarity analysis module.
Provides flexible similarity calculation with multiple strategies.
"""

from server.core.similarity.base import (
    SimilarityMethod,
    SimilarityScope,
    SimilarityContext,
    SimilarityResult,
    ISimilarityStrategy,
    ISimilarityFactory,
    ISimilarityBuilder,
    ISimilarityAggregator
)

from server.core.similarity.factory import SimilarityFactory
from server.core.similarity.builder import SimilarityBuilder
from server.core.similarity.aggregator import WeightedAggregator, ConsensusAggregator

from server.core.similarity.strategies import (
    CosineSimilarityStrategy,
    SemanticSimilarityStrategy,
    HybridSimilarityStrategy
)

__all__ = [
    # Base classes and enums
    'SimilarityMethod',
    'SimilarityScope',
    'SimilarityContext',
    'SimilarityResult',
    'ISimilarityStrategy',
    'ISimilarityFactory',
    'ISimilarityBuilder',
    'ISimilarityAggregator',

    # Factory and Builder
    'SimilarityFactory',
    'SimilarityBuilder',

    # Aggregators
    'WeightedAggregator',
    'ConsensusAggregator',

    # Strategies
    'CosineSimilarityStrategy',
    'SemanticSimilarityStrategy',
    'HybridSimilarityStrategy',
]
