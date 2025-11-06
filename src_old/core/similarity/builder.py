"""
Similarity Builder - fluent interface for configuring similarity analysis.
Implements Builder pattern for flexible similarity pipeline configuration.
"""

from typing import List, Optional, Dict
from loguru import logger

from server.core.similarity.base import (
    ISimilarityBuilder,
    ISimilarityStrategy,
    SimilarityMethod,
    SimilarityScope
)
from server.core.similarity.factory import SimilarityFactory
from server.core.similarity.aggregator import WeightedAggregator, ConsensusAggregator


class SimilarityBuilder(ISimilarityBuilder):
    """
    Fluent builder for configuring similarity analysis pipeline.

    Usage:
        builder = SimilarityBuilder()
        strategies = (builder
            .with_cosine_similarity()
            .with_semantic_similarity()
            .with_aggregation()
            .build())

    Or:
        strategies = SimilarityBuilder().with_all_methods().build()
    """

    def __init__(self, factory: Optional[SimilarityFactory] = None):
        """
        Initialize builder.

        Args:
            factory: Similarity factory (creates default if not provided)
        """
        self.factory = factory or SimilarityFactory()
        self._strategies: List[ISimilarityStrategy] = []
        self._selected_methods = set()
        self._aggregator = None
        self._aggregation_weights = None

    def with_cosine_similarity(
        self,
        scope: SimilarityScope = SimilarityScope.FULL_TEXT
    ) -> 'SimilarityBuilder':
        """
        Add cosine similarity.

        Args:
            scope: Similarity scope (default: FULL_TEXT)
        """
        return self._add_strategy(SimilarityMethod.COSINE, scope)

    def with_semantic_similarity(
        self,
        n_clusters: int = 10
    ) -> 'SimilarityBuilder':
        """
        Add semantic similarity.

        Args:
            n_clusters: Number of clusters for topic extraction
        """
        return self._add_strategy(
            SimilarityMethod.SEMANTIC,
            SimilarityScope.THEMATIC,
            n_clusters=n_clusters
        )

    def with_hybrid_similarity(
        self,
        scope: SimilarityScope = SimilarityScope.FULL_TEXT,
        weights: Optional[Dict[str, float]] = None
    ) -> 'SimilarityBuilder':
        """
        Add hybrid similarity.

        Args:
            scope: Similarity scope
            weights: Optional custom weights for components
        """
        return self._add_strategy(
            SimilarityMethod.HYBRID,
            scope,
            weights=weights
        )

    def with_method(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        **kwargs
    ) -> 'SimilarityBuilder':
        """
        Add specific similarity method.

        Args:
            method: Similarity method
            scope: Similarity scope
            **kwargs: Method-specific parameters

        Returns:
            Self for chaining
        """
        return self._add_strategy(method, scope, **kwargs)

    def with_all_methods(
        self,
        scope: SimilarityScope = SimilarityScope.FULL_TEXT
    ) -> 'SimilarityBuilder':
        """
        Add all available similarity methods.

        Args:
            scope: Similarity scope for all methods
        """
        self.with_cosine_similarity(scope)
        self.with_semantic_similarity()

        # Hybrid is redundant if we have all methods
        # self.with_hybrid_similarity(scope)

        return self

    def with_fast_methods(self) -> 'SimilarityBuilder':
        """
        Add only fast similarity methods (no clustering).

        Fast methods: Cosine
        """
        return self.with_cosine_similarity()

    def with_comprehensive_analysis(self) -> 'SimilarityBuilder':
        """
        Add comprehensive similarity analysis.

        Includes:
        - Cosine (full text)
        - Cosine (chunk level)
        - Semantic (thematic)
        - Hybrid
        """
        return (self
            .with_cosine_similarity(SimilarityScope.FULL_TEXT)
            .with_cosine_similarity(SimilarityScope.CHUNK)
            .with_semantic_similarity()
            .with_hybrid_similarity())

    def with_aggregation(
        self,
        weights: Optional[Dict[SimilarityMethod, float]] = None,
        method: str = "weighted"
    ) -> 'SimilarityBuilder':
        """
        Enable result aggregation.

        Args:
            weights: Optional weights for each method
            method: Aggregation method ('weighted' or 'consensus')

        Returns:
            Self for chaining
        """
        if method == "weighted":
            self._aggregator = WeightedAggregator(confidence_weighting=True)
        elif method == "consensus":
            self._aggregator = ConsensusAggregator(method="median")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        self._aggregation_weights = weights
        return self

    def build(self) -> List[ISimilarityStrategy]:
        """
        Build and return configured strategies.

        Returns:
            List of configured strategy instances
        """
        if not self._strategies:
            logger.warning("No strategies configured, returning empty list")

        logger.info(f"Built {len(self._strategies)} similarity strategies")

        return self._strategies.copy()

    def build_with_aggregator(self) -> tuple[List[ISimilarityStrategy], Optional[object]]:
        """
        Build and return strategies with aggregator.

        Returns:
            Tuple of (strategies, aggregator)
        """
        return self._strategies.copy(), self._aggregator

    def reset(self) -> 'SimilarityBuilder':
        """
        Reset builder to empty state.

        Returns:
            Self for chaining
        """
        self._strategies = []
        self._selected_methods = set()
        self._aggregator = None
        self._aggregation_weights = None
        return self

    def _add_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        **kwargs
    ) -> 'SimilarityBuilder':
        """
        Internal method to add strategy.

        Args:
            method: Similarity method
            scope: Similarity scope
            **kwargs: Strategy-specific parameters

        Returns:
            Self for chaining
        """
        # Create unique key
        key = (method, scope)

        # Prevent duplicates
        if key in self._selected_methods:
            logger.debug(f"Strategy {method.value} ({scope.value}) already added, skipping")
            return self

        try:
            strategy = self.factory.create_strategy(method, scope, **kwargs)
            self._strategies.append(strategy)
            self._selected_methods.add(key)

            logger.debug(f"Added strategy: {method.value} ({scope.value})")

        except Exception as e:
            logger.error(f"Failed to create strategy {method.value}: {e}")

        return self

    def get_estimated_time(self) -> float:
        """
        Get estimated total execution time.

        Returns:
            Estimated time in seconds
        """
        total_time = 0.0

        for strategy in self._strategies:
            total_time += strategy.get_estimated_time()

        # Add 10% overhead for orchestration
        total_time *= 1.1

        return total_time

    def summary(self) -> str:
        """
        Get summary of configured similarity methods.

        Returns:
            Human-readable summary
        """
        if not self._strategies:
            return "No similarity methods configured"

        lines = [f"Configured {len(self._strategies)} similarity methods:"]

        for i, strategy in enumerate(self._strategies, 1):
            method = strategy.get_method()
            scope = strategy.get_scope()
            requires_emb = "📊 Embeddings" if strategy.requires_embeddings() else "⚡ Fast"
            estimated_time = strategy.get_estimated_time()

            lines.append(
                f"  {i}. {method.value} ({scope.value}) - {requires_emb} (~{estimated_time:.1f}s)"
            )

        if self._aggregator:
            lines.append(f"\nAggregation: Enabled")

        total_time = self.get_estimated_time()
        lines.append(f"\nEstimated total time: ~{total_time:.1f}s")

        return "\n".join(lines)
