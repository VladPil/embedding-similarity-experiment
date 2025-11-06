"""
Base interfaces for similarity analysis system.
Following Strategy pattern and SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class SimilarityMethod(str, Enum):
    """Available similarity calculation methods."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SimilarityScope(str, Enum):
    """Scope of similarity comparison."""
    FULL_TEXT = "full_text"  # Compare entire texts
    CHAPTER = "chapter"  # Chapter-by-chapter
    CHUNK = "chunk"  # Chunk-level granular comparison
    THEMATIC = "thematic"  # Theme-based similarity
    CHARACTER = "character"  # Character-based similarity
    STYLE = "style"  # Writing style similarity


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SimilarityContext:
    """Context for similarity calculation."""
    text1: str
    text2: str
    embeddings1: Optional[List[Any]] = None
    embeddings2: Optional[List[Any]] = None
    chunks1: Optional[List[Any]] = None
    chunks2: Optional[List[Any]] = None
    metadata1: Optional[Dict] = None
    metadata2: Optional[Dict] = None


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""
    method: SimilarityMethod
    scope: SimilarityScope
    score: float  # 0.0 to 1.0
    confidence: float  # Confidence in the result
    details: Dict[str, Any]
    interpretation: str


# =============================================================================
# STRATEGY INTERFACE
# =============================================================================

class ISimilarityStrategy(ABC):
    """
    Interface for similarity calculation strategies.

    Each strategy implements a specific method of calculating similarity.
    Follows Strategy pattern and Interface Segregation Principle.
    """

    @abstractmethod
    def get_method(self) -> SimilarityMethod:
        """Get similarity method identifier."""
        pass

    @abstractmethod
    def get_scope(self) -> SimilarityScope:
        """Get similarity scope."""
        pass

    @abstractmethod
    async def calculate(
        self,
        context: SimilarityContext
    ) -> SimilarityResult:
        """
        Calculate similarity between two texts.

        Args:
            context: Similarity context with texts and embeddings

        Returns:
            Similarity result with score and details
        """
        pass

    @abstractmethod
    def requires_embeddings(self) -> bool:
        """Check if this strategy requires embeddings."""
        pass

    @abstractmethod
    def get_estimated_time(self) -> float:
        """
        Estimate execution time in seconds.

        Returns:
            Estimated time in seconds
        """
        pass

    @abstractmethod
    def interpret_score(self, score: float) -> str:
        """
        Interpret similarity score into human-readable text.

        Args:
            score: Similarity score (0.0 to 1.0)

        Returns:
            Human-readable interpretation
        """
        pass


# =============================================================================
# AGGREGATOR INTERFACE
# =============================================================================

class ISimilarityAggregator(ABC):
    """
    Interface for aggregating multiple similarity scores.

    Combines results from different strategies into final score.
    Follows Single Responsibility Principle.
    """

    @abstractmethod
    def aggregate(
        self,
        results: List[SimilarityResult],
        weights: Optional[Dict[SimilarityMethod, float]] = None
    ) -> Tuple[float, str]:
        """
        Aggregate multiple similarity results.

        Args:
            results: List of similarity results
            weights: Optional weights for each method

        Returns:
            Tuple of (final_score, interpretation)
        """
        pass


# =============================================================================
# FACTORY INTERFACE
# =============================================================================

class ISimilarityFactory(ABC):
    """
    Factory for creating similarity strategies.

    Follows Factory pattern and Dependency Inversion Principle.
    """

    @abstractmethod
    def create_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        **kwargs
    ) -> ISimilarityStrategy:
        """
        Create similarity strategy.

        Args:
            method: Similarity method
            scope: Similarity scope
            **kwargs: Strategy-specific parameters

        Returns:
            Configured strategy instance
        """
        pass

    @abstractmethod
    def register_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        strategy_class: type
    ) -> None:
        """Register new similarity strategy."""
        pass

    @abstractmethod
    def get_available_methods(self) -> List[SimilarityMethod]:
        """Get list of available similarity methods."""
        pass


# =============================================================================
# BUILDER INTERFACE
# =============================================================================

class ISimilarityBuilder(ABC):
    """
    Builder interface for configuring similarity analysis.

    Allows flexible configuration of multiple similarity strategies.
    Follows Builder pattern.
    """

    @abstractmethod
    def with_cosine_similarity(self) -> 'ISimilarityBuilder':
        """Add cosine similarity calculation."""
        pass

    @abstractmethod
    def with_semantic_similarity(self) -> 'ISimilarityBuilder':
        """Add semantic similarity calculation."""
        pass

    @abstractmethod
    def with_method(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope
    ) -> 'ISimilarityBuilder':
        """
        Add specific similarity method.

        Args:
            method: Similarity method
            scope: Similarity scope

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def with_aggregation(
        self,
        weights: Optional[Dict[SimilarityMethod, float]] = None
    ) -> 'ISimilarityBuilder':
        """
        Enable result aggregation with optional weights.

        Args:
            weights: Optional weights for each method

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def build(self) -> List[ISimilarityStrategy]:
        """Build and return configured strategies."""
        pass
