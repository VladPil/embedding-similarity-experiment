"""
Base interfaces and abstractions for book analysis system.
Following SOLID principles and design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class AnalysisType(str, Enum):
    """Available analysis types."""
    GENRE = "genre"
    CHARACTER = "character"
    TENSION = "tension"
    PACE = "pace"
    WATER = "water"
    THEME = "theme"
    STYLE = "style"
    AUDIENCE = "audience"
    WORLDBUILDING = "worldbuilding"
    RELATIONSHIP = "relationship"
    EVENT = "event"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnalysisContext:
    """Context for analysis execution."""
    text: str
    chunks: List[Any]
    embeddings: Optional[List[Any]] = None
    chunk_indices: Optional[Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """Result of a single analysis."""
    analysis_type: AnalysisType
    data: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None


# =============================================================================
# STRATEGY INTERFACE
# =============================================================================

class IAnalysisStrategy(ABC):
    """
    Interface for analysis strategies.

    Each strategy implements a specific type of analysis.
    Follows Strategy pattern and Interface Segregation Principle (ISP).
    """

    @abstractmethod
    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        pass

    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute analysis.

        Args:
            context: Analysis context with text, chunks, etc.

        Returns:
            Analysis results as dict
        """
        pass

    @abstractmethod
    def requires_llm(self) -> bool:
        """Check if this analysis requires LLM."""
        pass

    @abstractmethod
    def requires_embeddings(self) -> bool:
        """Check if this analysis requires embeddings."""
        pass

    @abstractmethod
    def get_estimated_time(self, chunk_count: int) -> float:
        """
        Estimate execution time in seconds.

        Args:
            chunk_count: Number of chunks to process

        Returns:
            Estimated time in seconds
        """
        pass

    @abstractmethod
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret analysis results into human-readable text for UI display.

        Args:
            results: Analysis results from analyze() method

        Returns:
            Human-readable interpretation text
        """
        pass


# =============================================================================
# INDEXER INTERFACE
# =============================================================================

class IChunkIndexer(ABC):
    """
    Interface for chunk indexing.

    Responsible for finding relevant chunks for selective analysis.
    Follows Single Responsibility Principle (SRP).
    """

    @abstractmethod
    def build_index(
        self,
        chunks: List[Any],
        criterion: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build index for specific criterion.

        Args:
            chunks: List of text chunks
            criterion: Indexing criterion (e.g., 'characters', 'tension')
            **kwargs: Additional parameters

        Returns:
            Index data
        """
        pass

    @abstractmethod
    def get_relevant_chunks(
        self,
        index: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> List[int]:
        """
        Get indices of relevant chunks.

        Args:
            index: Index data
            threshold: Optional score threshold

        Returns:
            List of chunk indices
        """
        pass


# =============================================================================
# ANALYZER INTERFACE
# =============================================================================

class IAnalyzer(ABC):
    """
    High-level analyzer interface.

    Orchestrates multiple strategies.
    Follows Open/Closed Principle (OCP) - open for extension, closed for modification.
    """

    @abstractmethod
    async def analyze(
        self,
        context: AnalysisContext,
        strategies: List[IAnalysisStrategy]
    ) -> List[AnalysisResult]:
        """
        Execute multiple analysis strategies.

        Args:
            context: Analysis context
            strategies: List of strategies to execute

        Returns:
            List of analysis results
        """
        pass

    @abstractmethod
    def add_strategy(self, strategy: IAnalysisStrategy) -> None:
        """Add new analysis strategy."""
        pass

    @abstractmethod
    def remove_strategy(self, analysis_type: AnalysisType) -> None:
        """Remove analysis strategy."""
        pass


# =============================================================================
# BUILDER INTERFACE
# =============================================================================

class IAnalysisBuilder(ABC):
    """
    Builder interface for configuring analysis pipeline.

    Follows Builder pattern for flexible configuration.
    """

    @abstractmethod
    def with_genre_analysis(self) -> 'IAnalysisBuilder':
        """Add genre analysis."""
        pass

    @abstractmethod
    def with_character_analysis(self) -> 'IAnalysisBuilder':
        """Add character analysis."""
        pass

    @abstractmethod
    def with_tension_analysis(self) -> 'IAnalysisBuilder':
        """Add tension analysis."""
        pass

    @abstractmethod
    def with_pace_analysis(self) -> 'IAnalysisBuilder':
        """Add pace analysis."""
        pass

    @abstractmethod
    def with_water_analysis(self) -> 'IAnalysisBuilder':
        """Add water level analysis."""
        pass

    @abstractmethod
    def with_theme_analysis(self) -> 'IAnalysisBuilder':
        """Add theme analysis."""
        pass

    @abstractmethod
    def with_all_analyses(self) -> 'IAnalysisBuilder':
        """Add all available analyses."""
        pass

    @abstractmethod
    def build(self) -> List[IAnalysisStrategy]:
        """Build and return configured strategies."""
        pass


# =============================================================================
# FACTORY INTERFACE
# =============================================================================

class IStrategyFactory(ABC):
    """
    Factory for creating analysis strategies.

    Follows Factory pattern and Dependency Inversion Principle (DIP).
    """

    @abstractmethod
    def create_strategy(
        self,
        analysis_type: AnalysisType,
        **kwargs
    ) -> IAnalysisStrategy:
        """
        Create analysis strategy by type.

        Args:
            analysis_type: Type of analysis
            **kwargs: Strategy-specific parameters

        Returns:
            Configured strategy instance
        """
        pass

    @abstractmethod
    def register_strategy(
        self,
        analysis_type: AnalysisType,
        strategy_class: type
    ) -> None:
        """Register new strategy type."""
        pass
