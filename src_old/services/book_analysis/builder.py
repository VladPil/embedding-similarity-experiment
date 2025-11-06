"""
Analysis Builder - fluent interface for configuring book analysis.
Implements Builder pattern for flexible analysis pipeline configuration.
"""

from typing import List, Optional
from loguru import logger

from server.core.analysis.base import IAnalysisBuilder, IAnalysisStrategy, AnalysisType
from server.services.book_analysis.factory import StrategyFactory


class AnalysisBuilder(IAnalysisBuilder):
    """
    Fluent builder for configuring book analysis pipeline.

    Usage:
        builder = AnalysisBuilder()
        strategies = (builder
            .with_genre_analysis()
            .with_character_analysis()
            .with_tension_analysis()
            .build())

    Or:
        strategies = AnalysisBuilder().with_all_analyses().build()
    """

    def __init__(self, factory: Optional[StrategyFactory] = None):
        """
        Initialize builder.

        Args:
            factory: Strategy factory (creates default if not provided)
        """
        self.factory = factory or StrategyFactory()
        self._strategies: List[IAnalysisStrategy] = []
        self._selected_types = set()

    def with_genre_analysis(self) -> 'AnalysisBuilder':
        """Add genre analysis."""
        return self._add_strategy(AnalysisType.GENRE)

    def with_character_analysis(self) -> 'AnalysisBuilder':
        """Add character analysis."""
        return self._add_strategy(AnalysisType.CHARACTER)

    def with_tension_analysis(self) -> 'AnalysisBuilder':
        """Add tension analysis."""
        return self._add_strategy(AnalysisType.TENSION)

    def with_pace_analysis(self) -> 'AnalysisBuilder':
        """Add pace analysis."""
        return self._add_strategy(AnalysisType.PACE)

    def with_water_analysis(self) -> 'AnalysisBuilder':
        """Add water level analysis."""
        return self._add_strategy(AnalysisType.WATER)

    def with_theme_analysis(self) -> 'AnalysisBuilder':
        """Add theme analysis."""
        return self._add_strategy(AnalysisType.THEME)

    def with_style_analysis(self) -> 'AnalysisBuilder':
        """Add style analysis."""
        return self._add_strategy(AnalysisType.STYLE)

    def with_audience_analysis(self) -> 'AnalysisBuilder':
        """Add audience analysis."""
        return self._add_strategy(AnalysisType.AUDIENCE)

    def with_analysis(self, analysis_type: AnalysisType) -> 'AnalysisBuilder':
        """
        Add analysis by type.

        Args:
            analysis_type: Type of analysis to add

        Returns:
            Self for chaining
        """
        return self._add_strategy(analysis_type)

    def with_analyses(self, *analysis_types: AnalysisType) -> 'AnalysisBuilder':
        """
        Add multiple analyses at once.

        Args:
            *analysis_types: Variable number of analysis types

        Returns:
            Self for chaining

        Example:
            builder.with_analyses(
                AnalysisType.GENRE,
                AnalysisType.CHARACTER,
                AnalysisType.TENSION
            )
        """
        for analysis_type in analysis_types:
            self._add_strategy(analysis_type)
        return self

    def with_all_analyses(self) -> 'AnalysisBuilder':
        """Add all available analyses."""
        return self.with_analyses(
            AnalysisType.GENRE,
            AnalysisType.CHARACTER,
            AnalysisType.TENSION,
            AnalysisType.PACE,
            AnalysisType.WATER,
            AnalysisType.THEME
        )

    def with_fast_analyses(self) -> 'AnalysisBuilder':
        """
        Add only fast analyses (no LLM required).

        Fast analyses: Pace, Water
        """
        return self.with_analyses(
            AnalysisType.PACE,
            AnalysisType.WATER
        )

    def with_essential_analyses(self) -> 'AnalysisBuilder':
        """
        Add essential analyses for MVP.

        Essential: Genre, Character, Tension, Pace, Water
        """
        return self.with_analyses(
            AnalysisType.GENRE,
            AnalysisType.CHARACTER,
            AnalysisType.TENSION,
            AnalysisType.PACE,
            AnalysisType.WATER
        )

    def build(self) -> List[IAnalysisStrategy]:
        """
        Build and return configured strategies.

        Returns:
            List of configured strategy instances
        """
        if not self._strategies:
            logger.warning("No strategies configured, returning empty list")

        logger.info(f"Built {len(self._strategies)} analysis strategies")
        return self._strategies.copy()

    def reset(self) -> 'AnalysisBuilder':
        """
        Reset builder to empty state.

        Returns:
            Self for chaining
        """
        self._strategies = []
        self._selected_types = set()
        return self

    def _add_strategy(self, analysis_type: AnalysisType) -> 'AnalysisBuilder':
        """
        Internal method to add strategy.

        Args:
            analysis_type: Type of analysis

        Returns:
            Self for chaining
        """
        # Prevent duplicates
        if analysis_type in self._selected_types:
            logger.debug(f"Strategy {analysis_type.value} already added, skipping")
            return self

        try:
            strategy = self.factory.create_strategy(analysis_type)
            self._strategies.append(strategy)
            self._selected_types.add(analysis_type)

            logger.debug(f"Added strategy: {analysis_type.value}")

        except Exception as e:
            logger.error(f"Failed to create strategy {analysis_type.value}: {e}")

        return self

    def get_estimated_time(self, chunk_count: int) -> float:
        """
        Get estimated total execution time.

        Args:
            chunk_count: Number of chunks to process

        Returns:
            Estimated time in seconds
        """
        total_time = 0.0

        for strategy in self._strategies:
            total_time += strategy.get_estimated_time(chunk_count)

        # Add 10% overhead for orchestration
        total_time *= 1.1

        return total_time

    def summary(self) -> str:
        """
        Get summary of configured analyses.

        Returns:
            Human-readable summary
        """
        if not self._strategies:
            return "No analyses configured"

        lines = [f"Configured {len(self._strategies)} analyses:"]

        for i, strategy in enumerate(self._strategies, 1):
            llm_required = "🤖 LLM" if strategy.requires_llm() else "⚡ Fast"
            lines.append(f"  {i}. {strategy.get_type().value} ({llm_required})")

        return "\n".join(lines)
