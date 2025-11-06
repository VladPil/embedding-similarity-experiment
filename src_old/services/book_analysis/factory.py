"""
Strategy Factory - creates analysis strategies.
Implements Factory pattern with registry for extensibility.
"""

from typing import Dict, Type
from loguru import logger

from server.core.analysis.base import IStrategyFactory, IAnalysisStrategy, AnalysisType
from server.services.book_analysis.strategies import (
    GenreAnalysisStrategy,
    CharacterAnalysisStrategy,
    TensionAnalysisStrategy,
    PaceAnalysisStrategy,
    WaterAnalysisStrategy,
    ThemeAnalysisStrategy,
)


class StrategyFactory(IStrategyFactory):
    """
    Factory for creating analysis strategies.

    - Maintains registry of available strategies
    - Allows registration of custom strategies
    - Creates strategy instances on demand
    """

    def __init__(self):
        """Initialize factory with default strategies."""
        self._registry: Dict[AnalysisType, Type[IAnalysisStrategy]] = {}

        # Register default strategies
        self._register_defaults()

    def _register_defaults(self):
        """Register all default analysis strategies."""
        self.register_strategy(AnalysisType.GENRE, GenreAnalysisStrategy)
        self.register_strategy(AnalysisType.CHARACTER, CharacterAnalysisStrategy)
        self.register_strategy(AnalysisType.TENSION, TensionAnalysisStrategy)
        self.register_strategy(AnalysisType.PACE, PaceAnalysisStrategy)
        self.register_strategy(AnalysisType.WATER, WaterAnalysisStrategy)
        self.register_strategy(AnalysisType.THEME, ThemeAnalysisStrategy)

        logger.debug(f"Registered {len(self._registry)} default strategies")

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

        Raises:
            ValueError: If strategy type not registered
        """
        if analysis_type not in self._registry:
            raise ValueError(
                f"Strategy type '{analysis_type.value}' not registered. "
                f"Available: {list(self._registry.keys())}"
            )

        strategy_class = self._registry[analysis_type]

        try:
            # Create instance with kwargs
            strategy = strategy_class(**kwargs)
            logger.debug(f"Created strategy: {analysis_type.value}")
            return strategy

        except Exception as e:
            logger.error(f"Failed to create strategy {analysis_type.value}: {e}")
            raise

    def register_strategy(
        self,
        analysis_type: AnalysisType,
        strategy_class: Type[IAnalysisStrategy]
    ) -> None:
        """
        Register new strategy type.

        Args:
            analysis_type: Type identifier
            strategy_class: Strategy class to register

        Example:
            factory.register_strategy(
                AnalysisType.CUSTOM,
                MyCustomStrategy
            )
        """
        if analysis_type in self._registry:
            logger.warning(
                f"Overwriting existing strategy: {analysis_type.value}"
            )

        self._registry[analysis_type] = strategy_class
        logger.info(f"Registered strategy: {analysis_type.value}")

    def unregister_strategy(self, analysis_type: AnalysisType) -> None:
        """
        Unregister strategy type.

        Args:
            analysis_type: Type to unregister
        """
        if analysis_type in self._registry:
            del self._registry[analysis_type]
            logger.info(f"Unregistered strategy: {analysis_type.value}")

    def get_available_types(self) -> list[AnalysisType]:
        """Get list of available analysis types."""
        return list(self._registry.keys())

    def is_registered(self, analysis_type: AnalysisType) -> bool:
        """Check if strategy type is registered."""
        return analysis_type in self._registry
