"""
Similarity Strategy Factory.
Creates similarity calculation strategies.
Implements Factory pattern with registry for extensibility.
"""

from typing import Dict, Tuple, Type, List
from loguru import logger

from server.core.similarity.base import (
    ISimilarityFactory,
    ISimilarityStrategy,
    SimilarityMethod,
    SimilarityScope
)
from server.core.similarity.strategies import (
    CosineSimilarityStrategy,
    SemanticSimilarityStrategy,
    HybridSimilarityStrategy
)


class SimilarityFactory(ISimilarityFactory):
    """
    Factory for creating similarity strategies.

    - Maintains registry of available strategies
    - Allows registration of custom strategies
    - Creates strategy instances on demand
    - Supports method+scope combinations
    """

    def __init__(self):
        """Initialize factory with default strategies."""
        self._registry: Dict[Tuple[SimilarityMethod, SimilarityScope], Type[ISimilarityStrategy]] = {}

        # Register default strategies
        self._register_defaults()

    def _register_defaults(self):
        """Register all default similarity strategies."""
        # Cosine similarity - all scopes
        self.register_strategy(
            SimilarityMethod.COSINE,
            SimilarityScope.FULL_TEXT,
            CosineSimilarityStrategy
        )
        self.register_strategy(
            SimilarityMethod.COSINE,
            SimilarityScope.CHUNK,
            CosineSimilarityStrategy
        )

        # Semantic similarity - thematic scope
        self.register_strategy(
            SimilarityMethod.SEMANTIC,
            SimilarityScope.THEMATIC,
            SemanticSimilarityStrategy
        )

        # Hybrid similarity - all scopes
        self.register_strategy(
            SimilarityMethod.HYBRID,
            SimilarityScope.FULL_TEXT,
            HybridSimilarityStrategy
        )
        self.register_strategy(
            SimilarityMethod.HYBRID,
            SimilarityScope.CHUNK,
            HybridSimilarityStrategy
        )

        logger.debug(f"Registered {len(self._registry)} default similarity strategies")

    def create_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        **kwargs
    ) -> ISimilarityStrategy:
        """
        Create similarity strategy by method and scope.

        Args:
            method: Similarity method
            scope: Similarity scope
            **kwargs: Strategy-specific parameters

        Returns:
            Configured strategy instance

        Raises:
            ValueError: If strategy type not registered
        """
        key = (method, scope)

        if key not in self._registry:
            # Try to find strategy with same method but different scope
            available_scopes = [
                s for (m, s) in self._registry.keys() if m == method
            ]

            if available_scopes:
                logger.warning(
                    f"Scope '{scope.value}' not available for '{method.value}', "
                    f"using '{available_scopes[0].value}'"
                )
                scope = available_scopes[0]
                key = (method, scope)
            else:
                raise ValueError(
                    f"Strategy '{method.value}' with scope '{scope.value}' not registered. "
                    f"Available: {list(self._registry.keys())}"
                )

        strategy_class = self._registry[key]

        try:
            # Create instance with scope and kwargs
            strategy = strategy_class(scope=scope, **kwargs)
            logger.debug(f"Created strategy: {method.value} ({scope.value})")
            return strategy

        except Exception as e:
            logger.error(f"Failed to create strategy {method.value}: {e}")
            raise

    def register_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope,
        strategy_class: Type[ISimilarityStrategy]
    ) -> None:
        """
        Register new strategy type.

        Args:
            method: Similarity method
            scope: Similarity scope
            strategy_class: Strategy class to register

        Example:
            factory.register_strategy(
                SimilarityMethod.CUSTOM,
                SimilarityScope.FULL_TEXT,
                MyCustomStrategy
            )
        """
        key = (method, scope)

        if key in self._registry:
            logger.warning(
                f"Overwriting existing strategy: {method.value} ({scope.value})"
            )

        self._registry[key] = strategy_class
        logger.info(f"Registered strategy: {method.value} ({scope.value})")

    def unregister_strategy(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope
    ) -> None:
        """
        Unregister strategy type.

        Args:
            method: Similarity method
            scope: Similarity scope
        """
        key = (method, scope)

        if key in self._registry:
            del self._registry[key]
            logger.info(f"Unregistered strategy: {method.value} ({scope.value})")

    def get_available_methods(self) -> List[SimilarityMethod]:
        """Get list of available similarity methods."""
        methods = set(method for (method, _) in self._registry.keys())
        return list(methods)

    def get_available_scopes(self, method: SimilarityMethod) -> List[SimilarityScope]:
        """
        Get available scopes for a specific method.

        Args:
            method: Similarity method

        Returns:
            List of available scopes
        """
        scopes = [scope for (m, scope) in self._registry.keys() if m == method]
        return scopes

    def is_registered(
        self,
        method: SimilarityMethod,
        scope: SimilarityScope
    ) -> bool:
        """Check if strategy is registered."""
        return (method, scope) in self._registry
