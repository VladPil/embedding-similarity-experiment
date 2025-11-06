"""
Similarity Analysis Service.
Orchestrates similarity calculation with multiple strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import time

from server.core.similarity.base import (
    SimilarityContext,
    SimilarityResult,
    SimilarityMethod,
    SimilarityScope
)
from server.core.similarity.builder import SimilarityBuilder
from server.core.similarity.factory import SimilarityFactory
from server.core.similarity.aggregator import WeightedAggregator


class SimilarityService:
    """
    Service for orchestrating similarity analysis.

    Responsibilities:
    - Prepare similarity context
    - Execute multiple similarity strategies
    - Aggregate results
    - Provide detailed breakdown
    - Handle errors gracefully
    """

    def __init__(self, factory: Optional[SimilarityFactory] = None):
        """
        Initialize similarity service.

        Args:
            factory: Similarity factory (creates default if not provided)
        """
        self.factory = factory or SimilarityFactory()
        self.builder = SimilarityBuilder(self.factory)

    async def calculate_similarity(
        self,
        text1: str,
        text2: str,
        embeddings1: Optional[List[Any]] = None,
        embeddings2: Optional[List[Any]] = None,
        chunks1: Optional[List[Any]] = None,
        chunks2: Optional[List[Any]] = None,
        selected_methods: Optional[List[Tuple[SimilarityMethod, SimilarityScope]]] = None,
        aggregate: bool = True,
        metadata1: Optional[Dict] = None,
        metadata2: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            embeddings1: Optional embeddings for text1
            embeddings2: Optional embeddings for text2
            chunks1: Optional chunks for text1
            chunks2: Optional chunks for text2
            selected_methods: List of (method, scope) tuples (None = default)
            aggregate: Whether to aggregate results
            metadata1: Optional metadata for text1
            metadata2: Optional metadata for text2

        Returns:
            Dictionary with similarity results and interpretations
        """
        try:
            logger.info("Starting similarity calculation")

            # Create context
            context = SimilarityContext(
                text1=text1,
                text2=text2,
                embeddings1=embeddings1,
                embeddings2=embeddings2,
                chunks1=chunks1,
                chunks2=chunks2,
                metadata1=metadata1,
                metadata2=metadata2
            )

            # Build strategies
            strategies, aggregator = self._build_strategies(
                selected_methods,
                aggregate
            )

            if not strategies:
                logger.warning("No similarity strategies configured")
                return {
                    "success": False,
                    "error": "No similarity strategies configured",
                    "results": {}
                }

            logger.info(f"Running {len(strategies)} similarity strategies")

            # Execute strategies
            results = []
            total_time = 0.0

            for strategy in strategies:
                try:
                    start_time = time.time()
                    result = await strategy.calculate(context)
                    execution_time = time.time() - start_time
                    total_time += execution_time

                    results.append(result)

                    logger.info(
                        f"✓ {strategy.get_method().value} ({strategy.get_scope().value}) "
                        f"completed: {result.score:.3f} in {execution_time:.1f}s"
                    )

                except Exception as e:
                    logger.error(
                        f"✗ {strategy.get_method().value} failed: {e}"
                    )

            # Aggregate if requested
            final_score = None
            aggregated_interpretation = None

            if aggregate and aggregator and results:
                try:
                    final_score, aggregated_interpretation = aggregator.aggregate(results)
                    logger.info(f"Aggregated similarity: {final_score:.3f}")
                except Exception as e:
                    logger.error(f"Aggregation failed: {e}")

            # Format response
            response = self._format_response(
                results,
                final_score,
                aggregated_interpretation,
                total_time,
                context
            )

            logger.info(
                f"Similarity calculation completed: {len(results)} methods"
            )

            return response

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }

    async def calculate_with_preset(
        self,
        preset: str,
        text1: str,
        text2: str,
        embeddings1: Optional[List[Any]] = None,
        embeddings2: Optional[List[Any]] = None,
        chunks1: Optional[List[Any]] = None,
        chunks2: Optional[List[Any]] = None,
        metadata1: Optional[Dict] = None,
        metadata2: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate similarity using preset configuration.

        Args:
            preset: Preset name ('fast', 'comprehensive', 'semantic')
            text1, text2: Texts to compare
            embeddings1, embeddings2: Optional embeddings
            chunks1, chunks2: Optional chunks
            metadata1, metadata2: Optional metadata

        Returns:
            Similarity results
        """
        # Map preset to methods
        preset_methods = {
            'fast': [
                (SimilarityMethod.COSINE, SimilarityScope.FULL_TEXT)
            ],
            'comprehensive': [
                (SimilarityMethod.COSINE, SimilarityScope.FULL_TEXT),
                (SimilarityMethod.COSINE, SimilarityScope.CHUNK),
                (SimilarityMethod.SEMANTIC, SimilarityScope.THEMATIC),
                (SimilarityMethod.HYBRID, SimilarityScope.FULL_TEXT)
            ],
            'semantic': [
                (SimilarityMethod.SEMANTIC, SimilarityScope.THEMATIC)
            ],
            'hybrid': [
                (SimilarityMethod.HYBRID, SimilarityScope.FULL_TEXT)
            ]
        }

        selected = preset_methods.get(preset)

        if selected is None:
            raise ValueError(f"Unknown preset: {preset}")

        return await self.calculate_similarity(
            text1=text1,
            text2=text2,
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            chunks1=chunks1,
            chunks2=chunks2,
            selected_methods=selected,
            aggregate=True,
            metadata1=metadata1,
            metadata2=metadata2
        )

    def get_available_methods(self) -> List[str]:
        """
        Get list of available similarity methods.

        Returns:
            List of method names
        """
        return [m.value for m in self.factory.get_available_methods()]

    def get_available_scopes(self, method: SimilarityMethod) -> List[str]:
        """
        Get available scopes for a method.

        Args:
            method: Similarity method

        Returns:
            List of scope names
        """
        scopes = self.factory.get_available_scopes(method)
        return [s.value for s in scopes]

    def _build_strategies(
        self,
        selected_methods: Optional[List[Tuple[SimilarityMethod, SimilarityScope]]] = None,
        aggregate: bool = True
    ) -> Tuple[List, Optional[Any]]:
        """
        Build similarity strategies based on selection.

        Args:
            selected_methods: List of (method, scope) tuples (None = default)
            aggregate: Whether to create aggregator

        Returns:
            Tuple of (strategies, aggregator)
        """
        builder = SimilarityBuilder(self.factory)

        if selected_methods is None:
            # Use default: cosine + semantic
            builder.with_cosine_similarity().with_semantic_similarity()
        else:
            # Use selected methods
            for method, scope in selected_methods:
                builder.with_method(method, scope)

        # Add aggregation if requested
        if aggregate:
            builder.with_aggregation(method="weighted")

        return builder.build_with_aggregator()

    def _format_response(
        self,
        results: List[SimilarityResult],
        final_score: Optional[float],
        aggregated_interpretation: Optional[str],
        total_time: float,
        context: SimilarityContext
    ) -> Dict[str, Any]:
        """
        Format similarity results into structured response.

        Args:
            results: List of similarity results
            final_score: Aggregated final score
            aggregated_interpretation: Aggregated interpretation
            total_time: Total execution time
            context: Similarity context

        Returns:
            Formatted response dictionary
        """
        # Build results dict
        results_dict = {}
        interpretations = {}

        for result in results:
            key = f"{result.method.value}_{result.scope.value}"
            results_dict[key] = {
                "score": result.score,
                "confidence": result.confidence,
                "details": result.details
            }
            interpretations[key] = result.interpretation

        # Statistics
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        statistics = {
            "total_methods": len(results),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "total_execution_time": round(total_time, 2),
            "text1_length": len(context.text1),
            "text2_length": len(context.text2)
        }

        # Build response
        response = {
            "success": True,
            "results": results_dict,
            "interpretations": interpretations,
            "statistics": statistics
        }

        # Add aggregated score if available
        if final_score is not None:
            response["final_score"] = round(final_score, 4)
            response["final_interpretation"] = aggregated_interpretation

        # Add metadata
        if context.metadata1 or context.metadata2:
            response["metadata"] = {
                "text1": context.metadata1,
                "text2": context.metadata2
            }

        return response
