"""
Book Analysis Service.
Orchestrates book analysis pipeline with multiple strategies.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import time

from server.core.analysis.base import (
    AnalysisContext,
    AnalysisResult,
    AnalysisType
)
from server.services.book_analysis.builder import AnalysisBuilder
from server.services.book_analysis.factory import StrategyFactory


class BookAnalysisService:
    """
    Service for orchestrating book analysis.

    Responsibilities:
    - Prepare analysis context
    - Execute multiple analysis strategies
    - Collect and format results
    - Handle errors gracefully
    - Provide progress tracking
    """

    def __init__(self, factory: Optional[StrategyFactory] = None):
        """
        Initialize book analysis service.

        Args:
            factory: Strategy factory (creates default if not provided)
        """
        self.factory = factory or StrategyFactory()
        self.builder = AnalysisBuilder(self.factory)

    async def analyze_book(
        self,
        text: str,
        chunks: List[Any],
        embeddings: Optional[List[Any]] = None,
        selected_analyses: Optional[List[AnalysisType]] = None,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Analyze book with selected analysis types.

        Args:
            text: Full book text
            chunks: Text chunks
            embeddings: Optional embeddings for chunks
            selected_analyses: List of analysis types to run (None = all)
            metadata: Optional metadata
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with all analysis results and interpretations
        """
        try:
            logger.info(f"Starting book analysis (chunks: {len(chunks)})")

            # Create analysis context
            context = AnalysisContext(
                text=text,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            # Build strategies
            strategies = self._build_strategies(selected_analyses)

            if not strategies:
                logger.warning("No analysis strategies configured")
                return {
                    "success": False,
                    "error": "No analysis strategies configured",
                    "results": {}
                }

            logger.info(f"Running {len(strategies)} analysis strategies")

            # Execute analyses
            results = []
            total = len(strategies)

            for i, strategy in enumerate(strategies, 1):
                try:
                    # Progress callback
                    if progress_callback:
                        progress_callback(i, total, strategy.get_type().value)

                    # Execute analysis
                    start_time = time.time()
                    analysis_data = await strategy.analyze(context)
                    execution_time = time.time() - start_time

                    # Interpret results
                    interpretation = strategy.interpret_results(analysis_data)

                    # Create result
                    result = AnalysisResult(
                        analysis_type=strategy.get_type(),
                        data=analysis_data,
                        execution_time=execution_time,
                        success=True
                    )

                    # Add interpretation
                    result.data['interpretation'] = interpretation

                    results.append(result)

                    logger.info(
                        f"✓ {strategy.get_type().value} completed in {execution_time:.1f}s"
                    )

                except Exception as e:
                    logger.error(f"✗ {strategy.get_type().value} failed: {e}")

                    # Create error result
                    result = AnalysisResult(
                        analysis_type=strategy.get_type(),
                        data={"error": str(e)},
                        execution_time=0.0,
                        success=False,
                        error=str(e)
                    )
                    results.append(result)

            # Format response
            response = self._format_response(results, context)

            logger.info(
                f"Book analysis completed: {response['statistics']['successful_analyses']}/{total} successful"
            )

            return response

        except Exception as e:
            logger.error(f"Book analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }

    async def analyze_with_preset(
        self,
        preset: str,
        text: str,
        chunks: List[Any],
        embeddings: Optional[List[Any]] = None,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Analyze book using preset configuration.

        Args:
            preset: Preset name ('all', 'fast', 'essential')
            text: Full book text
            chunks: Text chunks
            embeddings: Optional embeddings
            metadata: Optional metadata
            progress_callback: Optional progress callback

        Returns:
            Analysis results
        """
        # Map preset to analysis types
        preset_map = {
            'all': None,  # Will use with_all_analyses()
            'fast': [AnalysisType.PACE, AnalysisType.WATER],
            'essential': [
                AnalysisType.GENRE,
                AnalysisType.CHARACTER,
                AnalysisType.TENSION,
                AnalysisType.PACE,
                AnalysisType.WATER
            ]
        }

        selected = preset_map.get(preset)

        if selected is None and preset == 'all':
            # Use builder preset
            return await self.analyze_book(
                text=text,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
                progress_callback=progress_callback
            )
        elif selected:
            return await self.analyze_book(
                text=text,
                chunks=chunks,
                embeddings=embeddings,
                selected_analyses=selected,
                metadata=metadata,
                progress_callback=progress_callback
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def get_available_analyses(self) -> List[str]:
        """
        Get list of available analysis types.

        Returns:
            List of analysis type names
        """
        return [at.value for at in self.factory.get_available_types()]

    def get_estimated_time(
        self,
        chunk_count: int,
        selected_analyses: Optional[List[AnalysisType]] = None
    ) -> Dict[str, float]:
        """
        Get estimated execution time for analyses.

        Args:
            chunk_count: Number of chunks
            selected_analyses: List of analysis types (None = all)

        Returns:
            Dictionary with time estimates per analysis and total
        """
        strategies = self._build_strategies(selected_analyses)

        estimates = {}
        total = 0.0

        for strategy in strategies:
            analysis_type = strategy.get_type().value
            estimated_time = strategy.get_estimated_time(chunk_count)
            estimates[analysis_type] = estimated_time
            total += estimated_time

        estimates['total'] = total
        return estimates

    def _build_strategies(
        self,
        selected_analyses: Optional[List[AnalysisType]] = None
    ) -> List:
        """
        Build analysis strategies based on selection.

        Args:
            selected_analyses: List of analysis types (None = all)

        Returns:
            List of configured strategies
        """
        builder = AnalysisBuilder(self.factory)

        if selected_analyses is None:
            # Use all analyses
            return builder.with_all_analyses().build()
        else:
            # Use selected analyses
            return builder.with_analyses(*selected_analyses).build()

    def _format_response(
        self,
        results: List[AnalysisResult],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """
        Format analysis results into structured response.

        Args:
            results: List of analysis results
            context: Analysis context

        Returns:
            Formatted response dictionary
        """
        # Separate successful and failed
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Build results dict
        results_dict = {}
        interpretations = {}

        for result in results:
            analysis_type = result.analysis_type.value
            results_dict[analysis_type] = result.data

            # Extract interpretation if available
            if 'interpretation' in result.data:
                interpretations[analysis_type] = result.data['interpretation']

        # Calculate statistics
        total_time = sum(r.execution_time for r in results)

        statistics = {
            "total_analyses": len(results),
            "successful_analyses": len(successful),
            "failed_analyses": len(failed),
            "total_execution_time": round(total_time, 2),
            "chunks_analyzed": len(context.chunks),
            "text_length": len(context.text)
        }

        # Build response
        response = {
            "success": len(failed) == 0,
            "results": results_dict,
            "interpretations": interpretations,
            "statistics": statistics,
            "metadata": context.metadata or {}
        }

        # Add errors if any
        if failed:
            response["errors"] = {
                r.analysis_type.value: r.error
                for r in failed
            }

        return response
