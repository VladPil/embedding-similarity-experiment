"""
Service for comparing texts in collections.
"""

import asyncio
from typing import Dict, Any, Optional
import numpy as np

from .collections import (
    TextCollection,
    TextItem,
    ComparisonResult,
    ComparisonMatrix,
)
from .service import BookAnalysisService
from server.core.analysis.base import AnalysisType
from server.core.chunks.manager import ChunkManager
from server.core.analysis.chunk_indexer import Chunk
from server.services.similarity_service import SimilarityService
from server.core.embeddings.manager import EmbeddingManager


class CollectionComparisonService:
    """Service for comparing multiple texts in a collection."""

    def __init__(self):
        self.analysis_service = BookAnalysisService()
        self.similarity_service = SimilarityService()
        self.embedding_manager = EmbeddingManager()

    def _convert_chunks_to_objects(self, chunk_dicts):
        """Convert dictionary chunks to Chunk objects."""
        chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            chunk = Chunk(
                index=i,
                text=chunk_dict['text'],
                start_pos=chunk_dict.get('start', 0),
                end_pos=chunk_dict.get('end', len(chunk_dict['text'])),
                position_ratio=i / len(chunk_dicts) if chunk_dicts else 0.0
            )
            chunks.append(chunk)
        return chunks

    async def analyze_text(
        self,
        text_item: TextItem,
        analyses: list[str],
        chunk_size: int = 2000,
    ) -> Dict[str, Any]:
        """
        Analyze single text.

        Args:
            text_item: Text to analyze
            analyses: List of analysis types
            chunk_size: Chunk size

        Returns:
            Analysis results
        """
        # Convert string analysis names to AnalysisType enums
        valid_types = {
            "genre": AnalysisType.GENRE,
            "character": AnalysisType.CHARACTER,
            "pace": AnalysisType.PACE,
            "tension": AnalysisType.TENSION,
            "water": AnalysisType.WATER,
            "theme": AnalysisType.THEME,
        }

        selected_analyses = []
        for analysis_name in analyses:
            if analysis_name in valid_types:
                selected_analyses.append(valid_types[analysis_name])

        # Create chunks
        chunk_manager = ChunkManager(chunk_size=chunk_size)
        chunk_dicts = chunk_manager.chunk_by_characters(text_item.content)
        chunks = self._convert_chunks_to_objects(chunk_dicts)

        # Run analysis
        result = await self.analysis_service.analyze_book(
            text=text_item.content,
            chunks=chunks,
            selected_analyses=selected_analyses,
            metadata={
                "text_id": text_item.text_id,
                "title": text_item.title,
                **text_item.metadata,
            },
        )

        return result

    async def analyze_collection(
        self,
        collection: TextCollection,
        progress_callback=None,
    ) -> TextCollection:
        """
        Analyze all texts in collection.

        Args:
            collection: Text collection
            progress_callback: Optional callback for progress updates

        Returns:
            Collection with analysis results
        """
        total = len(collection.texts)

        for i, text_item in enumerate(collection.texts):
            if progress_callback:
                await progress_callback(
                    step="analysis",
                    current=i + 1,
                    total=total,
                    message=f"Analyzing: {text_item.title}",
                )

            # Run analysis
            result = await self.analyze_text(
                text_item=text_item,
                analyses=collection.strategy.analyses,
                chunk_size=collection.strategy.chunk_size,
            )

            # Cache results
            text_item.analysis_results = result

        return collection

    def _compare_analyses(
        self,
        text1_results: Dict[str, Any],
        text2_results: Dict[str, Any],
        analyses: list[str],
    ) -> Dict[str, Any]:
        """
        Compare analysis results between two texts.

        Args:
            text1_results: First text analysis results
            text2_results: Second text analysis results
            analyses: List of analyses to compare

        Returns:
            Comparison results per analyzer
        """
        comparisons = {}

        results1 = text1_results.get("results", {})
        results2 = text2_results.get("results", {})

        for analysis_name in analyses:
            if analysis_name not in results1 or analysis_name not in results2:
                continue

            result1 = results1[analysis_name]
            result2 = results2[analysis_name]

            # Delegate to specific comparison method
            if analysis_name == "genre":
                comparison = self._compare_genre(result1, result2)
            elif analysis_name == "character":
                comparison = self._compare_character(result1, result2)
            elif analysis_name == "pace":
                comparison = self._compare_pace(result1, result2)
            elif analysis_name == "tension":
                comparison = self._compare_tension(result1, result2)
            elif analysis_name == "water":
                comparison = self._compare_water(result1, result2)
            elif analysis_name == "theme":
                comparison = self._compare_theme(result1, result2)
            else:
                comparison = {"similarity": 0.5, "note": "No comparison method"}

            comparisons[analysis_name] = comparison

        return comparisons

    def _compare_genre(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare genre analysis results."""
        # Simple string comparison for now
        # TODO: Use semantic similarity or genre taxonomy
        r1_str = str(result1).lower()
        r2_str = str(result2).lower()

        if r1_str == r2_str:
            similarity = 1.0
        elif any(word in r2_str for word in r1_str.split()[:3]):
            similarity = 0.7
        else:
            similarity = 0.3

        return {
            "similarity": similarity,
            "text1_genre": result1,
            "text2_genre": result2,
        }

    def _compare_character(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare character analysis results."""
        # Extract character mentions
        # TODO: Implement proper character extraction
        return {
            "similarity": 0.5,
            "note": "Character comparison not fully implemented",
        }

    def _compare_pace(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare pace analysis results."""
        # TODO: Extract pace scores and compare
        return {
            "similarity": 0.5,
            "note": "Pace comparison not fully implemented",
        }

    def _compare_tension(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare tension analysis results."""
        # TODO: Extract tension scores and compare
        return {
            "similarity": 0.5,
            "note": "Tension comparison not fully implemented",
        }

    def _compare_water(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare water/quality analysis results."""
        # TODO: Extract quality metrics and compare
        return {
            "similarity": 0.5,
            "note": "Water comparison not fully implemented",
        }

    def _compare_theme(self, result1: Any, result2: Any) -> Dict[str, Any]:
        """Compare theme analysis results."""
        # TODO: Extract themes and compare overlap
        return {
            "similarity": 0.5,
            "note": "Theme comparison not fully implemented",
        }

    async def compare_pair(
        self,
        text1: TextItem,
        text2: TextItem,
        analyses: list[str],
        embedding_method: str = "hybrid",
        embedding_model: str = "multilingual-e5-small",
    ) -> ComparisonResult:
        """
        Compare two texts.

        Args:
            text1: First text
            text2: Second text
            analyses: List of analyses to compare
            embedding_method: Method for embedding comparison (cosine/semantic/hybrid)
            embedding_model: Embedding model to use

        Returns:
            Comparison result
        """
        # Get cached analysis results
        if not text1.analysis_results or not text2.analysis_results:
            raise ValueError("Texts must be analyzed before comparison")

        # Compare analyses
        comparisons = self._compare_analyses(
            text1.analysis_results,
            text2.analysis_results,
            analyses,
        )

        # Calculate embedding similarity
        embedding_similarity = None
        try:
            # Map embedding_method to preset name
            preset_map = {
                "cosine": "fast",
                "semantic": "semantic",
                "hybrid": "hybrid",
            }
            preset = preset_map.get(embedding_method, "hybrid")

            # Calculate similarity using preset
            similarity_result = await self.similarity_service.calculate_with_preset(
                preset=preset,
                text1=text1.content,
                text2=text2.content,
                metadata1={"text_id": text1.text_id, "title": text1.title},
                metadata2={"text_id": text2.text_id, "title": text2.title},
            )

            # Extract final score
            if similarity_result.get("success") and "final_score" in similarity_result:
                embedding_similarity = similarity_result["final_score"]

        except Exception as e:
            # Log error but don't fail the comparison
            print(f"Warning: Embedding comparison failed: {e}")

        # Calculate overall similarity (average of all comparisons + embedding)
        similarities = [
            comp.get("similarity", 0.5)
            for comp in comparisons.values()
        ]

        # Include embedding similarity in overall calculation if available
        if embedding_similarity is not None:
            similarities.append(embedding_similarity)

        overall_similarity = np.mean(similarities) if similarities else 0.5

        # Generate summary
        summary_parts = [f"Compared {len(analyses)} aspects"]
        if embedding_similarity is not None:
            summary_parts.append(f"embedding similarity: {embedding_similarity:.2%}")
        summary_parts.append(f"Overall: {overall_similarity:.2%}")
        summary = ". ".join(summary_parts)

        return ComparisonResult(
            text1_id=text1.text_id,
            text2_id=text2.text_id,
            text1_title=text1.title,
            text2_title=text2.title,
            overall_similarity=float(overall_similarity),
            analysis_comparisons=comparisons,
            embedding_similarity=embedding_similarity,
            summary=summary,
        )

    async def compare_collection(
        self,
        collection: TextCollection,
        progress_callback=None,
    ) -> ComparisonMatrix:
        """
        Compare all pairs in collection.

        Args:
            collection: Text collection
            progress_callback: Optional callback for progress updates

        Returns:
            Comparison matrix
        """
        # First, analyze all texts if not already done
        for text_item in collection.texts:
            if not text_item.analysis_results:
                await self.analyze_text(
                    text_item=text_item,
                    analyses=collection.strategy.analyses,
                    chunk_size=collection.strategy.chunk_size,
                )

        # Get pairs to compare
        pairs = collection.get_comparison_pairs()

        # Create matrix
        matrix = ComparisonMatrix(
            collection_id=collection.collection_id,
            total_comparisons=len(pairs),
        )

        # Compare each pair
        for i, (text1_id, text2_id) in enumerate(pairs):
            if progress_callback:
                await progress_callback(
                    step="comparison",
                    current=i + 1,
                    total=len(pairs),
                    message=f"Comparing texts {i + 1}/{len(pairs)}",
                )

            text1 = collection.get_text(text1_id)
            text2 = collection.get_text(text2_id)

            if not text1 or not text2:
                continue

            # Compare pair
            result = await self.compare_pair(
                text1=text1,
                text2=text2,
                analyses=collection.strategy.analyses,
                embedding_method=collection.strategy.embedding_method,
                embedding_model=collection.strategy.embedding_model,
            )

            # Add to matrix
            matrix.add_result(result)

        return matrix
