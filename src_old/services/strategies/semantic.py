"""
Semantic analysis strategy using embeddings.
"""

from typing import Dict, Any

from server.services.strategies.base import AnalysisStrategy
from server.services.embedding_service import EmbeddingService
from server.core.similarity_calc import SimilarityCalculator


class SemanticAnalysisStrategy(AnalysisStrategy):
    """Strategy for semantic analysis using embeddings."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.similarity_calc = SimilarityCalculator()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform semantic similarity analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters including text1_id, text2_id, model

        Returns:
            Dictionary with similarity, interpretation, model_used, dimensions
        """
        model_key = params.get("model", "multilingual-e5-base")

        # Get embeddings
        text1_id = params.get("text1_id")
        text2_id = params.get("text2_id")

        emb1 = await self.embedding_service.get_embedding(text1_id, model_key)
        emb2 = await self.embedding_service.get_embedding(text2_id, model_key)

        if emb1 is None or emb2 is None:
            raise ValueError("Failed to generate embeddings")

        # Calculate similarity
        similarity = self.similarity_calc.cosine_similarity(emb1, emb2)
        base_interpretation = self.similarity_calc.similarity_interpretation(similarity)

        # Build detailed report
        detailed_report = [
            f"📊 Семантический анализ",
            f"═══════════════════════════",
            f"",
            f"▪ Общая схожесть: {similarity * 100:.1f}%",
            f"",
            f"➤ Интерпретация:",
            f"  {base_interpretation}",
            f"",
            f"📋 Детали анализа:",
            f"  • Метод: Косинусное расстояние векторов",
            f"  • Модель: {model_key}",
            f"  • Размерность: {len(emb1)} измерений",
            f"  • Тип: Семантические эмбеддинги",
            f"",
            f"📊 Шкала оценки:",
            f"  • 90-100% - Практически идентичные тексты",
            f"  • 70-89% - Очень похожие по смыслу",
            f"  • 50-69% - Умеренная семантическая схожесть",
            f"  • 30-49% - Слабая связь по смыслу",
            f"  • 0-29% - Тексты практически не связаны"
        ]

        return {
            "similarity": float(similarity),
            "interpretation": "\n".join(detailed_report),
            "model_used": model_key,
            "dimensions": len(emb1)
        }

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "semantic"
