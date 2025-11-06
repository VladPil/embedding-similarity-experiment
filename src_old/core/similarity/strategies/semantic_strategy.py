"""
Semantic Similarity Strategy.
Advanced semantic comparison using clustering and topic modeling.
"""

import numpy as np
from typing import Dict, Any, List
from loguru import logger
from sklearn.cluster import KMeans
from collections import Counter

from server.core.similarity.base import (
    ISimilarityStrategy,
    SimilarityContext,
    SimilarityResult,
    SimilarityMethod,
    SimilarityScope
)


class SemanticSimilarityStrategy(ISimilarityStrategy):
    """
    Semantic similarity strategy using clustering.

    - Clusters embeddings to find topics
    - Compares topic distributions
    - More robust than simple cosine
    - Captures thematic similarity
    """

    def __init__(
        self,
        scope: SimilarityScope = SimilarityScope.THEMATIC,
        n_clusters: int = 10
    ):
        """
        Initialize semantic similarity strategy.

        Args:
            scope: Similarity scope
            n_clusters: Number of clusters for topic extraction
        """
        self.scope = scope
        self.n_clusters = n_clusters

    def get_method(self) -> SimilarityMethod:
        """Get similarity method identifier."""
        return SimilarityMethod.SEMANTIC

    def get_scope(self) -> SimilarityScope:
        """Get similarity scope."""
        return self.scope

    def requires_embeddings(self) -> bool:
        """Semantic similarity requires embeddings."""
        return True

    def get_estimated_time(self) -> float:
        """Estimate 5 seconds for semantic similarity."""
        return 5.0

    async def calculate(self, context: SimilarityContext) -> SimilarityResult:
        """
        Calculate semantic similarity using topic clustering.

        Args:
            context: Similarity context with embeddings

        Returns:
            Similarity result
        """
        try:
            logger.info(f"Calculating semantic similarity (scope: {self.scope.value})")

            if not context.embeddings1 or not context.embeddings2:
                raise ValueError("Embeddings required for semantic similarity")

            # Cluster embeddings to extract topics
            topics1 = self._extract_topics(context.embeddings1)
            topics2 = self._extract_topics(context.embeddings2)

            # Compare topic distributions
            score, details = self._compare_topics(topics1, topics2)

            interpretation = self.interpret_score(score)

            logger.info(f"Semantic similarity calculated: {score:.3f}")

            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=score,
                confidence=0.85,  # Good confidence for semantic
                details=details,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                interpretation="Ошибка вычисления семантической схожести"
            )

    def _extract_topics(self, embeddings: list) -> Dict[str, Any]:
        """
        Extract topics from embeddings using clustering.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary with topic information
        """
        emb_array = np.array(embeddings)

        # Adjust cluster count if needed
        n_clusters = min(self.n_clusters, len(embeddings))

        if n_clusters < 2:
            # Not enough data for clustering
            return {
                "distribution": [1.0],
                "cluster_count": 1,
                "centroids": [np.mean(emb_array, axis=0)]
            }

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_array)

        # Calculate topic distribution
        label_counts = Counter(labels)
        total = len(labels)
        distribution = [label_counts[i] / total for i in range(n_clusters)]

        return {
            "distribution": distribution,
            "cluster_count": n_clusters,
            "centroids": kmeans.cluster_centers_,
            "labels": labels
        }

    def _compare_topics(
        self,
        topics1: Dict[str, Any],
        topics2: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """
        Compare topic distributions between two texts.

        Uses:
        1. Distribution similarity (KL divergence / Jensen-Shannon)
        2. Centroid alignment (how similar are the topics themselves)
        3. Overlap score (shared topics)

        Returns:
            Tuple of (score, details)
        """
        # 1. Distribution similarity using Hellinger distance
        dist1 = np.array(topics1["distribution"])
        dist2 = np.array(topics2["distribution"])

        # Pad distributions to same length
        max_len = max(len(dist1), len(dist2))
        if len(dist1) < max_len:
            dist1 = np.pad(dist1, (0, max_len - len(dist1)))
        if len(dist2) < max_len:
            dist2 = np.pad(dist2, (0, max_len - len(dist2)))

        # Hellinger distance (0 = identical, 1 = completely different)
        hellinger = np.sqrt(np.sum((np.sqrt(dist1) - np.sqrt(dist2)) ** 2)) / np.sqrt(2)
        dist_similarity = 1.0 - hellinger

        # 2. Centroid alignment using Hungarian algorithm (simplified)
        centroids1 = topics1["centroids"]
        centroids2 = topics2["centroids"]

        # Create similarity matrix between centroids
        n1, n2 = len(centroids1), len(centroids2)
        centroid_sim_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                sim = self._cosine_similarity(centroids1[i], centroids2[j])
                centroid_sim_matrix[i, j] = (sim + 1) / 2  # Normalize to 0-1

        # Average of best matches
        if n1 > 0 and n2 > 0:
            max_sim_1 = np.max(centroid_sim_matrix, axis=1).mean()
            max_sim_2 = np.max(centroid_sim_matrix, axis=0).mean()
            centroid_similarity = (max_sim_1 + max_sim_2) / 2
        else:
            centroid_similarity = 0.0

        # 3. Overlap score (Jaccard-like for topics)
        # Find matching centroids (similarity > 0.7)
        matching_topics = 0
        for i in range(n1):
            for j in range(n2):
                if centroid_sim_matrix[i, j] > 0.7:
                    matching_topics += 1
                    break  # Count each topic1 only once

        overlap_score = matching_topics / max(n1, n2) if max(n1, n2) > 0 else 0.0

        # Combined score (weighted average)
        # Distribution: 30%, Centroid: 50%, Overlap: 20%
        final_score = (
            0.3 * dist_similarity +
            0.5 * centroid_similarity +
            0.2 * overlap_score
        )

        details = {
            "method": "topic_clustering",
            "topics_text1": n1,
            "topics_text2": n2,
            "distribution_similarity": float(dist_similarity),
            "centroid_similarity": float(centroid_similarity),
            "overlap_score": float(overlap_score),
            "matching_topics": matching_topics,
            "topic_distribution1": [float(x) for x in topics1["distribution"]],
            "topic_distribution2": [float(x) for x in topics2["distribution"]],
            "weights": {
                "distribution": 0.3,
                "centroid": 0.5,
                "overlap": 0.2
            }
        }

        return final_score, details

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def interpret_score(self, score: float) -> str:
        """
        Interpret semantic similarity score.

        Args:
            score: Similarity score (0.0 to 1.0)

        Returns:
            Human-readable interpretation
        """
        if score > 0.85:
            return "Очень высокая семантическая схожесть - тексты раскрывают одни и те же темы"
        elif score > 0.7:
            return "Высокая семантическая схожесть - тексты имеют много общих тем"
        elif score > 0.5:
            return "Умеренная семантическая схожесть - тексты частично пересекаются тематически"
        elif score > 0.3:
            return "Низкая семантическая схожесть - тексты затрагивают разные темы"
        else:
            return "Очень низкая семантическая схожесть - тексты о совершенно разном"
