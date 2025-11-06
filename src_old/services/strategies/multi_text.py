"""
Strategy for analyzing 3 or more texts simultaneously.
"""

from typing import Dict, Any, List
import numpy as np
from itertools import combinations

from server.services.strategies.base import AnalysisStrategy
from server.services.embedding_service import EmbeddingService
from server.core.similarity_calc import SimilarityCalculator


class MultiTextAnalysisStrategy(AnalysisStrategy):
    """
    Strategy for analyzing 3 or more texts.

    Creates pairwise similarity matrix and finds:
    - Most similar pair
    - Most different pair
    - Average similarity
    - Cluster analysis
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize multi-text analysis strategy.

        Args:
            embedding_service: Service for embeddings
        """
        self.embedding_service = embedding_service
        self.similarity_calc = SimilarityCalculator()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        This method is not used for multi-text analysis.
        Use analyze_multiple instead.
        """
        raise NotImplementedError(
            "Use analyze_multiple for multi-text analysis"
        )

    async def analyze_multiple(
        self,
        text_contents: List[str],
        text_ids: List[str],
        text_titles: List[str],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze multiple texts (3 or more).

        Args:
            text_contents: List of text contents
            text_ids: List of text IDs
            text_titles: List of text titles
            params: Analysis parameters

        Returns:
            Dictionary with analysis results
        """
        if len(text_contents) < 3:
            raise ValueError("Need at least 3 texts for multi-text analysis")

        model_key = params.get("model", "multilingual-e5-base")
        n_texts = len(text_contents)

        # Get embeddings for all texts
        embeddings = []
        for text_id in text_ids:
            emb = await self.embedding_service.get_embedding(text_id, model_key)
            if emb is None:
                raise ValueError(f"Failed to generate embedding for text {text_id}")
            embeddings.append(emb)

        # Create similarity matrix
        similarity_matrix = np.zeros((n_texts, n_texts))

        for i in range(n_texts):
            for j in range(i, n_texts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.similarity_calc.cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        # Find most similar and most different pairs
        pairwise_similarities = []
        for i, j in combinations(range(n_texts), 2):
            pairwise_similarities.append({
                'text1_index': i,
                'text2_index': j,
                'text1_id': text_ids[i],
                'text2_id': text_ids[j],
                'text1_title': text_titles[i],
                'text2_title': text_titles[j],
                'similarity': float(similarity_matrix[i, j])
            })

        # Sort by similarity
        pairwise_similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Calculate statistics
        similarities = [p['similarity'] for p in pairwise_similarities]
        avg_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))
        min_similarity = float(np.min(similarities))
        max_similarity = float(np.max(similarities))

        # Find clusters (simple approach: texts with similarity > threshold)
        threshold = avg_similarity + 0.1
        clusters = self._find_clusters(similarity_matrix, text_ids, text_titles, threshold)

        # Create result
        result = {
            "n_texts": n_texts,
            "model_used": model_key,
            "similarity_matrix": similarity_matrix.tolist(),
            "text_info": [
                {
                    "index": i,
                    "id": text_ids[i],
                    "title": text_titles[i],
                    "avg_similarity_to_others": float(
                        np.mean([similarity_matrix[i, j] for j in range(n_texts) if j != i])
                    )
                }
                for i in range(n_texts)
            ],
            "pairwise_similarities": pairwise_similarities,
            "most_similar_pair": pairwise_similarities[0],
            "most_different_pair": pairwise_similarities[-1],
            "statistics": {
                "average_similarity": avg_similarity,
                "std_similarity": std_similarity,
                "min_similarity": min_similarity,
                "max_similarity": max_similarity
            },
            "clusters": clusters,
            "interpretation": self._generate_interpretation(
                n_texts,
                avg_similarity,
                pairwise_similarities[0],
                pairwise_similarities[-1],
                clusters
            )
        }

        return result

    def _find_clusters(
        self,
        similarity_matrix: np.ndarray,
        text_ids: List[str],
        text_titles: List[str],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Find clusters of similar texts.

        Args:
            similarity_matrix: Similarity matrix
            text_ids: List of text IDs
            text_titles: List of text titles
            threshold: Similarity threshold for clustering

        Returns:
            List of clusters
        """
        n_texts = len(text_ids)
        assigned = set()
        clusters = []

        for i in range(n_texts):
            if i in assigned:
                continue

            # Find texts similar to text i
            cluster_members = [i]
            for j in range(n_texts):
                if j != i and j not in assigned:
                    if similarity_matrix[i, j] >= threshold:
                        cluster_members.append(j)

            if len(cluster_members) > 1:
                # Found a cluster
                cluster = {
                    "texts": [
                        {
                            "index": idx,
                            "id": text_ids[idx],
                            "title": text_titles[idx]
                        }
                        for idx in cluster_members
                    ],
                    "avg_similarity": float(
                        np.mean([
                            similarity_matrix[i, j]
                            for i in cluster_members
                            for j in cluster_members
                            if i < j
                        ])
                    )
                }
                clusters.append(cluster)
                assigned.update(cluster_members)

        return clusters

    def _generate_interpretation(
        self,
        n_texts: int,
        avg_similarity: float,
        most_similar: Dict[str, Any],
        most_different: Dict[str, Any],
        clusters: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable interpretation.

        Args:
            n_texts: Number of texts
            avg_similarity: Average similarity
            most_similar: Most similar pair
            most_different: Most different pair
            clusters: Found clusters

        Returns:
            Interpretation string
        """
        interpretation = f"Анализ {n_texts} текстов:\n\n"

        # Overall similarity
        if avg_similarity > 0.8:
            interpretation += f"Тексты очень похожи друг на друга (средняя схожесть: {avg_similarity:.1%}).\n"
        elif avg_similarity > 0.6:
            interpretation += f"Тексты умеренно похожи (средняя схожесть: {avg_similarity:.1%}).\n"
        else:
            interpretation += f"Тексты довольно различаются (средняя схожесть: {avg_similarity:.1%}).\n"

        # Most similar pair
        interpretation += f"\nНаиболее похожая пара:\n"
        interpretation += f"  '{most_similar['text1_title']}' и '{most_similar['text2_title']}' "
        interpretation += f"(схожесть: {most_similar['similarity']:.1%})\n"

        # Most different pair
        interpretation += f"\nНаименее похожая пара:\n"
        interpretation += f"  '{most_different['text1_title']}' и '{most_different['text2_title']}' "
        interpretation += f"(схожесть: {most_different['similarity']:.1%})\n"

        # Clusters
        if clusters:
            interpretation += f"\nОбнаружено кластеров: {len(clusters)}\n"
            for i, cluster in enumerate(clusters, 1):
                titles = [t['title'] for t in cluster['texts']]
                interpretation += f"  Кластер {i}: {', '.join(titles)}\n"
                interpretation += f"    Средняя схожесть: {cluster['avg_similarity']:.1%}\n"
        else:
            interpretation += f"\nКластеров не обнаружено - тексты слишком различаются.\n"

        return interpretation

    def get_type(self) -> str:
        """Get analysis type."""
        return "multi_text"
