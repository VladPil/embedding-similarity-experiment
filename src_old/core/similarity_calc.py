"""
Similarity calculation utilities.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class SimilarityCalculator:
    """Calculate similarity between embeddings."""

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0 to 1)
        """
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)

    @staticmethod
    def similarity_matrix(embeddings: List[np.ndarray], labels: List[str]) -> Dict:
        """
        Calculate pairwise similarity matrix for multiple embeddings.

        Args:
            embeddings: List of embedding vectors
            labels: List of labels for each embedding

        Returns:
            Dictionary with matrix data and labels
        """
        # Stack embeddings
        emb_array = np.vstack(embeddings)

        # Calculate pairwise similarities
        sim_matrix = cosine_similarity(emb_array)

        return {
            'matrix': sim_matrix.tolist(),
            'labels': labels
        }

    @staticmethod
    def chunk_similarity_matrix(
        chunks1: List[np.ndarray],
        chunks2: List[np.ndarray],
        label1: str,
        label2: str
    ) -> Dict:
        """
        Calculate similarity matrix between chunks of two texts.

        Args:
            chunks1: Embeddings of chunks from first text
            chunks2: Embeddings of chunks from second text
            label1: Label for first text
            label2: Label for second text

        Returns:
            Dictionary with matrix data and statistics
        """
        # Stack chunks
        emb1_array = np.vstack(chunks1)
        emb2_array = np.vstack(chunks2)

        # Calculate cross-similarity matrix
        sim_matrix = cosine_similarity(emb1_array, emb2_array)

        # Calculate statistics
        avg_similarity = float(np.mean(sim_matrix))
        max_similarity = float(np.max(sim_matrix))
        min_similarity = float(np.min(sim_matrix))

        # Find best matching chunks
        best_matches = []
        for i in range(len(chunks1)):
            best_j = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_j])
            best_matches.append({
                'chunk_1': i,
                'chunk_2': best_j,
                'similarity': best_score
            })

        return {
            'matrix': sim_matrix.tolist(),
            'label_1': label1,
            'label_2': label2,
            'num_chunks_1': len(chunks1),
            'num_chunks_2': len(chunks2),
            'statistics': {
                'average': avg_similarity,
                'maximum': max_similarity,
                'minimum': min_similarity
            },
            'best_matches': sorted(best_matches, key=lambda x: x['similarity'], reverse=True)[:10]
        }

    @staticmethod
    def format_similarity_percentage(similarity: float) -> str:
        """
        Format similarity score as percentage.

        Args:
            similarity: Similarity score (0 to 1)

        Returns:
            Formatted string
        """
        percentage = similarity * 100
        return f"{percentage:.2f}%"

    @staticmethod
    def similarity_interpretation(similarity: float) -> str:
        """
        Provide interpretation of similarity score.

        Args:
            similarity: Similarity score (0 to 1)

        Returns:
            Interpretation string
        """
        if similarity >= 0.9:
            return "Очень высокое сходство"
        elif similarity >= 0.7:
            return "Высокое сходство"
        elif similarity >= 0.5:
            return "Среднее сходство"
        elif similarity >= 0.3:
            return "Низкое сходство"
        else:
            return "Очень низкое сходство"
