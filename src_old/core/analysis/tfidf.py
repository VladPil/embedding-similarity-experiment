"""
TF-IDF based text similarity.
Calculates importance of unique words and vocabulary overlap.
"""

import re
import numpy as np
from typing import Dict, List
from collections import Counter
import math
from loguru import logger


class TFIDFAnalyzer:
    """Analyzes text similarity using TF-IDF (Term Frequency-Inverse Document Frequency)."""

    def __init__(self):
        """Initialize TF-IDF analyzer."""
        self.documents = []
        self.vocabulary = set()
        self.idf_scores = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 2]  # Filter short words

    def _calculate_tf(self, text: str) -> Dict[str, float]:
        """
        Calculate term frequency (TF) for a text.

        TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in d)
        """
        words = self._tokenize(text)
        if not words:
            return {}

        word_counts = Counter(words)
        total_words = len(words)

        tf = {word: count / total_words for word, count in word_counts.items()}
        return tf

    def _calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        """
        Calculate inverse document frequency (IDF) for vocabulary.

        IDF(t, D) = log((N + 1) / (df + 1)) + 1
        (Smoothed IDF to avoid division by zero and log(1) = 0 issues)
        """
        total_docs = len(documents)
        if total_docs == 0:
            return {}

        # Count document frequency for each word
        doc_frequency = Counter()
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                doc_frequency[word] += 1

        # Calculate IDF with smoothing
        idf = {}
        for word, freq in doc_frequency.items():
            # Smoothed IDF formula to prevent log(1) = 0 for identical docs
            idf[word] = math.log((total_docs + 1) / (freq + 1)) + 1

        return idf

    def _calculate_tfidf(self, text: str, idf_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for a text.

        TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
        """
        tf = self._calculate_tf(text)

        tfidf = {}
        for word, tf_score in tf.items():
            idf_score = idf_scores.get(word, 0)
            tfidf[word] = tf_score * idf_score

        return tfidf

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze multiple texts and calculate TF-IDF vectors.

        Args:
            texts: List of texts to analyze

        Returns:
            List of TF-IDF dictionaries (one per text)
        """
        # Calculate IDF across all documents
        idf_scores = self._calculate_idf(texts)

        # Calculate TF-IDF for each document
        tfidf_vectors = []
        for text in texts:
            tfidf = self._calculate_tfidf(text, idf_scores)
            tfidf_vectors.append(tfidf)

        return tfidf_vectors

    def cosine_similarity(self, tfidf1: Dict[str, float], tfidf2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors.

        Args:
            tfidf1: First TF-IDF vector
            tfidf2: Second TF-IDF vector

        Returns:
            Similarity score (0-1)
        """
        # Get all unique words
        all_words = set(tfidf1.keys()) | set(tfidf2.keys())

        if not all_words:
            return 0.0

        # Create vectors
        vec1 = np.array([tfidf1.get(word, 0) for word in all_words])
        vec2 = np.array([tfidf2.get(word, 0) for word in all_words])

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def compare_texts(self, text1: str, text2: str) -> Dict[str, any]:
        """
        Compare two texts using TF-IDF.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with similarity metrics and top unique words
        """
        # Calculate TF-IDF for both texts
        tfidf_vectors = self.analyze_texts([text1, text2])
        tfidf1, tfidf2 = tfidf_vectors[0], tfidf_vectors[1]

        # Calculate similarity
        similarity = self.cosine_similarity(tfidf1, tfidf2)

        # Find top unique words for each text
        top_words1 = sorted(tfidf1.items(), key=lambda x: x[1], reverse=True)[:20]
        top_words2 = sorted(tfidf2.items(), key=lambda x: x[1], reverse=True)[:20]

        # Calculate vocabulary overlap
        vocab1 = set(self._tokenize(text1))
        vocab2 = set(self._tokenize(text2))
        overlap = len(vocab1 & vocab2) / len(vocab1 | vocab2) if vocab1 or vocab2 else 0

        return {
            'tfidf_similarity': float(similarity),
            'vocabulary_overlap': float(overlap),
            'top_words_text1': [{'word': w, 'score': float(s)} for w, s in top_words1],
            'top_words_text2': [{'word': w, 'score': float(s)} for w, s in top_words2],
            'vocab_size_1': len(vocab1),
            'vocab_size_2': len(vocab2),
            'shared_vocab': len(vocab1 & vocab2),
            'interpretation': self._interpret_tfidf_similarity(similarity, overlap)
        }

    def _interpret_tfidf_similarity(self, similarity: float, overlap: float) -> str:
        """Interpret TF-IDF similarity score."""
        if similarity >= 0.7 and overlap >= 0.5:
            return "Очень похожая уникальная лексика (возможно, один жанр/автор)"
        elif similarity >= 0.5:
            return "Похожие ключевые слова и темы"
        elif similarity >= 0.3:
            return "Частично совпадающая лексика"
        elif similarity >= 0.15:
            return "Небольшое пересечение тем"
        else:
            return "Разная уникальная лексика и темы"
