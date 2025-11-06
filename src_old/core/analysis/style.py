"""
Style analysis for text similarity.
Analyzes writing style: sentence length, vocabulary, syntax patterns.
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List
from loguru import logger


class StyleAnalyzer:
    """Analyzes and compares text writing styles."""

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text style features.

        Args:
            text: Input text

        Returns:
            Dictionary of style features
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        words = self._tokenize(text)

        if not words:
            return self._empty_features()

        # Calculate features
        features = {
            # Sentence statistics
            'avg_sentence_length': np.mean([len(self._tokenize(s)) for s in sentences]) if sentences else 0,
            'sentence_length_std': np.std([len(self._tokenize(s)) for s in sentences]) if len(sentences) > 1 else 0,
            'max_sentence_length': max([len(self._tokenize(s)) for s in sentences]) if sentences else 0,
            'min_sentence_length': min([len(self._tokenize(s)) for s in sentences]) if sentences else 0,

            # Word statistics
            'avg_word_length': np.mean([len(w) for w in words]),
            'word_length_std': np.std([len(w) for w in words]) if len(words) > 1 else 0,

            # Vocabulary richness
            'type_token_ratio': len(set(words)) / len(words),  # unique words / total words
            'hapax_legomena_ratio': sum(1 for w, c in Counter(words).items() if c == 1) / len(words),

            # Punctuation
            'comma_ratio': text.count(',') / len(text),
            'period_ratio': text.count('.') / len(text),
            'question_ratio': text.count('?') / len(text),
            'exclamation_ratio': text.count('!') / len(text),
            'quote_ratio': (text.count('"') + text.count('"') + text.count('"')) / len(text),
            'dash_ratio': (text.count('—') + text.count('-')) / len(text),

            # Other
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text),
        }

        return features

    def compare_styles(self, text1: str, text2: str) -> Dict[str, any]:
        """
        Compare styles of two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with similarity score and feature comparison
        """
        features1 = self.analyze_text(text1)
        features2 = self.analyze_text(text2)

        # Calculate distance for each feature (normalized)
        distances = {}
        for key in features1:
            val1 = features1[key]
            val2 = features2[key]

            # Normalize by max value to get 0-1 range
            max_val = max(abs(val1), abs(val2), 1e-6)
            distance = abs(val1 - val2) / max_val
            distances[key] = 1.0 - min(distance, 1.0)  # Convert to similarity

        # Overall style similarity (average of all features)
        style_similarity = np.mean(list(distances.values()))

        return {
            'style_similarity': float(style_similarity),
            'feature_similarities': distances,
            'features1': features1,
            'features2': features2,
            'interpretation': self._interpret_style_similarity(style_similarity)
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w]

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features dict."""
        return {
            'avg_sentence_length': 0,
            'sentence_length_std': 0,
            'max_sentence_length': 0,
            'min_sentence_length': 0,
            'avg_word_length': 0,
            'word_length_std': 0,
            'type_token_ratio': 0,
            'hapax_legomena_ratio': 0,
            'comma_ratio': 0,
            'period_ratio': 0,
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'quote_ratio': 0,
            'dash_ratio': 0,
            'capital_ratio': 0,
            'digit_ratio': 0,
        }

    def _interpret_style_similarity(self, similarity: float) -> str:
        """Interpret style similarity score."""
        if similarity >= 0.9:
            return "Очень похожий стиль (возможно, один автор)"
        elif similarity >= 0.8:
            return "Похожий стиль написания"
        elif similarity >= 0.7:
            return "Умеренно похожий стиль"
        elif similarity >= 0.6:
            return "Немного похожий стиль"
        else:
            return "Различный стиль написания"
