"""
TF-IDF analysis strategy.
"""

from typing import Dict, Any

from server.services.strategies.base import AnalysisStrategy
from server.core.analysis import TFIDFAnalyzer


class TFIDFAnalysisStrategy(AnalysisStrategy):
    """Strategy for TF-IDF analysis."""

    def __init__(self):
        self.analyzer = TFIDFAnalyzer()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform TF-IDF analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters

        Returns:
            Dictionary with similarity, interpretation, top_words
        """
        result = self.analyzer.compare_texts(text1_content, text2_content)

        # Build detailed report
        detailed_report = [
            f"📊 TF-IDF анализ",
            f"═══════════════════════════",
            f"",
            f"▪ Схожесть по ключевым словам: {result['tfidf_similarity'] * 100:.1f}%",
            f"▪ Пересечение словаря: {result['vocabulary_overlap'] * 100:.1f}%",
            f"",
            f"➤ Интерпретация:",
            f"  {result['interpretation']}",
            f"",
            f"📋 Статистика словарей:",
            f"  • Уникальных слов в тексте 1: {result['vocab_size_1']}",
            f"  • Уникальных слов в тексте 2: {result['vocab_size_2']}",
            f"  • Общих слов: {result['shared_vocab']}",
            f"",
            f"▫ Топ-10 ключевых слов текста 1:"
        ]

        # Add top words for text 1
        for i, word_info in enumerate(result['top_words_text1'][:10], 1):
            if isinstance(word_info, dict):
                word, score = word_info['word'], word_info['score']
            else:
                word, score = word_info
            detailed_report.append(f"  {i}. {word} (вес: {score:.3f})")

        detailed_report.append(f"")
        detailed_report.append(f"▫ Топ-10 ключевых слов текста 2:")

        # Add top words for text 2
        for i, word_info in enumerate(result['top_words_text2'][:10], 1):
            if isinstance(word_info, dict):
                word, score = word_info['word'], word_info['score']
            else:
                word, score = word_info
            detailed_report.append(f"  {i}. {word} (вес: {score:.3f})")

        detailed_report.extend([
            f"",
            f"ℹ️ Метод анализа:",
            f"  • TF-IDF (Term Frequency - Inverse Document Frequency)",
            f"  • Выявление важных слов через частотный анализ",
            f"  • Косинусная схожесть TF-IDF векторов"
        ])

        return {
            "similarity": float(result['tfidf_similarity']),
            "interpretation": "\n".join(detailed_report),
            "vocabulary_overlap": result['vocabulary_overlap'],
            "top_words_text1": result['top_words_text1'],
            "top_words_text2": result['top_words_text2'],
            "vocab_size_1": result['vocab_size_1'],
            "vocab_size_2": result['vocab_size_2'],
            "shared_vocab": result['shared_vocab']
        }

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "tfidf"
