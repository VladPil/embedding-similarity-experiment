"""
Emotion analysis strategy.
"""

from typing import Dict, Any

from server.services.strategies.base import AnalysisStrategy
from server.core.analysis import EmotionAnalyzer


class EmotionAnalysisStrategy(AnalysisStrategy):
    """Strategy for emotion analysis."""

    def __init__(self):
        self.analyzer = EmotionAnalyzer()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform emotion analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters including num_segments

        Returns:
            Dictionary with similarity, interpretation, emotions
        """
        num_segments = params.get("num_segments", 10)

        result = self.analyzer.compare_emotional_trajectories(
            text1_content, text2_content, num_segments
        )

        # Build detailed report
        detailed_report = [
            f"😊 Эмоциональный анализ",
            f"═══════════════════════════",
            f"",
            f"▪ Общая эмоциональная схожесть: {result['emotional_similarity'] * 100:.1f}%",
            f"",
            f"➤ Интерпретация:",
            f"  {result['interpretation']}",
            f"",
            f"▪ Совпадение доминирующих эмоций: {result['dominant_overlap'] * 100:.1f}%",
            f"",
            f"📊 Схожесть по эмоциям:"
        ]

        # Add emotion-by-emotion comparison
        if 'emotion_similarities' in result:
            emotion_icons = {
                'радость': '😊',
                'грусть': '😢',
                'гнев': '😠',
                'страх': '😨',
                'удивление': '😮',
                'отвращение': '🤢',
                'нейтральность': '😐'
            }
            for emotion, sim in result['emotion_similarities'].items():
                icon = emotion_icons.get(emotion, '•')
                detailed_report.append(f"  • {icon} {emotion.capitalize()}: {sim * 100:.1f}%")

        detailed_report.extend([
            f"",
            f"ℹ️ Метод анализа:",
            f"  • Разбиение текста на {num_segments} сегментов",
            f"  • Анализ эмоций через RoBERTa модель",
            f"  • Сравнение эмоциональных траекторий",
            f"  • 7 базовых эмоций по Экману"
        ])

        return {
            "similarity": float(result['emotional_similarity']),
            "interpretation": "\n".join(detailed_report),
            "dominant_overlap": result['dominant_overlap'],
            "emotion_similarities": result['emotion_similarities'],
            "trajectory1": result['trajectory1'],
            "trajectory2": result['trajectory2']
        }

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "emotion"
