"""
Style analysis strategy.
"""

from typing import Dict, Any

from server.services.strategies.base import AnalysisStrategy
from server.core.analysis import StyleAnalyzer


class StyleAnalysisStrategy(AnalysisStrategy):
    """Strategy for style analysis."""

    def __init__(self):
        self.analyzer = StyleAnalyzer()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform style analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters

        Returns:
            Dictionary with similarity, interpretation, features
        """
        result = self.analyzer.compare_styles(text1_content, text2_content)

        # Build detailed report
        detailed_report = [
            f"✍️ Стилистический анализ",
            f"═══════════════════════════",
            f"",
            f"▪ Общая схожесть стиля: {result['style_similarity'] * 100:.1f}%",
            f"",
            f"➤ Интерпретация:",
            f"  {result['interpretation']}",
            f"",
            f"📊 Сравнение стилистических характеристик:"
        ]

        # Add feature comparison - группируем по типам
        if 'feature_similarities' in result:
            # Словарь перевода
            feature_names = {
                # Статистика предложений
                'avg_sentence_length': 'Средняя длина предложения',
                'sentence_length_std': 'Вариативность длины предложений',
                'max_sentence_length': 'Максимальная длина предложения',
                'min_sentence_length': 'Минимальная длина предложения',

                # Статистика слов
                'avg_word_length': 'Средняя длина слова',
                'word_length_std': 'Вариативность длины слов',

                # Богатство словаря
                'type_token_ratio': 'Разнообразие лексики',
                'hapax_legomena_ratio': 'Уникальные слова',

                # Пунктуация
                'comma_ratio': 'Использование запятых',
                'period_ratio': 'Использование точек',
                'question_ratio': 'Вопросительные предложения',
                'exclamation_ratio': 'Восклицательные предложения',
                'quote_ratio': 'Использование кавычек',
                'dash_ratio': 'Использование тире',

                # Другое
                'capital_ratio': 'Заглавные буквы',
                'digit_ratio': 'Использование цифр'
            }

            # Группируем метрики по категориям
            detailed_report.append("")
            detailed_report.append("▫ Структура предложений:")
            for feature in ['avg_sentence_length', 'sentence_length_std', 'max_sentence_length', 'min_sentence_length']:
                if feature in result['feature_similarities']:
                    sim = result['feature_similarities'][feature]
                    name = feature_names.get(feature, feature)
                    detailed_report.append(f"    • {name}: {sim * 100:.1f}%")

            detailed_report.append("")
            detailed_report.append("▫ Лексические особенности:")
            for feature in ['avg_word_length', 'word_length_std', 'type_token_ratio', 'hapax_legomena_ratio']:
                if feature in result['feature_similarities']:
                    sim = result['feature_similarities'][feature]
                    name = feature_names.get(feature, feature)
                    detailed_report.append(f"    • {name}: {sim * 100:.1f}%")

            detailed_report.append("")
            detailed_report.append("▫ Пунктуация:")
            for feature in ['comma_ratio', 'period_ratio', 'question_ratio', 'exclamation_ratio', 'quote_ratio', 'dash_ratio']:
                if feature in result['feature_similarities']:
                    sim = result['feature_similarities'][feature]
                    name = feature_names.get(feature, feature)
                    detailed_report.append(f"    • {name}: {sim * 100:.1f}%")

            detailed_report.append("")
            detailed_report.append("▫ Прочие характеристики:")
            for feature in ['capital_ratio', 'digit_ratio']:
                if feature in result['feature_similarities']:
                    sim = result['feature_similarities'][feature]
                    name = feature_names.get(feature, feature)
                    detailed_report.append(f"    • {name}: {sim * 100:.1f}%")

        detailed_report.extend([
            f"",
            f"ℹ️ Методология анализа:",
            f"  • Извлечение 16 стилистических признаков",
            f"  • Статистический анализ структуры текста",
            f"  • Нормализация и взвешенное сравнение метрик",
            f"  • Комплексная оценка авторского стиля"
        ])

        return {
            "similarity": float(result['style_similarity']),
            "interpretation": "\n".join(detailed_report),
            "features": {
                "text1": result['features1'],
                "text2": result['features2']
            },
            "feature_similarities": result['feature_similarities']
        }

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "style"
