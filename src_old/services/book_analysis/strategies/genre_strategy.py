"""
Genre Analysis Strategy.
Classifies book genre using LLM with optimized prompts.
"""

import json
import re
from typing import Dict, Any
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType
from server.core.analysis.prompt_templates import PromptTemplates
from server.core.analysis.llm_manager import get_llm_manager


class GenreAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for genre classification.

    - Uses LLM with structured prompt
    - Returns JSON with main genre, sub-genres, confidence
    - Fast execution (~5 seconds)
    """

    def __init__(self):
        """Initialize genre analysis strategy."""
        self.prompt_templates = PromptTemplates()

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.GENRE

    def requires_llm(self) -> bool:
        """Genre analysis requires LLM."""
        return True

    def requires_embeddings(self) -> bool:
        """Genre analysis doesn't strictly require embeddings."""
        return False

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate 5 seconds for genre analysis."""
        return 5.0

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute genre analysis.

        Args:
            context: Analysis context with text

        Returns:
            {
                "main_genre": str,
                "sub_genres": List[str],
                "confidence": float,
                "reasoning": str
            }
        """
        try:
            logger.info("Starting genre analysis...")

            # Prepare text (use first 4000 chars for genre determination)
            text_sample = context.text[:4000]

            # Get LLM manager
            llm_manager = await get_llm_manager()

            # Create prompt
            prompt = self.prompt_templates.format_genre_prompt(text_sample)

            # Execute LLM analysis
            result = await llm_manager.execute_task(
                task_type='custom',
                text1=prompt,
                text2=None
            )

            # Parse JSON response
            parsed = self._parse_llm_response(result)

            logger.info(f"Genre analysis complete: {parsed.get('main_genre', 'unknown')}")

            return parsed

        except Exception as e:
            logger.error(f"Genre analysis failed: {e}")
            return {
                "main_genre": "unknown",
                "sub_genres": [],
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {str(e)}",
                "error": str(e)
            }

    def _parse_llm_response(self, llm_response: Dict) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # LLM response might be in different formats
            response_text = ""

            if isinstance(llm_response, dict):
                # Try common keys
                for key in ['response', 'text', 'generated_text', 'content']:
                    if key in llm_response:
                        response_text = llm_response[key]
                        break
                if not response_text:
                    response_text = str(llm_response)
            else:
                response_text = str(llm_response)

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                # Validate required fields
                if 'main_genre' not in parsed:
                    parsed['main_genre'] = 'unknown'
                if 'sub_genres' not in parsed:
                    parsed['sub_genres'] = []
                if 'confidence' not in parsed:
                    parsed['confidence'] = 0.5

                return parsed
            else:
                raise ValueError("No JSON found in LLM response")

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "main_genre": "unknown",
                "sub_genres": [],
                "confidence": 0.0,
                "reasoning": "Failed to parse response",
                "error": str(e)
            }

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret genre analysis results for UI display.

        Args:
            results: Genre analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа жанра: {results['error']}"

        main_genre = results.get('main_genre', 'unknown')
        sub_genres = results.get('sub_genres', [])
        confidence = results.get('confidence', 0)
        reasoning = results.get('reasoning', '')

        if main_genre == 'unknown':
            return "❓ Не удалось определить жанр произведения."

        # Confidence emoji
        if confidence > 0.8:
            conf_emoji = '🎯'
            conf_text = 'высокая'
        elif confidence > 0.5:
            conf_emoji = '✅'
            conf_text = 'средняя'
        else:
            conf_emoji = '⚠️'
            conf_text = 'низкая'

        # Build interpretation
        lines = [
            f"📚 **Жанровый анализ произведения**\n",
            f"{conf_emoji} **Основной жанр**: {main_genre.capitalize()}",
            f"   • Уверенность: {conf_text} ({confidence*100:.0f}%)\n"
        ]

        if sub_genres:
            lines.append(f"🏷️ **Поджанры и элементы**:")
            for sub in sub_genres[:5]:  # Top 5 sub-genres
                lines.append(f"   • {sub.capitalize()}")
            lines.append("")

        if reasoning:
            lines.append(f"💡 **Обоснование**: {reasoning}\n")

        # Genre characteristics
        genre_tips = {
            'фантастика': 'Произведение содержит элементы научной фантастики, технологий будущего или альтернативной реальности.',
            'фэнтези': 'Присутствуют элементы магии, мифологии, или вымышленных миров.',
            'детектив': 'Сюжет построен вокруг расследования, загадки или тайны.',
            'роман': 'Фокус на развитии отношений между персонажами.',
            'триллер': 'Напряженный сюжет с элементами опасности и саспенса.',
            'драма': 'Глубокое исследование человеческих переживаний и конфликтов.',
            'приключения': 'Динамичный сюжет с путешествиями и испытаниями.'
        }

        genre_key = main_genre.lower()
        for key, tip in genre_tips.items():
            if key in genre_key:
                lines.append(f"📖 **Характеристика**: {tip}")
                break

        return '\n'.join(lines)
