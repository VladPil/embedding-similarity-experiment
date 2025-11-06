"""
Theme Analysis Strategy.
Identifies major themes using clustering and LLM.
"""

import json
import re
from typing import Dict, Any, List
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType
from server.core.analysis.prompt_templates import PromptTemplates
from server.core.analysis.llm_manager import get_llm_manager


class ThemeAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for theme analysis.

    - Can use embeddings for clustering (if available)
    - LLM for theme descriptions
    - Returns major themes with examples
    """

    def __init__(self, max_themes: int = 5):
        """
        Initialize theme analysis strategy.

        Args:
            max_themes: Maximum number of themes to extract
        """
        self.prompt_templates = PromptTemplates()
        self.max_themes = max_themes

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.THEME

    def requires_llm(self) -> bool:
        """Theme analysis requires LLM."""
        return True

    def requires_embeddings(self) -> bool:
        """Theme analysis can use embeddings but not required."""
        return False

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate ~15-20 seconds."""
        return 20.0

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute theme analysis.

        Args:
            context: Analysis context

        Returns:
            {
                "themes": List[Theme],
                "total_themes": int
            }
        """
        try:
            logger.info("Starting theme analysis...")

            llm_manager = await get_llm_manager()

            # Sample chunks for theme detection (every 10th chunk)
            sample_chunks = context.chunks[::max(len(context.chunks) // 10, 1)]
            sample_chunks = sample_chunks[:10]  # Max 10 samples

            logger.info(f"Analyzing {len(sample_chunks)} sample chunks for themes")

            # Collect themes from samples
            detected_themes = []

            for chunk in sample_chunks:
                try:
                    prompt = self.prompt_templates.format_theme_prompt(chunk.text)

                    result = await llm_manager.execute_task(
                        task_type='custom',
                        text1=prompt,
                        text2=None
                    )

                    theme_data = self._parse_theme_response(result)

                    if theme_data and theme_data.get('theme'):
                        theme_data['position'] = chunk.position_ratio
                        theme_data['chunk_index'] = chunk.index
                        detected_themes.append(theme_data)

                except Exception as e:
                    logger.warning(f"Failed to analyze theme in chunk {chunk.index}: {e}")
                    continue

            # Aggregate themes
            themes = self._aggregate_themes(detected_themes)

            logger.info(f"Theme analysis complete: {len(themes)} themes found")

            return {
                "themes": themes,
                "total_themes": len(themes)
            }

        except Exception as e:
            logger.error(f"Theme analysis failed: {e}")
            return {
                "themes": [],
                "total_themes": 0,
                "error": str(e)
            }

    def _parse_theme_response(self, llm_response: Dict) -> Dict[str, Any]:
        """Parse LLM response for theme data."""
        try:
            response_text = self._extract_response_text(llm_response)

            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Failed to parse theme response: {e}")
            return {}

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(response, dict):
            for key in ['response', 'text', 'generated_text', 'content']:
                if key in response:
                    return response[key]
            return str(response)
        return str(response)

    def _aggregate_themes(self, detected_themes: List[Dict]) -> List[Dict]:
        """
        Aggregate similar themes and rank by frequency.
        """
        theme_map = {}

        for theme_data in detected_themes:
            theme_name = theme_data.get('theme', '').lower().strip()

            if not theme_name:
                continue

            if theme_name not in theme_map:
                theme_map[theme_name] = {
                    "theme": theme_name,
                    "count": 0,
                    "confidence": [],
                    "examples": []
                }

            theme_info = theme_map[theme_name]
            theme_info['count'] += 1

            if 'confidence' in theme_data:
                theme_info['confidence'].append(theme_data['confidence'])

            if 'evidence' in theme_data:
                theme_info['examples'].append({
                    "position": theme_data.get('position', 0.0),
                    "evidence": theme_data['evidence']
                })

        # Convert to list and calculate weights
        themes = []
        for theme_info in theme_map.values():
            avg_confidence = (
                sum(theme_info['confidence']) / len(theme_info['confidence'])
                if theme_info['confidence'] else 0.5
            )

            # Weight = count * confidence
            weight = theme_info['count'] * avg_confidence

            themes.append({
                "name": theme_info['theme'],
                "weight": round(weight, 2),
                "confidence": round(avg_confidence, 2),
                "frequency": theme_info['count'],
                "examples": theme_info['examples'][:3]  # Top 3 examples
            })

        # Sort by weight
        themes.sort(key=lambda x: x['weight'], reverse=True)

        # Return top themes
        return themes[:self.max_themes]

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret theme analysis results for UI display.

        Args:
            results: Theme analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа тем: {results['error']}"

        themes = results.get('themes', [])
        total_themes = results.get('total_themes', 0)

        if not themes:
            return "📚 Темы не обнаружены в тексте."

        # Build interpretation
        lines = [f"📖 **Анализ тем произведения** ({total_themes} тем обнаружено)\n"]

        for i, theme in enumerate(themes, 1):
            name = theme.get('name', 'Неизвестная тема')
            weight = theme.get('weight', 0)
            confidence = theme.get('confidence', 0)
            frequency = theme.get('frequency', 0)

            # Weight interpretation
            if weight > 4:
                importance = "центральная"
            elif weight > 2:
                importance = "важная"
            else:
                importance = "второстепенная"

            lines.append(
                f"**{i}. {name.capitalize()}** ({importance} тема)\n"
                f"   • Вес: {weight:.1f} | Уверенность: {confidence*100:.0f}% | "
                f"Частота упоминаний: {frequency}\n"
            )

            # Add examples if available
            examples = theme.get('examples', [])
            if examples:
                lines.append(f"   📝 Примеры из текста:")
                for ex in examples[:2]:  # Show top 2 examples
                    position = ex.get('position', 0) * 100
                    evidence = ex.get('evidence', '')[:100]
                    lines.append(f"      - На {position:.0f}% текста: \"{evidence}...\"")
                lines.append("")

        # Summary
        main_themes = [t['name'] for t in themes[:3]]
        lines.append(
            f"\n💡 **Вывод**: Произведение сосредоточено на темах: "
            f"{', '.join(main_themes)}."
        )

        return '\n'.join(lines)
