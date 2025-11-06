"""
Tension Analysis Strategy.
Analyzes tension timeline using selective chunk processing.
"""

import json
import re
from typing import Dict, Any, List
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType
from server.core.analysis.prompt_templates import PromptTemplates
from server.core.analysis.chunk_indexer import ChunkIndexer
from server.core.analysis.llm_manager import get_llm_manager


class TensionAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for tension analysis.

    - Finds high-tension chunks using indexer
    - LLM analyzes only tension peaks
    - Returns tension timeline with descriptions
    """

    def __init__(self, tension_threshold: float = 6.0):
        """
        Initialize tension analysis strategy.

        Args:
            tension_threshold: Minimum tension score to analyze (0-10)
        """
        self.prompt_templates = PromptTemplates()
        self.indexer = ChunkIndexer()
        self.tension_threshold = tension_threshold

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.TENSION

    def requires_llm(self) -> bool:
        """Tension analysis requires LLM for peak descriptions."""
        return True

    def requires_embeddings(self) -> bool:
        """Tension analysis can use embeddings but doesn't require them."""
        return False

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate time: ~1.5 sec per tension peak."""
        # Approximately 10-15% of chunks are high-tension
        peak_count = int(chunk_count * 0.15)
        return max(peak_count * 1.5, 10.0)  # Min 10 seconds

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute tension analysis.

        Args:
            context: Analysis context with chunks

        Returns:
            {
                "average_tension": float,
                "timeline": List[TensionPoint],
                "peaks": List[int],
                "analysis_coverage": float
            }
        """
        try:
            logger.info("Starting tension analysis...")

            # Build tension index
            tension_index = self.indexer.build_tension_index(
                context.chunks,
                threshold=self.tension_threshold
            )

            logger.info(
                f"Tension index built: {len(tension_index.chunk_indices)} "
                f"high-tension chunks ({tension_index.coverage:.1%})"
            )

            # Calculate average tension from all chunks
            all_scores = [
                self.indexer._calculate_tension_from_keywords(chunk.text)
                for chunk in context.chunks
            ]
            average_tension = sum(all_scores) / len(all_scores) if all_scores else 0.0

            # Get high-tension chunks
            tension_chunks = self.indexer.get_chunk_subset(
                context.chunks,
                'tension'
            )

            if not tension_chunks:
                logger.info("No high-tension chunks found")
                return {
                    "average_tension": average_tension,
                    "timeline": [],
                    "peaks": [],
                    "analysis_coverage": 0.0
                }

            # Analyze tension peaks with LLM
            llm_manager = await get_llm_manager()
            tension_points = []

            # Limit to top 15 peaks
            max_peaks = min(len(tension_chunks), 15)

            for i, chunk in enumerate(tension_chunks[:max_peaks]):
                try:
                    # Create prompt
                    prompt = self.prompt_templates.format_tension_prompt(chunk.text)

                    # LLM analysis
                    result = await llm_manager.execute_task(
                        task_type='custom',
                        text1=prompt,
                        text2=None
                    )

                    # Parse tension data
                    tension_data = self._parse_tension_response(result)

                    if tension_data:
                        # Get score from index
                        chunk_idx_in_index = tension_index.chunk_indices.index(chunk.index)
                        score = tension_index.scores[chunk_idx_in_index]

                        tension_point = {
                            "position": chunk.position_ratio,
                            "score": score,
                            "source": tension_data.get('source', 'unknown'),
                            "description": tension_data.get('description', ''),
                            "excerpt": tension_data.get('excerpt', chunk.text[:200])
                        }

                        tension_points.append(tension_point)

                except Exception as e:
                    logger.warning(f"Failed to analyze tension chunk {chunk.index}: {e}")
                    continue

            # Sort by position
            tension_points.sort(key=lambda x: x['position'])

            logger.info(f"Tension analysis complete: {len(tension_points)} peaks analyzed")

            return {
                "average_tension": round(average_tension, 2),
                "timeline": tension_points,
                "peaks": [p['position'] for p in tension_points],
                "peak_count": len(tension_points),
                "analysis_coverage": tension_index.coverage
            }

        except Exception as e:
            logger.error(f"Tension analysis failed: {e}")
            return {
                "average_tension": 0.0,
                "timeline": [],
                "peaks": [],
                "peak_count": 0,
                "error": str(e)
            }

    def _parse_tension_response(self, llm_response: Dict) -> Dict[str, Any]:
        """Parse LLM response for tension data."""
        try:
            response_text = self._extract_response_text(llm_response)

            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Failed to parse tension response: {e}")
            return {}

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(response, dict):
            for key in ['response', 'text', 'generated_text', 'content']:
                if key in response:
                    return response[key]
            return str(response)
        return str(response)

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret tension analysis results for UI display.

        Args:
            results: Tension analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа напряжения: {results['error']}"

        avg_tension = results.get('average_tension', 0)
        timeline = results.get('timeline', [])
        peaks = results.get('peaks', [])
        peak_count = results.get('peak_count', 0)

        # Tension emoji based on average
        if avg_tension > 7:
            tension_emoji = '🔥'
            tension_level = 'очень высокое'
        elif avg_tension > 5:
            tension_emoji = '⚡'
            tension_level = 'высокое'
        elif avg_tension > 3:
            tension_emoji = '📊'
            tension_level = 'умеренное'
        else:
            tension_emoji = '🌊'
            tension_level = 'низкое'

        # Build interpretation
        lines = [
            f"🎭 **Анализ напряжения в произведении**\n",
            f"{tension_emoji} **Среднее напряжение**: {tension_level} ({avg_tension:.1f}/10)",
            f"📈 **Обнаружено пиков напряжения**: {peak_count}\n"
        ]

        # Interpretation based on tension level
        if avg_tension > 7:
            interpretation = (
                "Высоконапряженное повествование. Текст насыщен конфликтами, "
                "опасностями и эмоциональными моментами. Характерно для триллеров и экшн-жанров."
            )
        elif avg_tension > 5:
            interpretation = (
                "Умеренно напряженное повествование. Хороший баланс спокойных "
                "и драматичных моментов, поддерживающий интерес читателя."
            )
        elif avg_tension > 3:
            interpretation = (
                "Спокойное повествование с редкими всплесками напряжения. "
                "Фокус на атмосфере, размышлениях и развитии персонажей."
            )
        else:
            interpretation = (
                "Очень спокойное повествование. Минимум конфликтов и драматических ситуаций. "
                "Характерно для медитативной, философской прозы."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}\n")

        # Timeline analysis
        if timeline:
            lines.append(f"📊 **Динамика напряжения по ходу повествования**:\n")

            # Group peaks by position
            if len(timeline) > 5:
                lines.append("   Основные пики напряжения:")
                for i, point in enumerate(timeline[:5], 1):
                    position = point.get('position', 0) * 100
                    score = point.get('score', 0)
                    source = point.get('source', 'unknown')
                    description = point.get('description', '')[:80]

                    lines.append(
                        f"\n   {i}. На {position:.0f}% текста (напряжение: {score:.1f}/10)",
                        f"      Источник: {source}",
                        f"      {description}..."
                    )
            else:
                for point in timeline:
                    position = point.get('position', 0) * 100
                    score = point.get('score', 0)
                    source = point.get('source', 'unknown')

                    lines.append(
                        f"   • {position:.0f}%: {source} ({score:.1f}/10)"
                    )

            lines.append("")

        # Pacing analysis
        if peaks and len(peaks) > 2:
            # Calculate distribution
            first_half_peaks = sum(1 for p in peaks if p < 0.5)
            second_half_peaks = len(peaks) - first_half_peaks

            lines.append(f"📌 **Распределение пиков**:")
            lines.append(f"   • Первая половина: {first_half_peaks} пиков")
            lines.append(f"   • Вторая половина: {second_half_peaks} пиков")

            if second_half_peaks > first_half_peaks * 1.5:
                lines.append("   💡 Напряжение нарастает к финалу")
            elif first_half_peaks > second_half_peaks * 1.5:
                lines.append("   💡 Основные конфликты в начале произведения")
            else:
                lines.append("   💡 Равномерное распределение напряжения")

        return '\n'.join(lines)
