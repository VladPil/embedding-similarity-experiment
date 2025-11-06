"""
Pace Analysis Strategy.
Analyzes pace without LLM using event density metrics.
"""

from typing import Dict, Any, List
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType
from server.core.analysis.chunk_indexer import ChunkIndexer


class PaceAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for pace analysis.

    - NO LLM required (fast!)
    - Uses event density calculation
    - Returns pace timeline and overall score
    """

    def __init__(self):
        """Initialize pace analysis strategy."""
        self.indexer = ChunkIndexer()

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.PACE

    def requires_llm(self) -> bool:
        """Pace analysis does NOT require LLM."""
        return False

    def requires_embeddings(self) -> bool:
        """Pace analysis doesn't require embeddings."""
        return False

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate 10 seconds regardless of chunk count (fast)."""
        return 10.0

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute pace analysis.

        Args:
            context: Analysis context with chunks

        Returns:
            {
                "overall_pace": str,
                "pace_score": float,
                "timeline": List[PacePoint],
                "statistics": Dict
            }
        """
        try:
            logger.info("Starting pace analysis...")

            # Calculate event density for each chunk
            pace_scores = []

            for chunk in context.chunks:
                # Event density = pace indicator
                density = self.indexer._calculate_event_density(chunk.text)

                # Dialogue ratio
                dialogue_ratio = self.indexer._calculate_dialogue_ratio(chunk.text)

                # Combined pace score (0-10)
                pace_score = self._calculate_pace_score(density, dialogue_ratio)

                pace_scores.append({
                    "position": chunk.position_ratio,
                    "score": pace_score,
                    "event_density": density,
                    "dialogue_ratio": dialogue_ratio
                })

            # Calculate overall pace (median)
            scores_only = [p['score'] for p in pace_scores]
            overall_score = self._median(scores_only)

            # Determine pace rating
            if overall_score < 4:
                pace_rating = "slow"
            elif overall_score < 7:
                pace_rating = "medium"
            else:
                pace_rating = "fast"

            # Create timeline (sample every 10% of book)
            timeline = self._create_timeline(pace_scores, sample_rate=0.1)

            # Calculate statistics
            stats = self._calculate_statistics(pace_scores)

            logger.info(f"Pace analysis complete: {pace_rating} ({overall_score:.1f}/10)")

            return {
                "overall_pace": pace_rating,
                "pace_score": round(overall_score, 2),
                "timeline": timeline,
                "statistics": stats
            }

        except Exception as e:
            logger.error(f"Pace analysis failed: {e}")
            return {
                "overall_pace": "unknown",
                "pace_score": 0.0,
                "timeline": [],
                "statistics": {},
                "error": str(e)
            }

    def _calculate_pace_score(self, event_density: float, dialogue_ratio: float) -> float:
        """
        Calculate pace score from density and dialogue.

        Args:
            event_density: Event density (0-1)
            dialogue_ratio: Dialogue ratio (0-1)

        Returns:
            Pace score (0-10)
        """
        # Event density contributes 70%, dialogue 30%
        score = (event_density * 7.0) + (dialogue_ratio * 3.0)
        return min(score * 10, 10.0)

    def _median(self, values: List[float]) -> float:
        """Calculate median value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

    def _create_timeline(self, pace_scores: List[Dict], sample_rate: float = 0.1) -> List[Dict]:
        """
        Create timeline by sampling pace scores.

        Args:
            pace_scores: All pace scores
            sample_rate: Sampling rate (0.1 = every 10%)

        Returns:
            Sampled timeline points
        """
        if not pace_scores:
            return []

        timeline = []
        n = len(pace_scores)
        step = max(int(n * sample_rate), 1)

        for i in range(0, n, step):
            point = pace_scores[i]
            timeline.append({
                "position": round(point["position"], 2),
                "score": round(point["score"], 1)
            })

        return timeline

    def _calculate_statistics(self, pace_scores: List[Dict]) -> Dict:
        """Calculate pace statistics."""
        if not pace_scores:
            return {}

        scores = [p['score'] for p in pace_scores]
        event_densities = [p['event_density'] for p in pace_scores]
        dialogue_ratios = [p['dialogue_ratio'] for p in pace_scores]

        return {
            "avg_pace_score": round(sum(scores) / len(scores), 2),
            "min_pace": round(min(scores), 2),
            "max_pace": round(max(scores), 2),
            "avg_event_density": round(sum(event_densities) / len(event_densities), 3),
            "avg_dialogue_ratio": round(sum(dialogue_ratios) / len(dialogue_ratios), 3)
        }

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret pace analysis results for UI display.

        Args:
            results: Pace analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа темпа: {results['error']}"

        overall_pace = results.get('overall_pace', 'unknown')
        pace_score = results.get('pace_score', 0)
        timeline = results.get('timeline', [])
        stats = results.get('statistics', {})

        # Pace emoji
        pace_emoji = {
            'slow': '🐌',
            'medium': '🚶',
            'fast': '🏃',
            'unknown': '❓'
        }
        emoji = pace_emoji.get(overall_pace, '❓')

        # Build interpretation
        lines = [
            f"⚡ **Анализ темпа повествования**\n",
            f"{emoji} **Общий темп**: {overall_pace.upper()} ({pace_score:.1f}/10)\n"
        ]

        # Pace interpretation
        if overall_pace == 'slow':
            interpretation = (
                "Медленный темп повествования. Текст сосредоточен на описаниях, "
                "внутренних переживаниях и атмосфере. Подходит для философской прозы "
                "и глубокого погружения в мир произведения."
            )
        elif overall_pace == 'medium':
            interpretation = (
                "Сбалансированный темп. Хорошее соотношение действия и описаний. "
                "Читатель успевает и следить за событиями, и погружаться в атмосферу."
            )
        else:  # fast
            interpretation = (
                "Быстрый темп повествования. Много событий, действий и диалогов. "
                "Динамичный стиль, характерный для жанров экшн, триллер, приключения."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}\n")

        # Statistics
        if stats:
            avg_event = stats.get('avg_event_density', 0)
            avg_dialogue = stats.get('avg_dialogue_ratio', 0)
            min_pace = stats.get('min_pace', 0)
            max_pace = stats.get('max_pace', 0)

            lines.extend([
                f"📊 **Детальная статистика**:",
                f"   • Средняя плотность событий: {avg_event:.3f}",
                f"   • Средний процент диалогов: {avg_dialogue*100:.1f}%",
                f"   • Разброс темпа: {min_pace:.1f} - {max_pace:.1f}",
                f"   • Вариативность: {'высокая' if (max_pace - min_pace) > 5 else 'умеренная'}\n"
            ])

        # Timeline analysis
        if timeline and len(timeline) > 3:
            # Find peaks and valleys
            timeline_scores = [p['score'] for p in timeline]
            avg_timeline = sum(timeline_scores) / len(timeline_scores)

            peaks = [p for p in timeline if p['score'] > avg_timeline + 2]
            valleys = [p for p in timeline if p['score'] < avg_timeline - 2]

            if peaks:
                lines.append(f"📈 **Пики напряжения** (высокий темп):")
                for peak in peaks[:3]:
                    lines.append(f"   • На {peak['position']*100:.0f}% текста (темп: {peak['score']:.1f})")

            if valleys:
                lines.append(f"\n📉 **Спады темпа** (медленные участки):")
                for valley in valleys[:3]:
                    lines.append(f"   • На {valley['position']*100:.0f}% текста (темп: {valley['score']:.1f})")

        return '\n'.join(lines)
