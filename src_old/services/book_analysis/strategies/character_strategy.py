"""
Character Analysis Strategy.
Analyzes characters using selective chunk processing with LLM.
"""

import json
import re
from typing import Dict, Any, List
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType
from server.core.analysis.prompt_templates import PromptTemplates
from server.core.analysis.chunk_indexer import ChunkIndexer
from server.core.analysis.llm_manager import get_llm_manager


class CharacterAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for character analysis.

    - Uses ChunkIndexer to find character-relevant chunks
    - Selective LLM analysis (only 20-30% of chunks)
    - Returns characters with traits, development timeline
    """

    def __init__(self):
        """Initialize character analysis strategy."""
        self.prompt_templates = PromptTemplates()
        self.indexer = ChunkIndexer()

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.CHARACTER

    def requires_llm(self) -> bool:
        """Character analysis requires LLM."""
        return True

    def requires_embeddings(self) -> bool:
        """Character analysis doesn't require embeddings."""
        return False

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate time: ~1 sec per relevant chunk."""
        # Approximately 30% of chunks are character-relevant
        relevant_count = int(chunk_count * 0.3)
        return max(relevant_count * 1.0, 10.0)  # Min 10 seconds

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute character analysis.

        Args:
            context: Analysis context with chunks

        Returns:
            {
                "characters": List[Character],
                "total_characters": int,
                "chunks_analyzed": int,
                "coverage": float
            }
        """
        try:
            logger.info("Starting character analysis...")

            # Build character index
            char_index = self.indexer.build_character_index(context.chunks)

            logger.info(
                f"Character index built: {len(char_index.chunk_indices)} "
                f"chunks selected ({char_index.coverage:.1%})"
            )

            # Get relevant chunks
            relevant_chunks = self.indexer.get_chunk_subset(
                context.chunks,
                'characters'
            )

            if not relevant_chunks:
                logger.warning("No character-relevant chunks found")
                return {
                    "characters": [],
                    "total_characters": 0,
                    "chunks_analyzed": 0,
                    "coverage": 0.0
                }

            # Analyze characters in relevant chunks
            llm_manager = await get_llm_manager()
            character_mentions = []

            # Process chunks (limit to avoid too long processing)
            max_chunks = min(len(relevant_chunks), 30)  # Max 30 chunks

            for i, chunk in enumerate(relevant_chunks[:max_chunks]):
                try:
                    # Get context (previous chunk if available)
                    context_text = ""
                    if i > 0:
                        context_text = relevant_chunks[i-1].text[:500]

                    # Create prompt
                    prompt = self.prompt_templates.format_character_prompt(
                        chunk.text,
                        context_text
                    )

                    # LLM analysis
                    result = await llm_manager.execute_task(
                        task_type='custom',
                        text1=prompt,
                        text2=None
                    )

                    # Parse character data
                    char_data = self._parse_character_response(result)

                    if char_data and char_data.get('name'):
                        char_data['chunk_index'] = chunk.index
                        char_data['position'] = chunk.position_ratio
                        character_mentions.append(char_data)

                except Exception as e:
                    logger.warning(f"Failed to analyze chunk {chunk.index}: {e}")
                    continue

            # Aggregate characters (combine multiple mentions)
            characters = self._aggregate_characters(character_mentions)

            logger.info(f"Character analysis complete: {len(characters)} characters found")

            return {
                "characters": characters,
                "total_characters": len(characters),
                "chunks_analyzed": len(relevant_chunks[:max_chunks]),
                "coverage": char_index.coverage,
                "character_mentions": len(character_mentions)
            }

        except Exception as e:
            logger.error(f"Character analysis failed: {e}")
            return {
                "characters": [],
                "total_characters": 0,
                "chunks_analyzed": 0,
                "coverage": 0.0,
                "error": str(e)
            }

    def _parse_character_response(self, llm_response: Dict) -> Dict[str, Any]:
        """Parse LLM response for character data."""
        try:
            response_text = self._extract_response_text(llm_response)

            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Failed to parse character response: {e}")
            return {}

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(response, dict):
            for key in ['response', 'text', 'generated_text', 'content']:
                if key in response:
                    return response[key]
            return str(response)
        return str(response)

    def _aggregate_characters(self, mentions: List[Dict]) -> List[Dict]:
        """
        Aggregate multiple character mentions into unified profiles.

        Combines mentions of same character across different chunks.
        """
        character_map = {}

        for mention in mentions:
            name = mention.get('name', '').strip()
            if not name or name == "Имя Персонажа":
                continue

            # Normalize name (basic)
            name_key = name.lower()

            if name_key not in character_map:
                character_map[name_key] = {
                    "name": name,
                    "traits": [],
                    "role": mention.get('role', 'episodic'),
                    "appearances": [],
                    "first_appearance": mention.get('position', 0.0)
                }

            char = character_map[name_key]

            # Add traits
            if 'traits' in mention:
                for trait in mention['traits']:
                    if isinstance(trait, dict):
                        char['traits'].append(trait)
                    else:
                        char['traits'].append({"trait": str(trait), "evidence": ""})

            # Add appearance
            char['appearances'].append({
                "position": mention.get('position', 0.0),
                "chunk_index": mention.get('chunk_index', 0)
            })

            # Update role (prefer "main" over others)
            if mention.get('role') == 'main':
                char['role'] = 'main'
            elif mention.get('role') == 'secondary' and char['role'] != 'main':
                char['role'] = 'secondary'

        # Convert to list and deduplicate traits
        characters = []
        for char in character_map.values():
            # Deduplicate traits
            unique_traits = {}
            for trait_dict in char['traits']:
                trait_name = trait_dict.get('trait', '').lower()
                if trait_name and trait_name not in unique_traits:
                    unique_traits[trait_name] = trait_dict

            char['traits'] = list(unique_traits.values())[:5]  # Top 5 traits

            # Sort appearances
            char['appearances'].sort(key=lambda x: x['position'])

            # Create development timeline (simplified)
            if len(char['appearances']) > 1:
                char['development_timeline'] = [
                    {
                        "position": char['first_appearance'],
                        "description": "Первое появление"
                    },
                    {
                        "position": char['appearances'][-1]['position'],
                        "description": "Последнее появление"
                    }
                ]
            else:
                char['development_timeline'] = []

            characters.append(char)

        # Sort by role importance
        role_order = {'main': 0, 'secondary': 1, 'episodic': 2}
        characters.sort(key=lambda x: (role_order.get(x['role'], 3), -len(x['appearances'])))

        return characters

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret character analysis results for UI display.

        Args:
            results: Character analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа персонажей: {results['error']}"

        characters = results.get('characters', [])
        total = results.get('total_characters', 0)
        coverage = results.get('coverage', 0)

        if not characters:
            return "👥 Персонажи не обнаружены в тексте."

        # Build interpretation
        lines = [
            f"👥 **Анализ персонажей произведения**\n",
            f"📊 Обнаружено персонажей: {total} (охват текста: {coverage*100:.1f}%)\n"
        ]

        # Main characters
        main_chars = [c for c in characters if c.get('role') == 'main']
        secondary_chars = [c for c in characters if c.get('role') == 'secondary']
        episodic_chars = [c for c in characters if c.get('role') == 'episodic']

        if main_chars:
            lines.append(f"⭐ **Главные персонажи** ({len(main_chars)}):")
            for char in main_chars[:5]:  # Top 5
                name = char.get('name', 'Неизвестный')
                traits = char.get('traits', [])
                appearances = len(char.get('appearances', []))

                trait_names = [t.get('trait', '') for t in traits[:3]]
                trait_text = ', '.join(trait_names) if trait_names else 'нет данных'

                lines.append(
                    f"\n   **{name}**",
                    f"   • Появлений: {appearances}",
                    f"   • Черты: {trait_text}"
                )

                # Development timeline
                timeline = char.get('development_timeline', [])
                if timeline:
                    first_pos = timeline[0].get('position', 0) * 100
                    last_pos = timeline[-1].get('position', 0) * 100
                    lines.append(f"   • Развитие: {first_pos:.0f}% → {last_pos:.0f}%")

            lines.append("")

        if secondary_chars:
            lines.append(f"👤 **Второстепенные персонажи** ({len(secondary_chars)}):")
            for char in secondary_chars[:3]:
                name = char.get('name', 'Неизвестный')
                appearances = len(char.get('appearances', []))
                lines.append(f"   • {name} ({appearances} появлений)")
            lines.append("")

        if episodic_chars:
            lines.append(
                f"👥 **Эпизодические персонажи**: {len(episodic_chars)}\n"
            )

        # Summary
        if main_chars:
            main_names = [c.get('name', '') for c in main_chars[:3]]
            lines.append(
                f"💡 **Вывод**: Повествование сфокусировано на персонажах: "
                f"{', '.join(main_names)}."
            )

        return '\n'.join(lines)
