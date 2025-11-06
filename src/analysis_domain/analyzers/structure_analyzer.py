"""
Анализатор структуры и композиции текста с использованием NLP
"""
import re
import json
import math
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class StructureAnalyzer(BaseAnalyzer):
    """
    Анализатор структуры и композиции текста

    Анализирует структурные элементы текста:
    - Композиционная структура и организация
    - Логические связи и переходы
    - Ритм и темп повествования
    - Абзацная структура и сегментация
    - Информационная архитектура
    - Нарративные паттерны
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        super().__init__(
            name="StructureAnalyzer",
            description="Анализирует структуру, композицию и организацию текста"
        )
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._get_default_prompt()

        # Маркеры структурных элементов
        self.structural_markers = {
            'введение': ['введение', 'предисловие', 'пролог', 'начало', 'во-первых', 'прежде всего'],
            'развитие': ['далее', 'затем', 'потом', 'после этого', 'во-вторых', 'в-третьих', 'кроме того'],
            'кульминация': ['наконец', 'в конце концов', 'главное', 'самое важное', 'кульминация'],
            'заключение': ['заключение', 'в заключение', 'итак', 'таким образом', 'в итоге', 'напоследок']
        }

        # Связующие элементы
        self.connective_words = {
            'причина': ['потому что', 'поскольку', 'так как', 'из-за', 'вследствие', 'благодаря'],
            'следствие': ['поэтому', 'следовательно', 'в результате', 'таким образом', 'отсюда'],
            'противопоставление': ['но', 'однако', 'а', 'зато', 'тем не менее', 'напротив', 'все же'],
            'дополнение': ['также', 'кроме того', 'более того', 'к тому же', 'помимо этого'],
            'сравнение': ['как', 'подобно', 'словно', 'будто', 'по сравнению с', 'аналогично'],
            'уступка': ['хотя', 'несмотря на', 'пусть', 'даже если', 'правда']
        }

        # Темпоральные маркеры
        self.temporal_markers = {
            'одновременность': ['в то же время', 'одновременно', 'параллельно', 'в тот момент'],
            'последовательность': ['сначала', 'потом', 'затем', 'после', 'позже', 'впоследствии'],
            'предшествование': ['до того как', 'прежде чем', 'ранее', 'перед тем как'],
            'внезапность': ['вдруг', 'неожиданно', 'внезапно', 'резко', 'мгновенно']
        }

        # Паттерны повествования
        self.narrative_patterns = {
            'вопрос': [r'\?'],
            'восклицание': [r'!'],
            'прямая_речь': [r'[—–]\s*[А-Яа-я]', r'["«][^"»]*["»]'],
            'перечисление': [r'во-первых|во-вторых|в-третьих', r'\d+\)', r'•|·|\*'],
            'цитирование': [r'как сказал|по словам|цитируя|согласно']
        }

    def _get_default_prompt(self) -> PromptTemplate:
        """Получить промпт по умолчанию для LLM анализа"""
        template = """
        Проанализируй структуру и композицию данного текста:
        1. Логичность построения и организации
        2. Качество переходов между частями
        3. Общую архитектуру текста
        4. Соответствие структуры содержанию

        Текст: {text}

        Верни результат в формате JSON:
        {{
            "structure_quality": float (0-1),
            "logical_flow": float (0-1),
            "coherence_score": float (0-1),
            "composition_type": "linear|circular|parallel|mosaic|other",
            "structural_elements": ["список обнаруженных структурных элементов"],
            "recommendations": ["рекомендации по улучшению структуры"]
        }}
        """
        return PromptTemplate(
            template=template,
            variables=["text"],
            name="structure_analysis_prompt"
        )

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        **kwargs
    ) -> AnalysisResult:
        """
        Анализ структуры текста

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа структуры
        """
        try:
            logger.info(f"Начинаем анализ структуры текста: {text.title}")

            if mode == AnalysisMode.CHUNKED:
                return await self._analyze_chunks(text, **kwargs)
            else:
                return await self._analyze_full_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка анализа структуры: {e}")
            raise AnalysisError(f"Не удалось проанализировать структуру текста: {e}")

    async def _analyze_full_text(self, text: BaseText, **kwargs) -> AnalysisResult:
        """Анализ полного текста"""
        content = text.content

        if not content.strip():
            return AnalysisResult(
            text_id=text.id,
                analyzer_name=self.name,
                data={"error": "Пустой текст"},
                metadata={"mode": "full_text"}
            )

        # Базовый структурный анализ
        basic_structure = self._analyze_basic_structure(content)

        # Анализ абзацной структуры
        paragraph_analysis = self._analyze_paragraph_structure(content)

        # Анализ логических связей
        logical_analysis = self._analyze_logical_connections(content)

        # Анализ композиционных элементов
        composition_analysis = self._analyze_composition(content)

        # Анализ темпа и ритма
        rhythm_analysis = self._analyze_text_rhythm(content)

        # Анализ информационной плотности
        information_analysis = self._analyze_information_density(content)

        # Анализ нарративных паттернов
        narrative_analysis = self._analyze_narrative_patterns(content)

        # Объединение результатов
        all_metrics = {
            **basic_structure,
            **paragraph_analysis,
            **logical_analysis,
            **composition_analysis,
            **rhythm_analysis,
            **information_analysis,
            **narrative_analysis
        }

        # Общая оценка качества структуры
        overall_score = self._calculate_overall_structure_score(all_metrics)
        all_metrics["overall_structure_quality"] = overall_score
        all_metrics["structure_grade"] = self._get_structure_grade(overall_score)

        # Рекомендации
        all_metrics["automated_recommendations"] = self._generate_structure_recommendations(all_metrics)

        # LLM анализ если доступен
        if self.llm_service:
            llm_analysis = await self._get_llm_structure_analysis(content[:2000])
            all_metrics.update(llm_analysis)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data=all_metrics,
            metadata={
                "mode": "full_text",
                "text_length": len(content)
            }
        )

    def _analyze_basic_structure(self, text: str) -> Dict[str, Any]:
        """Базовый анализ структуры текста"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = self._split_sentences(text)

        # Базовые метрики
        total_chars = len(text)
        total_words = len(text.split())

        # Структурные пропорции
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            paragraph_length_variance = self._calculate_variance([len(p.split()) for p in paragraphs])
        else:
            avg_paragraph_length = 0
            paragraph_length_variance = 0

        if sentences:
            avg_sentence_length = total_words / len(sentences)
            sentence_length_variance = self._calculate_variance([len(s.split()) for s in sentences])
        else:
            avg_sentence_length = 0
            sentence_length_variance = 0

        # Структурная регулярность
        structure_regularity = self._calculate_structure_regularity(paragraphs)

        return {
            "total_paragraphs": len(paragraphs),
            "total_sentences": len(sentences),
            "avg_paragraph_length": round(avg_paragraph_length, 1),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "paragraph_length_variance": round(paragraph_length_variance, 2),
            "sentence_length_variance": round(sentence_length_variance, 2),
            "structure_regularity": round(structure_regularity, 3),
            "words_per_sentence": round(avg_sentence_length, 1),
            "sentences_per_paragraph": round(len(sentences) / max(len(paragraphs), 1), 1)
        }

    def _analyze_paragraph_structure(self, text: str) -> Dict[str, Any]:
        """Анализ абзацной структуры"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            return {"paragraph_structure_score": 0}

        # Длины абзацев в словах
        paragraph_lengths = [len(p.split()) for p in paragraphs]

        # Анализ распределения длин
        short_paragraphs = sum(1 for length in paragraph_lengths if length < 50)
        medium_paragraphs = sum(1 for length in paragraph_lengths if 50 <= length <= 150)
        long_paragraphs = sum(1 for length in paragraph_lengths if length > 150)

        total = len(paragraphs)
        if total > 0:
            short_ratio = short_paragraphs / total
            medium_ratio = medium_paragraphs / total
            long_ratio = long_paragraphs / total
        else:
            short_ratio = medium_ratio = long_ratio = 0

        # Оптимальное распределение: больше средних, меньше крайностей
        balance_score = medium_ratio - (short_ratio * 0.5) - (long_ratio * 0.3)
        balance_score = max(0, balance_score)

        # Переходы между абзацами
        transition_quality = self._analyze_paragraph_transitions(paragraphs)

        return {
            "paragraph_structure_score": round((balance_score + transition_quality) / 2, 3),
            "short_paragraphs_ratio": round(short_ratio, 3),
            "medium_paragraphs_ratio": round(medium_ratio, 3),
            "long_paragraphs_ratio": round(long_ratio, 3),
            "transition_quality": round(transition_quality, 3),
            "paragraph_balance_score": round(balance_score, 3)
        }

    def _analyze_paragraph_transitions(self, paragraphs: List[str]) -> float:
        """Анализ качества переходов между абзацами"""
        if len(paragraphs) < 2:
            return 1.0

        transition_score = 0
        total_transitions = len(paragraphs) - 1

        for i in range(len(paragraphs) - 1):
            current_para = paragraphs[i].lower()
            next_para = paragraphs[i + 1].lower()

            # Анализируем последнее предложение текущего и первое следующего
            current_sentences = self._split_sentences(current_para)
            next_sentences = self._split_sentences(next_para)

            if current_sentences and next_sentences:
                last_sentence = current_sentences[-1].lower()
                first_sentence = next_sentences[0].lower()

                # Проверяем связующие элементы
                transition_score += self._score_transition(last_sentence, first_sentence)

        return transition_score / total_transitions if total_transitions > 0 else 0

    def _score_transition(self, last_sentence: str, first_sentence: str) -> float:
        """Оценка качества перехода между предложениями"""
        score = 0.5  # Базовая оценка

        # Проверяем наличие связующих слов в первом предложении следующего абзаца
        for category, connectives in self.connective_words.items():
            for connective in connectives:
                if first_sentence.startswith(connective) or f" {connective} " in first_sentence:
                    score += 0.3
                    break

        # Проверяем тематическую связь (простая проверка общих слов)
        last_words = set(last_sentence.split())
        first_words = set(first_sentence.split())

        # Исключаем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'за', 'через'}
        last_content = last_words - stop_words
        first_content = first_words - stop_words

        if last_content and first_content:
            overlap = len(last_content & first_content)
            if overlap > 0:
                score += min(overlap * 0.1, 0.2)

        return min(score, 1.0)

    def _analyze_logical_connections(self, text: str) -> Dict[str, Any]:
        """Анализ логических связей в тексте"""
        text_lower = text.lower()

        # Подсчет связующих элементов
        connection_counts = {}
        total_connections = 0

        for category, connectives in self.connective_words.items():
            count = sum(text_lower.count(connective) for connective in connectives)
            connection_counts[f"{category}_connections"] = count
            total_connections += count

        # Плотность связей
        total_sentences = len(self._split_sentences(text))
        if total_sentences > 0:
            connection_density = total_connections / total_sentences
            logical_flow_score = min(connection_density / 0.3, 1.0)  # Оптимум: 0.3 связи на предложение
        else:
            connection_density = 0
            logical_flow_score = 0

        # Разнообразие типов связей
        types_used = sum(1 for count in connection_counts.values() if count > 0)
        connection_diversity = types_used / len(self.connective_words)

        return {
            "logical_flow_score": round(logical_flow_score, 3),
            "connection_density": round(connection_density, 3),
            "connection_diversity": round(connection_diversity, 3),
            "total_logical_connections": total_connections,
            **connection_counts
        }

    def _analyze_composition(self, text: str) -> Dict[str, Any]:
        """Анализ композиционной структуры"""
        text_lower = text.lower()

        # Обнаружение структурных элементов
        structural_elements = {}
        for element, markers in self.structural_markers.items():
            count = sum(text_lower.count(marker) for marker in markers)
            structural_elements[f"{element}_markers"] = count

        # Определение типа композиции
        composition_type = self._determine_composition_type(text, structural_elements)

        # Композиционная полнота
        elements_present = sum(1 for count in structural_elements.values() if count > 0)
        composition_completeness = elements_present / len(self.structural_markers)

        # Композиционный баланс
        total_elements = sum(structural_elements.values())
        if total_elements > 0:
            element_distribution = [count / total_elements for count in structural_elements.values()]
            composition_balance = 1 - self._calculate_variance(element_distribution)
        else:
            composition_balance = 0

        return {
            "composition_type": composition_type,
            "composition_completeness": round(composition_completeness, 3),
            "composition_balance": round(composition_balance, 3),
            **structural_elements
        }

    def _determine_composition_type(self, text: str, structural_elements: Dict[str, int]) -> str:
        """Определение типа композиции"""
        # Простая эвристика основанная на структурных маркерах
        intro_count = structural_elements.get("введение_markers", 0)
        development_count = structural_elements.get("развитие_markers", 0)
        climax_count = structural_elements.get("кульминация_markers", 0)
        conclusion_count = structural_elements.get("заключение_markers", 0)

        if intro_count > 0 and conclusion_count > 0 and development_count > 0:
            return "linear"  # Линейная структура
        elif intro_count > 0 and conclusion_count > 0:
            return "circular"  # Кольцевая структура
        elif development_count > climax_count + conclusion_count:
            return "parallel"  # Параллельная структура
        elif all(count > 0 for count in [intro_count, development_count, climax_count, conclusion_count]):
            return "classical"  # Классическая структура
        else:
            return "mosaic"  # Мозаичная структура

    def _analyze_text_rhythm(self, text: str) -> Dict[str, Any]:
        """Анализ ритма и темпа текста"""
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not sentences:
            return {"rhythm_score": 0}

        # Анализ длин предложений
        sentence_lengths = [len(s.split()) for s in sentences]

        # Ритмическая вариативность
        length_variance = self._calculate_variance(sentence_lengths)
        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        # Нормализованная вариативность (коэффициент вариации)
        if avg_length > 0:
            variation_coefficient = (length_variance ** 0.5) / avg_length
        else:
            variation_coefficient = 0

        # Оптимальная вариативность: не слишком монотонно, не слишком хаотично
        optimal_variation = 0.5
        rhythm_regularity = 1 - abs(variation_coefficient - optimal_variation) / optimal_variation

        # Анализ темпа (на основе знаков препинания)
        tempo_analysis = self._analyze_tempo_markers(text)

        # Анализ пауз (абзацы как паузы)
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            pause_frequency = len(paragraphs) / sum(len(p.split()) for p in paragraphs) * 1000
        else:
            avg_paragraph_length = 0
            pause_frequency = 0

        return {
            "rhythm_regularity": round(rhythm_regularity, 3),
            "variation_coefficient": round(variation_coefficient, 3),
            "pause_frequency": round(pause_frequency, 2),
            **tempo_analysis
        }

    def _analyze_tempo_markers(self, text: str) -> Dict[str, float]:
        """Анализ маркеров темпа в тексте"""
        # Подсчет знаков препинания
        slow_markers = text.count(',') + text.count(';') + text.count(':')  # Замедляющие
        fast_markers = text.count('!') + text.count('?')  # Ускоряющие
        pause_markers = text.count('...') + text.count('—') + text.count('–')  # Паузы

        total_chars = len(text)
        if total_chars > 0:
            slow_density = slow_markers / total_chars * 1000
            fast_density = fast_markers / total_chars * 1000
            pause_density = pause_markers / total_chars * 1000
        else:
            slow_density = fast_density = pause_density = 0

        # Общий темп (отношение быстрых к медленным маркерам)
        if slow_markers > 0:
            tempo_ratio = fast_markers / slow_markers
        else:
            tempo_ratio = fast_markers  # Если нет медленных маркеров

        return {
            "slow_tempo_density": round(slow_density, 2),
            "fast_tempo_density": round(fast_density, 2),
            "pause_density": round(pause_density, 2),
            "overall_tempo_ratio": round(tempo_ratio, 3)
        }

    def _analyze_information_density(self, text: str) -> Dict[str, Any]:
        """Анализ информационной плотности"""
        sentences = self._split_sentences(text)
        words = text.split()

        if not sentences or not words:
            return {"information_density": 0}

        # Плотность ключевых слов (эвристическая оценка)
        key_word_markers = [
            'важно', 'главное', 'основное', 'ключевое', 'центральное',
            'проблема', 'решение', 'результат', 'вывод', 'факт'
        ]

        key_word_count = sum(text.lower().count(marker) for marker in key_word_markers)
        key_word_density = key_word_count / len(words)

        # Плотность фактов (предложения с числами, датами, именами)
        fact_patterns = [
            r'\d{4}',  # Годы
            r'\d+%',   # Проценты
            r'\d+\.\d+',  # Десятичные числа
            r'[А-Я][а-я]+ [А-Я][а-я]+',  # Имена собственные
        ]

        fact_sentences = 0
        for sentence in sentences:
            if any(re.search(pattern, sentence) for pattern in fact_patterns):
                fact_sentences += 1

        fact_density = fact_sentences / len(sentences)

        # Информационная насыщенность
        information_score = (key_word_density * 2 + fact_density) / 3

        return {
            "information_density": round(information_score, 3),
            "key_word_density": round(key_word_density, 3),
            "fact_density": round(fact_density, 3),
            "fact_sentences": fact_sentences
        }

    def _analyze_narrative_patterns(self, text: str) -> Dict[str, Any]:
        """Анализ нарративных паттернов"""
        pattern_counts = {}
        total_patterns = 0

        for pattern_type, patterns in self.narrative_patterns.items():
            count = 0
            for pattern in patterns:
                if isinstance(pattern, str) and not pattern.startswith('r'):
                    count += text.count(pattern)
                else:
                    regex_pattern = pattern[2:] if pattern.startswith('r') else pattern
                    matches = re.findall(regex_pattern, text)
                    count += len(matches)

            pattern_counts[f"{pattern_type}_count"] = count
            total_patterns += count

        # Нарративное разнообразие
        patterns_used = sum(1 for count in pattern_counts.values() if count > 0)
        narrative_diversity = patterns_used / len(self.narrative_patterns)

        # Плотность нарративных элементов
        total_sentences = len(self._split_sentences(text))
        if total_sentences > 0:
            narrative_density = total_patterns / total_sentences
        else:
            narrative_density = 0

        return {
            "narrative_diversity": round(narrative_diversity, 3),
            "narrative_density": round(narrative_density, 3),
            "total_narrative_patterns": total_patterns,
            **pattern_counts
        }

    def _calculate_overall_structure_score(self, metrics: Dict[str, Any]) -> float:
        """Вычисление общей оценки качества структуры"""
        # Веса компонентов
        components = [
            ("paragraph_structure_score", 0.2),
            ("logical_flow_score", 0.25),
            ("composition_completeness", 0.2),
            ("rhythm_regularity", 0.15),
            ("information_density", 0.1),
            ("narrative_diversity", 0.1)
        ]

        score = 0
        total_weight = 0

        for metric, weight in components:
            if metric in metrics:
                value = metrics[metric]
                score += value * weight
                total_weight += weight

        return round(score / total_weight if total_weight > 0 else 0, 3)

    def _get_structure_grade(self, score: float) -> str:
        """Определение оценки качества структуры"""
        if score >= 0.8:
            return "отличная"
        elif score >= 0.65:
            return "хорошая"
        elif score >= 0.5:
            return "удовлетворительная"
        elif score >= 0.3:
            return "слабая"
        else:
            return "неудовлетворительная"

    def _generate_structure_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению структуры"""
        recommendations = []

        paragraph_score = metrics.get("paragraph_structure_score", 0)
        if paragraph_score < 0.5:
            recommendations.append("Улучшите абзацную структуру: варьируйте длину абзацев и добавьте переходы")

        logical_flow = metrics.get("logical_flow_score", 0)
        if logical_flow < 0.5:
            recommendations.append("Добавьте больше логических связок между предложениями и абзацами")

        composition_completeness = metrics.get("composition_completeness", 0)
        if composition_completeness < 0.5:
            recommendations.append("Добавьте структурные элементы: введение, развитие, заключение")

        rhythm_regularity = metrics.get("rhythm_regularity", 0)
        if rhythm_regularity < 0.5:
            recommendations.append("Варьируйте длину предложений для создания ритма")

        connection_diversity = metrics.get("connection_diversity", 0)
        if connection_diversity < 0.4:
            recommendations.append("Используйте разнообразные типы логических связей")

        if not recommendations:
            recommendations.append("Структура текста имеет хорошее качество")

        return recommendations

    async def _get_llm_structure_analysis(self, text_sample: str) -> Dict[str, Any]:
        """Анализ структуры с помощью LLM"""
        try:
            prompt = self.prompt_template.format(text=text_sample)
            response = await self.llm_service.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Ошибка LLM анализа структуры: {e}")
            return {}

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивка текста на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_variance(self, values: List[float]) -> float:
        """Вычисление дисперсии"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _calculate_structure_regularity(self, paragraphs: List[str]) -> float:
        """Вычисление структурной регулярности"""
        if not paragraphs:
            return 0

        lengths = [len(p.split()) for p in paragraphs]
        if not lengths:
            return 0

        # Анализ паттернов длин абзацев
        # Хорошая структура имеет определенную закономерность
        mean_length = sum(lengths) / len(lengths)
        deviations = [abs(length - mean_length) for length in lengths]
        avg_deviation = sum(deviations) / len(deviations)

        # Нормализация относительно средней длины
        if mean_length > 0:
            regularity = max(0, 1 - (avg_deviation / mean_length))
        else:
            regularity = 0

        return regularity

    # Методы для анализа по чанкам

    async def _analyze_chunks(self, text: BaseText, **kwargs) -> AnalysisResult:
        """Анализ структуры по чанкам"""
        chunking_strategy = kwargs.get('chunking_strategy', ChunkingStrategy())
        chunks = text.get_chunks(chunking_strategy)

        if not chunks:
            return AnalysisResult(
            text_id=text.id,
                analyzer_name=self.name,
                data={"error": "Нет чанков для анализа"},
                metadata={"mode": "chunks"}
            )

        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_result = await self._analyze_chunk(chunk, i)
            chunk_results.append(chunk_result)

        # Анализ глобальной структуры
        global_structure = self._analyze_inter_chunk_structure(chunks)

        # Агрегирование результатов
        aggregated_metrics = self._aggregate_chunk_structure(chunk_results)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data={
                "chunk_count": len(chunks),
                "chunk_results": chunk_results,
                "global_structure": global_structure,
                **aggregated_metrics
            },
            metadata={"mode": "chunks", "chunk_count": len(chunks)}
        )

    async def _analyze_chunk(self, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """Анализ структуры одного чанка"""
        basic_structure = self._analyze_basic_structure(chunk_text)
        logical_analysis = self._analyze_logical_connections(chunk_text)
        rhythm_analysis = self._analyze_text_rhythm(chunk_text)

        overall_score = self._calculate_overall_structure_score({
            **basic_structure,
            **logical_analysis,
            **rhythm_analysis
        })

        return {
            "chunk_index": chunk_index,
            "overall_structure_score": overall_score,
            "structure_grade": self._get_structure_grade(overall_score),
            **basic_structure,
            **logical_analysis,
            **rhythm_analysis
        }

    def _analyze_inter_chunk_structure(self, chunks: List[str]) -> Dict[str, Any]:
        """Анализ структуры между чанками"""
        if len(chunks) < 2:
            return {"inter_chunk_coherence": 1.0}

        # Анализ переходов между чанками
        transition_scores = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Берем последнее предложение текущего чанка и первое следующего
            current_sentences = self._split_sentences(current_chunk)
            next_sentences = self._split_sentences(next_chunk)

            if current_sentences and next_sentences:
                transition_score = self._score_transition(
                    current_sentences[-1].lower(),
                    next_sentences[0].lower()
                )
                transition_scores.append(transition_score)

        inter_chunk_coherence = sum(transition_scores) / len(transition_scores) if transition_scores else 0

        # Анализ тематической связности между чанками
        thematic_coherence = self._analyze_thematic_coherence(chunks)

        return {
            "inter_chunk_coherence": round(inter_chunk_coherence, 3),
            "thematic_coherence": round(thematic_coherence, 3),
            "chunk_transitions_count": len(transition_scores)
        }

    def _analyze_thematic_coherence(self, chunks: List[str]) -> float:
        """Анализ тематической связности между чанками"""
        if len(chunks) < 2:
            return 1.0

        # Простая эвристика: пересечение ключевых слов между соседними чанками
        coherence_scores = []

        for i in range(len(chunks) - 1):
            chunk1_words = set(re.findall(r'\b\w{4,}\b', chunks[i].lower()))
            chunk2_words = set(re.findall(r'\b\w{4,}\b', chunks[i + 1].lower()))

            # Исключаем очень частые слова
            stop_words = {'который', 'которая', 'которые', 'этого', 'этой', 'этому', 'этим'}
            chunk1_words -= stop_words
            chunk2_words -= stop_words

            if chunk1_words and chunk2_words:
                overlap = len(chunk1_words & chunk2_words)
                union = len(chunk1_words | chunk2_words)
                coherence = overlap / union if union > 0 else 0
                coherence_scores.append(coherence)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    def _aggregate_chunk_structure(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирование результатов анализа структуры по чанкам"""
        if not chunk_results:
            return {}

        # Средние значения метрик
        numeric_metrics = [
            "overall_structure_score", "logical_flow_score", "rhythm_regularity", "connection_density"
        ]

        aggregated = {}
        for metric in numeric_metrics:
            values = [chunk.get(metric, 0) for chunk in chunk_results if chunk.get(metric) is not None]
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 3)
                aggregated[f"min_{metric}"] = round(min(values), 3)
                aggregated[f"max_{metric}"] = round(max(values), 3)

        # Общая оценка структуры
        avg_score = aggregated.get("avg_overall_structure_score", 0)
        aggregated["overall_structure_grade"] = self._get_structure_grade(avg_score)

        # Структурная консистентность между чанками
        scores = [chunk.get("overall_structure_score", 0) for chunk in chunk_results]
        if len(scores) > 1:
            score_variance = self._calculate_variance(scores)
            consistency = max(0, 1 - score_variance)
            aggregated["structural_consistency"] = round(consistency, 3)

        return aggregated

    def estimate_time(self, text: BaseText, mode: AnalysisMode = AnalysisMode.FULL_TEXT) -> float:
        """Оценка времени анализа структуры"""
        base_time = 1.5  # Базовое время для анализа структуры

        # Время зависит от количества абзацев и предложений
        estimated_paragraphs = text.content.count('\n\n') + 1
        estimated_sentences = text.content.count('.') + text.content.count('!') + text.content.count('?')

        complexity_factor = (estimated_paragraphs / 20) + (estimated_sentences / 100)

        if mode == AnalysisMode.CHUNKED:
            chunking_strategy = ChunkingStrategy()
            estimated_chunks = max(1, len(text.content) // chunking_strategy.chunk_size)
            return (base_time + complexity_factor) * estimated_chunks * 0.5
        else:
            llm_time = 2.5 if self.llm_service else 0
            return base_time + complexity_factor + llm_time

    def get_supported_modes(self) -> list[AnalysisMode]:
        """Получить поддерживаемые режимы анализа"""
        return [AnalysisMode.FULL_TEXT, AnalysisMode.CHUNKED]

    # Реализация абстрактных методов BaseAnalyzer

    @property
    def name(self) -> str:
        return "structure"

    @property
    def display_name(self) -> str:
        return "Анализ структуры"

    @property
    def description(self) -> str:
        return "Анализирует структуру, композицию и организацию текста"

    @property
    def requires_llm(self) -> bool:
        return False

    def interpret_results(self, result: AnalysisResult) -> str:
        data = result.result_data
        if "error" in data:
            return f"Ошибка анализа структуры: {data['error']}"

        overall_quality = data.get("overall_structure_quality", 0)
        structure_grade = data.get("structure_grade", "неудовлетворительная")
        composition_type = data.get("composition_type", "неопределенная")

        interpretation = f"Структура текста: {structure_grade} ({overall_quality:.3f})\n"
        interpretation += f"Тип композиции: {composition_type}\n"

        logical_flow = data.get("logical_flow_score", 0)
        if logical_flow > 0:
            interpretation += f"Логическая связность: {logical_flow:.3f}\n"

        paragraph_score = data.get("paragraph_structure_score", 0)
        if paragraph_score > 0:
            interpretation += f"Качество абзацной структуры: {paragraph_score:.3f}\n"

        return interpretation.strip()