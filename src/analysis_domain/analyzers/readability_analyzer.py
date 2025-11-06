"""
Анализатор читаемости текста с использованием научных метрик
"""
import re
import math
import json
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class ReadabilityAnalyzer(BaseAnalyzer):
    """
    Анализатор читаемости текста

    Использует научно обоснованные метрики читаемости:
    - Индекс Флеша (адаптация для русского языка)
    - Формула SMOG
    - Индекс Коулмана-Лиау
    - Индекс автоматической читаемости (ARI)
    - Российские адаптации формул читаемости
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._get_default_prompt()

        # Российские стоп-слова для анализа
        self.stop_words = {
            'а', 'без', 'более', 'бы', 'был', 'была', 'были', 'было', 'быть', 'в', 'вам',
            'вас', 'весь', 'во', 'вот', 'все', 'всего', 'всех', 'вы', 'где', 'да', 'даже',
            'для', 'до', 'его', 'ее', 'если', 'есть', 'еще', 'же', 'за', 'здесь', 'и', 'из',
            'или', 'им', 'их', 'к', 'как', 'ко', 'когда', 'кто', 'ли', 'либо', 'мне', 'может',
            'мы', 'на', 'надо', 'наш', 'не', 'него', 'нее', 'нет', 'ни', 'них', 'но', 'ну',
            'о', 'об', 'одной', 'он', 'она', 'они', 'оно', 'от', 'очень', 'по', 'под', 'при',
            'с', 'со', 'так', 'также', 'такой', 'там', 'те', 'тем', 'то', 'того', 'тоже',
            'той', 'только', 'том', 'ты', 'у', 'уже', 'хотя', 'чего', 'чем', 'что', 'чтобы',
            'чье', 'эта', 'эти', 'это', 'я'
        }

        # Частотные суффиксы русского языка для определения сложности слов
        self.complex_suffixes = {
            'ность', 'ость', 'ение', 'ание', 'ция', 'сть', 'тель', 'ист', 'изм',
            'ическ', 'альн', 'ивн', 'енн', 'онн', 'ированн'
        }

    def _get_default_prompt(self) -> PromptTemplate:
        """Получить промпт по умолчанию для LLM анализа"""
        template = """
        Проанализируй читаемость данного текста с точки зрения:
        1. Доступности для целевой аудитории
        2. Ясности изложения
        3. Структурированности материала

        Текст: {text}

        Верни результат в формате JSON:
        {{
            "target_audience": "children|teenagers|adults|specialists",
            "clarity_score": float (0-1),
            "structure_quality": float (0-1),
            "engagement_level": float (0-1),
            "recommendations": ["список конкретных рекомендаций"]
        }}
        """
        return PromptTemplate(
            template=template,
            variables=["text"],
            name="readability_analysis_prompt"
        )

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        **kwargs
    ) -> AnalysisResult:
        """
        Анализ читаемости текста

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа читаемости
        """
        try:
            logger.info(f"Начинаем анализ читаемости текста: {text.title}")

            if mode == AnalysisMode.CHUNKED:
                return await self._analyze_chunks(text, **kwargs)
            else:
                return await self._analyze_full_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка анализа читаемости: {e}")
            raise AnalysisError(f"Не удалось проанализировать читаемость текста: {e}")

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

        # Базовые метрики текста
        basic_metrics = self._calculate_basic_metrics(content)

        # Индексы читаемости
        readability_indices = self._calculate_readability_indices(content, basic_metrics)

        # Лингвистические характеристики
        linguistic_metrics = self._calculate_linguistic_metrics(content, basic_metrics)

        # Структурные характеристики
        structural_metrics = self._calculate_structural_metrics(content)

        # Когнитивная нагрузка
        cognitive_metrics = self._calculate_cognitive_load(content, basic_metrics)

        # Объединяем все метрики
        all_metrics = {
            **basic_metrics,
            **readability_indices,
            **linguistic_metrics,
            **structural_metrics,
            **cognitive_metrics
        }

        # Вычисляем общий индекс читаемости
        overall_readability = self._calculate_overall_readability(all_metrics)
        all_metrics["overall_readability_score"] = overall_readability
        all_metrics["readability_level"] = self._get_readability_level(overall_readability)
        all_metrics["target_grade_level"] = self._estimate_grade_level(overall_readability)

        # Генерируем рекомендации
        all_metrics["automated_recommendations"] = self._generate_recommendations(all_metrics)

        # Если есть LLM сервис, добавляем углубленный анализ
        if self.llm_service:
            llm_analysis = await self._get_llm_analysis(content[:2000])
            all_metrics.update(llm_analysis)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data=all_metrics,
            metadata={
                "mode": "full_text",
                "text_length": len(content),
                "analysis_timestamp": logger.start_time.isoformat() if hasattr(logger, 'start_time') else None
            }
        )

    def _calculate_basic_metrics(self, text: str) -> Dict[str, Any]:
        """Вычисление базовых метрик текста"""
        # Разбивка на составные части
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = self._split_sentences(text)
        words = self._extract_words(text)
        syllables = self._count_total_syllables(words)
        characters = len(re.sub(r'\s+', '', text))

        return {
            "character_count": characters,
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "syllable_count": syllables,
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
            "avg_syllables_per_word": round(syllables / max(len(words), 1), 2),
            "avg_characters_per_word": round(characters / max(len(words), 1), 2),
            "avg_sentences_per_paragraph": round(len(sentences) / max(len(paragraphs), 1), 2)
        }

    def _calculate_readability_indices(self, text: str, basic_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Вычисление индексов читаемости"""
        words_count = basic_metrics["word_count"]
        sentences_count = basic_metrics["sentence_count"]
        syllables_count = basic_metrics["syllable_count"]
        characters_count = basic_metrics["character_count"]

        if not words_count or not sentences_count:
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "smog_index": 0,
                "ari_index": 0,
                "coleman_liau_index": 0
            }

        # Flesch Reading Ease (адаптация для русского)
        avg_sentence_length = words_count / sentences_count
        avg_syllables_per_word = syllables_count / words_count

        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_normalized = max(0, min(100, flesch_score))

        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        fk_grade = max(0, fk_grade)

        # SMOG Index (адаптация)
        complex_words = self._count_complex_words_smog(text)
        if sentences_count >= 3:
            smog_index = 1.043 * math.sqrt(complex_words * 30 / sentences_count) + 3.1291
        else:
            smog_index = 0

        # Automated Readability Index (ARI)
        avg_chars_per_word = characters_count / words_count
        ari_index = (4.71 * avg_chars_per_word) + (0.5 * avg_sentence_length) - 21.43
        ari_index = max(0, ari_index)

        # Coleman-Liau Index
        L = (characters_count / words_count) * 100  # среднее количество букв на 100 слов
        S = (sentences_count / words_count) * 100   # среднее количество предложений на 100 слов
        cli_index = (0.0588 * L) - (0.296 * S) - 15.8
        cli_index = max(0, cli_index)

        return {
            "flesch_reading_ease": round(flesch_normalized, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "smog_index": round(smog_index, 2),
            "ari_index": round(ari_index, 2),
            "coleman_liau_index": round(cli_index, 2)
        }

    def _calculate_linguistic_metrics(self, text: str, basic_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Вычисление лингвистических метрик"""
        words = self._extract_words(text.lower())
        if not words:
            return {}

        # Лексическое разнообразие
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)

        # Доля сложных слов
        complex_words = self._count_morphologically_complex_words(words)
        complex_words_ratio = complex_words / len(words)

        # Доля стоп-слов
        stop_words_count = sum(1 for word in words if word in self.stop_words)
        stop_words_ratio = stop_words_count / len(words)

        # Средняя частота слов (приблизительная оценка)
        word_frequency_score = self._estimate_word_frequency(words)

        # Морфологическая сложность
        morphological_complexity = self._calculate_morphological_complexity(words)

        return {
            "lexical_diversity": round(lexical_diversity, 3),
            "complex_words_ratio": round(complex_words_ratio, 3),
            "stop_words_ratio": round(stop_words_ratio, 3),
            "word_frequency_score": round(word_frequency_score, 3),
            "morphological_complexity": round(morphological_complexity, 3),
            "unique_words_count": len(unique_words)
        }

    def _calculate_structural_metrics(self, text: str) -> Dict[str, float]:
        """Вычисление метрик структуры текста"""
        # Анализ знаков препинания
        punctuation_density = self._calculate_punctuation_density(text)

        # Анализ организации текста
        organization_score = self._analyze_text_organization(text)

        # Анализ связности
        coherence_score = self._analyze_text_coherence(text)

        # Анализ параллельных конструкций
        parallel_structures = self._detect_parallel_structures(text)

        return {
            "punctuation_density": round(punctuation_density, 3),
            "organization_score": round(organization_score, 3),
            "coherence_score": round(coherence_score, 3),
            "parallel_structures_score": round(parallel_structures, 3)
        }

    def _calculate_cognitive_load(self, text: str, basic_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Вычисление когнитивной нагрузки"""
        # Информационная плотность
        info_density = self._calculate_information_density(text, basic_metrics)

        # Сложность концепций (эвристическая оценка)
        concept_complexity = self._estimate_concept_complexity(text)

        # Нагрузка на рабочую память
        memory_load = self._estimate_memory_load(text, basic_metrics)

        return {
            "information_density": round(info_density, 3),
            "concept_complexity": round(concept_complexity, 3),
            "memory_load": round(memory_load, 3)
        }

    def _calculate_overall_readability(self, metrics: Dict[str, Any]) -> float:
        """Вычисление общего индекса читаемости"""
        # Веса для различных компонентов
        weights = {
            "flesch_reading_ease": 0.25,
            "lexical_diversity": -0.1,  # Слишком высокое разнообразие может усложнить чтение
            "complex_words_ratio": -0.2,
            "avg_words_per_sentence": -0.15,
            "stop_words_ratio": 0.1,
            "coherence_score": 0.15,
            "organization_score": 0.15,
            "information_density": -0.1
        }

        score = 50  # Базовая оценка

        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == "flesch_reading_ease":
                    # Flesch уже в нужном диапазоне 0-100
                    score += value * weight
                elif metric == "avg_words_per_sentence":
                    # Нормализуем длину предложений (оптимум 15-20 слов)
                    optimal_length = 17.5
                    deviation = abs(value - optimal_length) / optimal_length
                    score += (1 - min(deviation, 1)) * weight * 100
                else:
                    # Остальные метрики в диапазоне 0-1
                    score += value * weight * 100

        return max(0, min(100, score)) / 100

    def _get_readability_level(self, score: float) -> str:
        """Определение уровня читаемости"""
        if score >= 0.8:
            return "очень легкий"
        elif score >= 0.65:
            return "легкий"
        elif score >= 0.5:
            return "средний"
        elif score >= 0.3:
            return "трудный"
        else:
            return "очень трудный"

    def _estimate_grade_level(self, readability_score: float) -> str:
        """Оценка уровня образования для понимания текста"""
        if readability_score >= 0.8:
            return "начальная школа (1-4 класс)"
        elif readability_score >= 0.65:
            return "средняя школа (5-8 класс)"
        elif readability_score >= 0.5:
            return "старшая школа (9-11 класс)"
        elif readability_score >= 0.3:
            return "высшее образование"
        else:
            return "специализированное образование"

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению читаемости"""
        recommendations = []

        # Анализ длины предложений
        avg_words_per_sentence = metrics.get("avg_words_per_sentence", 0)
        if avg_words_per_sentence > 25:
            recommendations.append("Сократите длинные предложения (более 25 слов)")
        elif avg_words_per_sentence > 20:
            recommendations.append("Рассмотрите возможность разбивки предложений длиннее 20 слов")

        # Анализ сложности слов
        complex_words_ratio = metrics.get("complex_words_ratio", 0)
        if complex_words_ratio > 0.3:
            recommendations.append("Замените сложные слова на более простые синонимы где возможно")

        # Анализ лексического разнообразия
        lexical_diversity = metrics.get("lexical_diversity", 0)
        if lexical_diversity < 0.3:
            recommendations.append("Используйте более разнообразную лексику")
        elif lexical_diversity > 0.8:
            recommendations.append("Уменьшите количество редких слов для улучшения понимания")

        # Анализ структуры
        organization_score = metrics.get("organization_score", 0)
        if organization_score < 0.5:
            recommendations.append("Улучшите структуру текста: добавьте заголовки и логические переходы")

        # Анализ связности
        coherence_score = metrics.get("coherence_score", 0)
        if coherence_score < 0.5:
            recommendations.append("Добавьте связующие слова и фразы между предложениями")

        if not recommendations:
            recommendations.append("Текст имеет хорошие показатели читаемости")

        return recommendations

    async def _get_llm_analysis(self, text: str) -> Dict[str, Any]:
        """Анализ с помощью LLM"""
        try:
            prompt = self.prompt_template.format(text=text)
            response = await self.llm_service.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Ошибка LLM анализа читаемости: {e}")
            return {}

    # Вспомогательные методы для вычислений

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивка на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_words(self, text: str) -> List[str]:
        """Извлечение слов"""
        return re.findall(r'\b[а-яёa-z]+\b', text.lower())

    def _count_total_syllables(self, words: List[str]) -> int:
        """Подсчет общего количества слогов"""
        return sum(self._count_syllables(word) for word in words)

    def _count_syllables(self, word: str) -> int:
        """Подсчет слогов в слове"""
        vowels = 'аеёиоуыэюя'
        count = 0
        prev_was_vowel = False

        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        return max(count, 1)

    def _count_complex_words_smog(self, text: str) -> int:
        """Подсчет сложных слов для SMOG индекса (3+ слога)"""
        words = self._extract_words(text)
        return sum(1 for word in words if self._count_syllables(word) >= 3)

    def _count_morphologically_complex_words(self, words: List[str]) -> int:
        """Подсчет морфологически сложных слов"""
        complex_count = 0
        for word in words:
            # Проверяем длину слова
            if len(word) > 10:
                complex_count += 1
            # Проверяем наличие сложных суффиксов
            elif any(suffix in word for suffix in self.complex_suffixes):
                complex_count += 1
            # Проверяем количество слогов
            elif self._count_syllables(word) > 4:
                complex_count += 1

        return complex_count

    def _estimate_word_frequency(self, words: List[str]) -> float:
        """Оценка частотности слов (эвристическая)"""
        # Простая эвристика: чем больше стоп-слов, тем выше частотность
        stop_word_ratio = sum(1 for word in words if word in self.stop_words) / len(words)
        return stop_word_ratio

    def _calculate_morphological_complexity(self, words: List[str]) -> float:
        """Вычисление морфологической сложности"""
        if not words:
            return 0

        complexity_scores = []
        for word in words:
            score = 0
            # Длина слова
            score += min(len(word) / 15, 1)
            # Наличие сложных морфем
            if any(suffix in word for suffix in self.complex_suffixes):
                score += 0.5
            complexity_scores.append(score)

        return sum(complexity_scores) / len(complexity_scores)

    def _calculate_punctuation_density(self, text: str) -> float:
        """Вычисление плотности знаков препинания"""
        punctuation_marks = ',;:!?-—()[]{}«»"'
        punctuation_count = sum(text.count(mark) for mark in punctuation_marks)
        return punctuation_count / max(len(text), 1) * 1000  # на 1000 символов

    def _analyze_text_organization(self, text: str) -> float:
        """Анализ организации текста"""
        score = 0

        # Наличие абзацев
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.3

        # Регулярность структуры абзацев
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        if paragraph_lengths:
            avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
            variance = sum((length - avg_length) ** 2 for length in paragraph_lengths) / len(paragraph_lengths)
            regularity = 1 / (1 + variance / (avg_length ** 2))
            score += 0.3 * regularity

        # Наличие связующих элементов
        connectives = ['однако', 'поэтому', 'таким образом', 'кроме того', 'во-первых', 'во-вторых']
        connective_count = sum(text.lower().count(conn) for conn in connectives)
        connective_density = connective_count / max(len(text.split()), 1)
        score += min(connective_density * 10, 0.4)

        return min(score, 1.0)

    def _analyze_text_coherence(self, text: str) -> float:
        """Анализ связности текста"""
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 1.0

        coherence_score = 0

        # Анализ повторяющихся ключевых слов между предложениями
        for i in range(1, len(sentences)):
            prev_words = set(self._extract_words(sentences[i-1]))
            curr_words = set(self._extract_words(sentences[i]))

            # Исключаем стоп-слова
            prev_content = prev_words - self.stop_words
            curr_content = curr_words - self.stop_words

            if prev_content and curr_content:
                overlap = len(prev_content & curr_content)
                max_words = max(len(prev_content), len(curr_content))
                sentence_coherence = overlap / max_words
                coherence_score += sentence_coherence

        return coherence_score / max(len(sentences) - 1, 1)

    def _detect_parallel_structures(self, text: str) -> float:
        """Обнаружение параллельных структур"""
        # Простая эвристика: поиск повторяющихся паттернов
        sentences = self._split_sentences(text)

        parallel_score = 0
        pattern_count = 0

        for sentence in sentences:
            # Поиск списков и перечислений
            if re.search(r'\b(во-первых|во-вторых|в-третьих|первый|второй|третий)', sentence.lower()):
                pattern_count += 1
            # Поиск повторяющихся структур
            if re.search(r'\b(если|когда|поскольку).*,.*то\b', sentence.lower()):
                pattern_count += 1

        if sentences:
            parallel_score = pattern_count / len(sentences)

        return min(parallel_score, 1.0)

    def _calculate_information_density(self, text: str, basic_metrics: Dict[str, Any]) -> float:
        """Вычисление информационной плотности"""
        words = self._extract_words(text.lower())
        content_words = [w for w in words if w not in self.stop_words]

        if not words:
            return 0

        # Доля содержательных слов
        content_ratio = len(content_words) / len(words)

        # Уникальность содержательных слов
        unique_content = set(content_words)
        uniqueness_ratio = len(unique_content) / max(len(content_words), 1)

        # Информационная плотность
        info_density = (content_ratio + uniqueness_ratio) / 2

        return min(info_density, 1.0)

    def _estimate_concept_complexity(self, text: str) -> float:
        """Оценка сложности концепций (эвристическая)"""
        # Ключевые слова, указывающие на абстрактные концепции
        abstract_markers = [
            'концепция', 'теория', 'принцип', 'подход', 'метод', 'система', 'процесс',
            'анализ', 'синтез', 'структура', 'функция', 'отношение', 'взаимодействие'
        ]

        # Специализированные термины (эвристика по длине и суффиксам)
        words = self._extract_words(text.lower())

        abstract_count = sum(text.lower().count(marker) for marker in abstract_markers)
        specialized_count = sum(1 for word in words
                              if len(word) > 8 and any(suffix in word for suffix in self.complex_suffixes))

        total_words = len(words)
        if total_words == 0:
            return 0

        complexity = (abstract_count + specialized_count) / total_words
        return min(complexity * 5, 1.0)  # Масштабирование

    def _estimate_memory_load(self, text: str, basic_metrics: Dict[str, Any]) -> float:
        """Оценка нагрузки на рабочую память"""
        avg_sentence_length = basic_metrics.get("avg_words_per_sentence", 0)

        # Базовая нагрузка от длины предложений
        length_load = min(avg_sentence_length / 30, 1.0)  # 30 слов = максимальная нагрузка

        # Нагрузка от сложных конструкций
        complex_constructions = [
            r'\b\w+,\s*\w+,\s*\w+\b',  # списки
            r'\([^)]+\)',               # скобки
            r'\b(который|которая|которые|что|чтобы)\b'  # придаточные
        ]

        construction_count = sum(len(re.findall(pattern, text.lower()))
                               for pattern in complex_constructions)

        sentences_count = basic_metrics.get("sentence_count", 1)
        construction_load = min(construction_count / sentences_count, 1.0)

        # Объединенная оценка
        memory_load = (length_load + construction_load) / 2

        return memory_load

    # Методы для анализа по чанкам

    async def _analyze_chunks(self, text: BaseText, **kwargs) -> AnalysisResult:
        """Анализ по чанкам"""
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

        # Агрегирование результатов
        aggregated_metrics = self._aggregate_chunk_readability(chunk_results)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data={
                "chunk_count": len(chunks),
                "chunk_results": chunk_results,
                **aggregated_metrics
            },
            metadata={"mode": "chunks", "chunk_count": len(chunks)}
        )

    async def _analyze_chunk(self, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """Анализ одного чанка"""
        basic_metrics = self._calculate_basic_metrics(chunk_text)
        readability_indices = self._calculate_readability_indices(chunk_text, basic_metrics)
        linguistic_metrics = self._calculate_linguistic_metrics(chunk_text, basic_metrics)

        overall_readability = self._calculate_overall_readability({
            **basic_metrics,
            **readability_indices,
            **linguistic_metrics
        })

        return {
            "chunk_index": chunk_index,
            "overall_readability_score": overall_readability,
            "readability_level": self._get_readability_level(overall_readability),
            **basic_metrics,
            **readability_indices,
            **linguistic_metrics
        }

    def _aggregate_chunk_readability(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирование результатов анализа чанков"""
        if not chunk_results:
            return {}

        # Метрики для усреднения
        numeric_metrics = [
            "overall_readability_score", "flesch_reading_ease", "avg_words_per_sentence",
            "lexical_diversity", "complex_words_ratio"
        ]

        aggregated = {}
        for metric in numeric_metrics:
            values = [chunk.get(metric, 0) for chunk in chunk_results]
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 3)
                aggregated[f"min_{metric}"] = round(min(values), 3)
                aggregated[f"max_{metric}"] = round(max(values), 3)

        # Общий уровень читаемости
        avg_readability = aggregated.get("avg_overall_readability_score", 0.5)
        aggregated["overall_readability_level"] = self._get_readability_level(avg_readability)
        aggregated["estimated_grade_level"] = self._estimate_grade_level(avg_readability)

        return aggregated

    def estimate_time(self, text: BaseText, mode: AnalysisMode = AnalysisMode.FULL_TEXT) -> float:
        """Оценка времени анализа"""
        base_time = 0.5  # Базовое время для математических вычислений

        # Время зависит от длины текста
        text_factor = len(text.content) / 10000  # 10000 символов = 1 секунда

        if mode == AnalysisMode.CHUNKED:
            chunking_strategy = ChunkingStrategy()
            estimated_chunks = max(1, len(text.content) // chunking_strategy.chunk_size)
            return (base_time + text_factor) * estimated_chunks * 0.2
        else:
            llm_time = 1.5 if self.llm_service else 0
            return base_time + text_factor + llm_time

    def get_supported_modes(self) -> list[AnalysisMode]:
        """Получить поддерживаемые режимы анализа"""
        return [AnalysisMode.FULL_TEXT, AnalysisMode.CHUNKED]

    # Реализация абстрактных методов BaseAnalyzer

    @property
    def name(self) -> str:
        """Уникальное имя анализатора"""
        return "readability"

    @property
    def display_name(self) -> str:
        """Человекочитаемое название"""
        return "Анализ читаемости"

    @property
    def description(self) -> str:
        """Описание анализатора"""
        return "Анализирует читаемость текста с помощью научных метрик читаемости"

    @property
    def requires_llm(self) -> bool:
        """Требует ли анализатор LLM"""
        return False

    def interpret_results(self, result: AnalysisResult) -> str:
        """Интерпретация результатов анализа"""
        data = result.result_data

        if "error" in data:
            return f"Ошибка анализа читаемости: {data['error']}"

        readability_score = data.get("overall_readability_score", 0)
        readability_level = data.get("readability_level", "неизвестно")
        grade_level = data.get("target_grade_level", "неопределен")

        flesch_score = data.get("flesch_reading_ease", 0)

        interpretation = f"Читаемость текста: {readability_level} "
        interpretation += f"(общая оценка: {readability_score:.2f})\n"
        interpretation += f"Целевая аудитория: {grade_level}\n"
        interpretation += f"Индекс Флеша: {flesch_score:.1f}\n"

        # Добавляем рекомендации если есть
        recommendations = data.get("automated_recommendations", [])
        if recommendations:
            interpretation += "\nРекомендации:\n"
            for rec in recommendations[:3]:  # Первые 3 рекомендации
                interpretation += f"• {rec}\n"

        return interpretation.strip()