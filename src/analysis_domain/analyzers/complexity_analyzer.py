"""
Анализатор сложности текста с использованием NLP метрик
"""
import re
import math
import json
from collections import Counter
from typing import Optional, Dict, Any, List
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class ComplexityAnalyzer(BaseAnalyzer):
    """
    Анализатор сложности текста на основе NLP метрик

    Использует различные алгоритмы для оценки сложности:
    - Flesch Reading Ease
    - Automated Readability Index (ARI)
    - Fog Index
    - Лексическое разнообразие
    - Синтаксическая сложность
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        super().__init__(
            name="ComplexityAnalyzer",
            description="Анализирует сложность текста с помощью NLP метрик"
        )
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._get_default_prompt()

        # Обогащенный список общеупотребительных слов (русский язык)
        self.common_words = {
            # Основные служебные слова
            'а', 'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'к', 'ко',
            'его', 'но', 'да', 'ты', 'по', 'только', 'её', 'мне', 'было', 'вот', 'от', 'меня',
            'ещё', 'нет', 'о', 'об', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
            'опять', 'уж', 'вам', 'сказал', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
            'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
            'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'человек', 'чего', 'раз', 'тоже',
            'себе', 'под', 'жизнь', 'будет', 'ж', 'тогда', 'кто', 'этот', 'сказать', 'того',
            'за', 'чтобы', 'дело', 'всё', 'этой', 'лучше', 'через', 'эти', 'нас', 'про',
            'всего', 'них', 'какая', 'много', 'разве', 'сказала', 'три', 'эту', 'моя',
            'впрочем', 'хорошо', 'свою', 'этого', 'лет', 'куда', 'зачем', 'всех', 'никогда',
            'сегодня', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после',
            'над', 'больше', 'тот', 'через', 'эта', 'нас', 'про', 'всего', 'них', 'какой',

            # Базовая лексика
            'день', 'года', 'дом', 'время', 'рука', 'нога', 'голова', 'глаз', 'слово', 'дело',
            'мир', 'земля', 'вода', 'огонь', 'воздух', 'солнце', 'луна', 'звезда', 'небо', 'море',
            'дерево', 'цветок', 'трава', 'птица', 'собака', 'кошка', 'лошадь', 'корова', 'рыба',
            'мать', 'отец', 'сын', 'дочь', 'брат', 'сестра', 'дед', 'бабушка', 'друг', 'враг',
            'любовь', 'ненависть', 'радость', 'грусть', 'счастье', 'печаль', 'страх', 'надежда',
            'красота', 'сила', 'слабость', 'правда', 'ложь', 'добро', 'зло', 'мир', 'война',
            'работа', 'отдых', 'учеба', 'игра', 'еда', 'питье', 'сон', 'здоровье', 'болезнь',
            'деньги', 'богатство', 'бедность', 'дорога', 'путь', 'город', 'деревня', 'страна',

            # Простые глаголы
            'идти', 'ехать', 'лететь', 'бежать', 'ходить', 'стоять', 'сидеть', 'лежать', 'спать',
            'есть', 'пить', 'говорить', 'молчать', 'слушать', 'смотреть', 'видеть', 'слышать',
            'знать', 'думать', 'помнить', 'забывать', 'учить', 'читать', 'писать', 'считать',
            'работать', 'играть', 'петь', 'танцевать', 'смеяться', 'плакать', 'любить', 'ненавидеть',

            # Простые прилагательные
            'большой', 'маленький', 'хороший', 'плохой', 'новый', 'старый', 'молодой', 'красивый',
            'умный', 'глупый', 'сильный', 'слабый', 'быстрый', 'медленный', 'высокий', 'низкий',
            'длинный', 'короткий', 'широкий', 'узкий', 'толстый', 'тонкий', 'тяжелый', 'легкий',
            'горячий', 'холодный', 'теплый', 'прохладный', 'сухой', 'мокрый', 'чистый', 'грязный'
        }

        # Категории сложных слов для лучшего анализа
        self.specialized_domains = {
            'научные': {
                'математика', 'физика', 'химия', 'биология', 'география', 'астрономия',
                'алгебра', 'геометрия', 'тригонометрия', 'интеграл', 'дифференциал',
                'молекула', 'атом', 'элемент', 'реакция', 'катализатор', 'кислота'
            },
            'медицинские': {
                'диагноз', 'симптом', 'синдром', 'терапия', 'хирургия', 'анестезия',
                'антибиотик', 'вакцина', 'иммунитет', 'патология', 'анатомия', 'физиология'
            },
            'технические': {
                'алгоритм', 'программа', 'процессор', 'память', 'интерфейс', 'протокол',
                'архитектура', 'компилятор', 'операционная', 'база данных', 'сервер'
            },
            'юридические': {
                'конституция', 'закон', 'кодекс', 'статья', 'параграф', 'юрисдикция',
                'правонарушение', 'судопроизводство', 'истец', 'ответчик', 'адвокат'
            },
            'экономические': {
                'экономика', 'финансы', 'инвестиции', 'капитал', 'прибыль', 'убыток',
                'инфляция', 'дефляция', 'девальвация', 'валютный', 'биржевой', 'акционерный'
            }
        }

    def _get_default_prompt(self) -> PromptTemplate:
        """Получить промпт по умолчанию для LLM анализа"""
        template = """
        Проанализируй сложность данного текста с точки зрения:
        1. Терминологической сложности
        2. Абстрактности концепций
        3. Предметной области

        Текст: {text}

        Верни результат в формате JSON:
        {{
            "domain_complexity": float (0-1),
            "conceptual_difficulty": float (0-1),
            "terminology_density": float (0-1),
            "domain": "область знаний",
            "key_concepts": ["список ключевых концепций"],
            "difficult_terms": ["список сложных терминов"]
        }}
        """
        return PromptTemplate(
            template=template,
            variables=["text"],
            name="complexity_analysis_prompt"
        )

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        **kwargs
    ) -> AnalysisResult:
        """
        Анализ сложности текста

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа сложности
        """
        try:
            logger.info(f"Начинаем анализ сложности текста: {text.title}")

            if mode == AnalysisMode.CHUNKED:
                return await self._analyze_chunks(text, **kwargs)
            else:
                return await self._analyze_full_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка анализа сложности: {e}")
            raise AnalysisError(f"Не удалось проанализировать сложность текста: {e}")

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

        # Основные NLP метрики
        linguistic_metrics = self._calculate_linguistic_complexity(content)

        # Структурные метрики
        structural_metrics = self._calculate_structural_complexity(content)

        # Лексические метрики
        lexical_metrics = self._calculate_lexical_complexity(content)

        # Синтаксические метрики
        syntactic_metrics = self._calculate_syntactic_complexity(content)

        # Объединяем все метрики
        all_metrics = {
            **linguistic_metrics,
            **structural_metrics,
            **lexical_metrics,
            **syntactic_metrics
        }

        # Вычисляем общий индекс сложности
        overall_complexity = self._calculate_overall_complexity(all_metrics)
        all_metrics["overall_complexity"] = overall_complexity
        all_metrics["complexity_level"] = self._get_complexity_level(overall_complexity)

        # Если есть LLM сервис, добавляем семантический анализ
        if self.llm_service:
            semantic_analysis = await self._get_semantic_analysis(content)
            all_metrics.update(semantic_analysis)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data=all_metrics,
            metadata={"mode": "full_text", "text_length": len(content)}
        )

    def _calculate_linguistic_complexity(self, text: str) -> Dict[str, float]:
        """Вычисление лингвистических метрик сложности"""
        # Подготовка текста
        sentences = self._split_sentences(text)
        words = self._extract_words(text)
        syllables = self._count_syllables_total(words)

        if not sentences or not words:
            return {"flesch_score": 0, "ari_score": 0, "fog_index": 0}

        # Flesch Reading Ease (адаптация для русского языка)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        flesch_score = 206.835 - 1.52 * avg_sentence_length - 65.14 * avg_syllables_per_word
        flesch_normalized = max(0, min(100, flesch_score)) / 100

        # Automated Readability Index
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        ari_score = 4.71 * avg_chars_per_word + 0.5 * avg_sentence_length - 21.43
        ari_normalized = max(0, min(20, ari_score)) / 20

        # Fog Index (адаптация для русского)
        complex_words = self._count_complex_words(words)
        fog_index = 0.4 * (avg_sentence_length + 100 * (complex_words / len(words)))
        fog_normalized = max(0, min(20, fog_index)) / 20

        return {
            "flesch_score": round(flesch_normalized, 3),
            "ari_score": round(ari_normalized, 3),
            "fog_index": round(fog_normalized, 3),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "complex_words_ratio": round(complex_words / len(words), 3)
        }

    def _calculate_structural_complexity(self, text: str) -> Dict[str, float]:
        """Вычисление структурных метрик"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = self._split_sentences(text)

        # Вариативность длины предложений
        sentence_lengths = [len(self._extract_words(s)) for s in sentences]
        if sentence_lengths:
            length_variance = self._calculate_variance(sentence_lengths)
            max_length = max(sentence_lengths)
            min_length = min(sentence_lengths)
            length_ratio = (max_length - min_length) / max(max_length, 1)
        else:
            length_variance = 0
            length_ratio = 0

        return {
            "paragraph_count": len(paragraphs),
            "avg_paragraph_length": round(len(sentences) / max(len(paragraphs), 1), 2),
            "sentence_length_variance": round(length_variance, 3),
            "sentence_length_ratio": round(length_ratio, 3)
        }

    def _calculate_lexical_complexity(self, text: str) -> Dict[str, float]:
        """Вычисление лексических метрик"""
        words = self._extract_words(text.lower())
        if not words:
            return {"lexical_diversity": 0, "rare_words_ratio": 0, "avg_word_length": 0}

        # Лексическое разнообразие (TTR - Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)

        # Доля редких слов (не входящих в список частотных)
        rare_words = [w for w in words if w not in self.common_words]
        rare_words_ratio = len(rare_words) / len(words)

        # Средняя длина слова
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Энтропия лексики
        word_freq = Counter(words)
        total_words = len(words)
        entropy = -sum((freq/total_words) * math.log2(freq/total_words)
                      for freq in word_freq.values())
        entropy_normalized = min(entropy / 10, 1)  # Нормализация

        return {
            "lexical_diversity": round(lexical_diversity, 3),
            "rare_words_ratio": round(rare_words_ratio, 3),
            "avg_word_length": round(avg_word_length, 2),
            "lexical_entropy": round(entropy_normalized, 3),
            "unique_words": len(unique_words),
            "total_words": len(words)
        }

    def _calculate_syntactic_complexity(self, text: str) -> Dict[str, float]:
        """Вычисление синтаксических метрик"""
        sentences = self._split_sentences(text)

        # Подсчет запятых как индикатор сложности синтаксиса
        comma_density = text.count(',') / max(len(text), 1) * 1000  # на 1000 символов

        # Подсчет союзов и связок
        conjunctions = ['что', 'чтобы', 'как', 'когда', 'если', 'хотя', 'поскольку',
                       'потому', 'так как', 'несмотря', 'благодаря']
        conjunction_count = sum(text.lower().count(conj) for conj in conjunctions)
        conjunction_density = conjunction_count / max(len(self._extract_words(text)), 1)

        # Анализ структуры предложений
        complex_sentences = 0
        for sentence in sentences:
            if (sentence.count(',') > 2 or
                any(conj in sentence.lower() for conj in conjunctions) or
                len(self._extract_words(sentence)) > 20):
                complex_sentences += 1

        complex_sentence_ratio = complex_sentences / max(len(sentences), 1)

        return {
            "comma_density": round(comma_density, 2),
            "conjunction_density": round(conjunction_density, 3),
            "complex_sentence_ratio": round(complex_sentence_ratio, 3),
            "avg_commas_per_sentence": round(text.count(',') / max(len(sentences), 1), 2)
        }

    def _calculate_overall_complexity(self, metrics: Dict[str, float]) -> float:
        """Вычисление общего индекса сложности"""
        weights = {
            'flesch_score': -0.3,  # Отрицательный вес (чем выше flesch, тем проще)
            'ari_score': 0.2,
            'fog_index': 0.2,
            'lexical_diversity': 0.15,
            'rare_words_ratio': 0.15,
            'conjunction_density': 0.1,
            'complex_sentence_ratio': 0.1,
            'lexical_entropy': 0.1
        }

        complexity_score = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in metrics:
                complexity_score += metrics[metric] * weight
                total_weight += abs(weight)

        if total_weight > 0:
            normalized_score = abs(complexity_score) / total_weight
        else:
            normalized_score = 0.5

        return round(min(max(normalized_score, 0), 1), 3)

    def _get_complexity_level(self, score: float) -> str:
        """Определение уровня сложности по числовой оценке"""
        if score < 0.3:
            return "elementary"
        elif score < 0.6:
            return "intermediate"
        elif score < 0.8:
            return "advanced"
        else:
            return "expert"

    async def _get_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Семантический анализ с помощью LLM"""
        try:
            prompt = self.prompt_template.format(text=text[:2000])
            response = await self.llm_service.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Ошибка семантического анализа: {e}")
            return {
                "domain_complexity": 0.5,
                "conceptual_difficulty": 0.5,
                "terminology_density": 0.5
            }

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивка текста на предложения"""
        # Простой алгоритм разбивки по знакам препинания
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_words(self, text: str) -> List[str]:
        """Извлечение слов из текста"""
        return re.findall(r'\b[а-яёa-z]+\b', text.lower())

    def _count_syllables_total(self, words: List[str]) -> int:
        """Подсчет общего количества слогов"""
        total = 0
        for word in words:
            total += self._count_syllables_word(word)
        return total

    def _count_syllables_word(self, word: str) -> int:
        """Подсчет слогов в слове (упрощенный алгоритм для русского языка)"""
        vowels = 'аеёиоуыэюя'
        count = 0
        prev_was_vowel = False

        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        return max(count, 1)  # Минимум 1 слог

    def _count_complex_words(self, words: List[str]) -> int:
        """Подсчет сложных слов (более 3 слогов)"""
        return sum(1 for word in words if self._count_syllables_word(word) > 3)

    def _calculate_variance(self, values: List[float]) -> float:
        """Вычисление дисперсии"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

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
        aggregated_metrics = self._aggregate_chunk_metrics(chunk_results)

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
        metrics = {}
        metrics.update(self._calculate_linguistic_complexity(chunk_text))
        metrics.update(self._calculate_lexical_complexity(chunk_text))
        metrics.update(self._calculate_syntactic_complexity(chunk_text))

        overall_complexity = self._calculate_overall_complexity(metrics)
        metrics["overall_complexity"] = overall_complexity
        metrics["complexity_level"] = self._get_complexity_level(overall_complexity)
        metrics["chunk_index"] = chunk_index

        return metrics

    def _aggregate_chunk_metrics(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирование метрик по чанкам"""
        if not chunk_results:
            return {}

        # Усредняем числовые метрики
        numeric_metrics = [
            "flesch_score", "ari_score", "fog_index", "lexical_diversity",
            "rare_words_ratio", "overall_complexity"
        ]

        aggregated = {}
        for metric in numeric_metrics:
            values = [chunk.get(metric, 0) for chunk in chunk_results]
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 3)
                aggregated[f"min_{metric}"] = round(min(values), 3)
                aggregated[f"max_{metric}"] = round(max(values), 3)

        # Общий уровень сложности
        avg_complexity = aggregated.get("avg_overall_complexity", 0.5)
        aggregated["overall_complexity_level"] = self._get_complexity_level(avg_complexity)

        return aggregated

    def estimate_time(self, text: BaseText, mode: AnalysisMode = AnalysisMode.FULL_TEXT) -> float:
        """
        Оценка времени анализа

        Args:
            text: Текст для анализа
            mode: Режим анализа

        Returns:
            float: Оценка времени в секундах
        """
        # Базовое время для NLP вычислений
        base_time = 1.0

        # Время зависит от длины текста
        text_factor = len(text.content) / 5000  # 5000 символов = 1 секунда

        if mode == AnalysisMode.CHUNKED:
            chunking_strategy = ChunkingStrategy()
            estimated_chunks = max(1, len(text.content) // chunking_strategy.chunk_size)
            return (base_time + text_factor) * estimated_chunks * 0.3
        else:
            # Время для семантического анализа с LLM
            llm_time = 2.0 if self.llm_service else 0
            return base_time + text_factor + llm_time

    def get_supported_modes(self) -> list[AnalysisMode]:
        """Получить поддерживаемые режимы анализа"""
        return [AnalysisMode.FULL_TEXT, AnalysisMode.CHUNKED]

    # Реализация абстрактных методов BaseAnalyzer

    @property
    def name(self) -> str:
        """Уникальное имя анализатора"""
        return "complexity"

    @property
    def display_name(self) -> str:
        """Человекочитаемое название"""
        return "Анализ сложности"

    @property
    def description(self) -> str:
        """Описание анализатора"""
        return "Анализирует сложность текста с помощью NLP метрик"

    @property
    def requires_llm(self) -> bool:
        """Требует ли анализатор LLM"""
        return False  # Основная функциональность работает без LLM

    def interpret_results(self, result: AnalysisResult) -> str:
        """Интерпретация результатов анализа"""
        data = result.result_data

        if "error" in data:
            return f"Ошибка анализа сложности: {data['error']}"

        overall_complexity = data.get("overall_complexity", 0)
        complexity_level = data.get("complexity_level", "неизвестно")

        flesch_score = data.get("flesch_score", 0)
        lexical_diversity = data.get("lexical_diversity", 0)
        rare_words_ratio = data.get("rare_words_ratio", 0)

        interpretation = f"Сложность текста: {complexity_level} "
        interpretation += f"(общая оценка: {overall_complexity:.3f})\n"
        interpretation += f"Читаемость по Флешу: {flesch_score:.3f}\n"
        interpretation += f"Лексическое разнообразие: {lexical_diversity:.3f}\n"
        interpretation += f"Доля сложных слов: {rare_words_ratio:.1%}\n"

        # Дополнительные детали если доступны
        if "avg_sentence_length" in data:
            interpretation += f"Средняя длина предложений: {data['avg_sentence_length']:.1f} слов\n"

        return interpretation.strip()