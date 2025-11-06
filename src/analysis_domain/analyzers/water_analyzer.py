"""
Анализатор "воды" в тексте (water level)
"""
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from src.text_domain.entities.base_text import BaseText
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class WaterAnalyzer(BaseAnalyzer):
    """
    Анализатор "воды" (избыточного контента)

    НЕ ТРЕБУЕТ LLM - очень быстрый статистический анализ!
    НЕ ТРЕБУЕТ embeddings - использует статистические метрики

    Использует:
    - Повторы слов и фраз (высокая частота = вода)
    - Длину предложений (очень длинные/короткие = вода)
    - Плотность уникальных слов
    - Соотношение служебных к значимым словам

    Возвращает:
    - Процент "воды" (0-100%)
    - Информационную плотность
    - Рейтинг (concise/balanced/verbose)
    - Статистику по повторам
    """

    def __init__(self):
        """Инициализация анализатора воды"""
        pass

    @property
    def name(self) -> str:
        """Уникальное имя анализатора"""
        return "water"

    @property
    def display_name(self) -> str:
        """Человекочитаемое название"""
        return "Анализ воды в тексте"

    @property
    def description(self) -> str:
        """Описание анализатора"""
        return "Определяет уровень 'воды' (избыточного контента) в тексте на основе семантической избыточности"

    @property
    def requires_llm(self) -> bool:
        """НЕ требуется LLM"""
        return False

    @property
    def requires_embeddings(self) -> bool:
        """НЕ требуются embeddings - статистический анализ"""
        return False

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode,
        **kwargs
    ) -> AnalysisResult:
        """
        Выполнить анализ воды

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры (должны содержать embeddings)

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа воды: {text.title}")

            # Получаем чанки и embeddings
            chunks = kwargs.get('chunks', [])
            embeddings = kwargs.get('embeddings', [])

            if not chunks:
                raise AnalysisError(
                    "Анализ воды требует чанки текста",
                    details={"text_id": text.id}
                )

            if embeddings and len(embeddings) > 0:
                # Используем embeddings для точного анализа
                water_data = self._analyze_with_embeddings(chunks, embeddings)
            else:
                # Fallback на эвристику
                logger.warning("Embeddings недоступны, используем эвристику")
                water_data = self._analyze_with_heuristics(chunks)

            logger.info(f"Анализ воды завершён: {water_data['water_percentage']:.1f}%")

            return AnalysisResult(
                text_id=text.id,
                analyzer_name=self.name,
                mode=mode,
                data=water_data
            )

        except Exception as e:
            logger.error(f"Ошибка анализа воды: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать воду: {str(e)}",
                details={"text_id": text.id}
            )

    def _analyze_with_embeddings(
        self,
        chunks: List[Any],
        embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Анализировать воду используя semantic embeddings

        Args:
            chunks: Список чанков
            embeddings: Список векторов

        Returns:
            Dict: Результаты анализа
        """
        if len(embeddings) != len(chunks):
            logger.warning("Несоответствие количества embeddings, fallback")
            return self._analyze_with_heuristics(chunks)

        # Конвертируем в numpy array
        emb_matrix = np.array([emb for emb in embeddings])

        # 1. Вычисляем семантическую вариативность для каждого чанка
        # Низкая вариативность = низкая информационная плотность
        chunk_variances = []
        for emb in emb_matrix:
            variance = np.var(emb)
            chunk_variances.append(variance)

        # Нормализуем вариативность
        max_var = max(chunk_variances) if chunk_variances else 1.0
        normalized_variances = [v / max_var for v in chunk_variances]

        # 2. Вычисляем избыточность (схожесть соседних чанков)
        redundancy_scores = []
        for i in range(len(emb_matrix) - 1):
            similarity = self._cosine_similarity(emb_matrix[i], emb_matrix[i+1])
            redundancy_scores.append(similarity)

        avg_redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0

        # 3. Вычисляем информационную плотность
        # Высокая вариативность + низкая избыточность = высокая плотность
        info_density_scores = []
        for i, variance in enumerate(normalized_variances):
            if i < len(redundancy_scores):
                # Низкая избыточность с соседним = более уникальная информация
                uniqueness = 1.0 - redundancy_scores[i]
                density = (variance + uniqueness) / 2.0
            else:
                density = variance

            info_density_scores.append(density)

        # Средняя информационная плотность
        avg_info_density = sum(info_density_scores) / len(info_density_scores)

        # Процент воды = обратная информационная плотность
        water_percentage = (1.0 - avg_info_density) * 100

        # Находим "водянистые" чанки (нижние 20%)
        sorted_indices = sorted(
            range(len(info_density_scores)),
            key=lambda i: info_density_scores[i]
        )
        verbose_count = max(int(len(chunks) * 0.2), 1)
        verbose_chunks = sorted_indices[:verbose_count]

        # Рейтинг
        if water_percentage < 20:
            rating = "concise"
            rating_ru = "лаконичный"
            rating_emoji = "✨"
        elif water_percentage < 50:
            rating = "balanced"
            rating_ru = "сбалансированный"
            rating_emoji = "✅"
        else:
            rating = "verbose"
            rating_ru = "многословный"
            rating_emoji = "💧"

        return {
            "water_percentage": round(water_percentage, 2),
            "info_density": round(avg_info_density, 2),
            "rating": rating,
            "rating_ru": rating_ru,
            "rating_emoji": rating_emoji,
            "verbose_chunks": verbose_chunks,
            "avg_redundancy": round(avg_redundancy, 2)
        }

    def _analyze_with_heuristics(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Анализировать воду используя эвристику

        Args:
            chunks: Список чанков

        Returns:
            Dict: Результаты анализа
        """

        # Обогащенные словари для обнаружения "водности"
        water_patterns = {
            "слова_паразиты": {
                "как бы", "типа", "короче", "вообще", "в общем", "так сказать", "можно сказать",
                "если честно", "честно говоря", "прямо скажем", "надо сказать", "скажем так",
                "в принципе", "в основном", "по большому счету", "если можно так выразиться",
                "собственно говоря", "если разобраться", "как говорится", "между прочим",
                "в некотором смысле", "в каком-то смысле", "до определенной степени"
            },
            "канцеляризмы": {
                "в настоящее время", "на сегодняшний день", "в данный момент времени",
                "по состоянию на", "в течение длительного времени", "в ходе проведения",
                "с целью осуществления", "для обеспечения возможности", "в процессе реализации",
                "при условии соблюдения", "в рамках выполнения", "согласно установленным требованиям"
            },
            "избыточные_конструкции": {
                "дело в том что", "суть дела заключается в том", "хочется отметить что",
                "следует подчеркнуть что", "важно понимать что", "нельзя не сказать что",
                "стоит обратить внимание на то что", "необходимо отметить что",
                "хотелось бы подчеркнуть что", "не могу не отметить что"
            },
            "тавтологии": {
                "главная суть", "свободная вакансия", "прейскурант цен", "памятный сувенир",
                "совместное сотрудничество", "взаимное сотрудничество", "первая премьера",
                "старый ветеран", "молодой юноша", "народная демократия", "бесплатный подарок",
                "окружающая среда вокруг", "период времени", "огромная махина"
            },
            "штампы_клише": {
                "само собой разумеется", "без всякого сомнения", "как известно",
                "не секрет что", "всем известно что", "играет важную роль",
                "имеет большое значение", "нетрудно догадаться", "легко понять",
                "очевидно что", "понятно что", "ясно что"
            }
        }

        # Анализируем текст по чанкам
        total_words = 0
        total_water_words = 0
        water_examples = {category: [] for category in water_patterns.keys()}
        word_frequencies = {}

        full_text = " ".join(chunk.content for chunk in chunks).lower()

        # Подсчитываем слова
        words = full_text.split()
        total_words = len(words)

        # Считаем частоты слов для обнаружения повторов
        for word in words:
            if len(word) > 3:  # Игнорируем короткие слова
                word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Ищем паттерны водности
        for category, patterns in water_patterns.items():
            for pattern in patterns:
                count = full_text.count(pattern)
                if count > 0:
                    water_examples[category].append((pattern, count))
                    total_water_words += count * len(pattern.split())

        # Считаем повторяющиеся слова (>5 раз = вода)
        repetitive_words = sum(
            count for count in word_frequencies.values() if count > 5
        )

        # Обнаружение семантических повторов (описание одного и того же разными словами)
        semantic_repeats = self._detect_semantic_repetitions(chunks)
        semantic_water_percentage = (len(semantic_repeats) / max(len(chunks), 1)) * 100

        # Рассчитываем процент водности
        if total_words == 0:
            water_percentage = 50.0
        else:
            # Комбинированная оценка: паттерны + повторы + семантические повторы
            pattern_water = (total_water_words / total_words) * 100
            repetition_water = (repetitive_words / total_words) * 100
            water_percentage = min(pattern_water + repetition_water + semantic_water_percentage, 100)

        # Находим топ повторяющихся слов
        top_repetitive = sorted(
            [(word, count) for word, count in word_frequencies.items() if count > 3],
            key=lambda x: x[1], reverse=True
        )[:10]

        # Рейтинг
        if water_percentage < 20:
            rating = "concise"
            rating_ru = "лаконичный"
            rating_emoji = "✨"
        elif water_percentage < 50:
            rating = "balanced"
            rating_ru = "сбалансированный"
            rating_emoji = "✅"
        else:
            rating = "verbose"
            rating_ru = "многословный"
            rating_emoji = "💧"

        return {
            "water_percentage": round(water_percentage, 2),
            "info_density": round(1.0 - (water_percentage / 100), 2),
            "rating": rating,
            "rating_ru": rating_ru,
            "rating_emoji": rating_emoji,
            "verbose_chunks": [],
            "avg_redundancy": 0.0,
            "method": "enhanced_heuristic",
            "water_examples": water_examples,
            "top_repetitive_words": top_repetitive,
            "semantic_repeats": semantic_repeats,
            "pattern_water_percentage": round((total_water_words / max(total_words, 1)) * 100, 2),
            "repetition_water_percentage": round((repetitive_words / max(total_words, 1)) * 100, 2),
            "semantic_water_percentage": round(semantic_water_percentage, 2)
        }

    def _detect_semantic_repetitions(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """
        Обнаружить семантические повторы - когда одно и то же описывается разными словами

        Args:
            chunks: Список чанков

        Returns:
            List: Список найденных семантических повторов
        """
        semantic_repeats = []

        # Паттерны для обнаружения семантических повторов
        repetition_patterns = {
            "повтор_идеи": [
                # Паттерны, указывающие на переформулировку
                "другими словами", "иными словами", "то есть", "а именно",
                "проще говоря", "говоря иначе", "можно сказать", "иначе говоря"
            ],
            "избыточные_объяснения": [
                # Лишние объяснения очевидного
                "как известно", "само собой разумеется", "очевидно что", "понятно что",
                "не нужно объяснять", "всем ясно", "не секрет", "ни для кого не секрет"
            ],
            "дублирование_действий": [
                # Описание одного действия несколько раз
                "опять", "снова", "еще раз", "повторно", "опять же", "по-прежнему",
                "как и прежде", "как обычно", "как всегда"
            ]
        }

        # Анализируем соседние чанки на предмет семантических повторов
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].content.lower()
            chunk2 = chunks[i + 1].content.lower()

            # Ищем паттерны переформулировки
            for category, patterns in repetition_patterns.items():
                for pattern in patterns:
                    if pattern in chunk1 or pattern in chunk2:
                        # Проверяем схожесть ключевых слов
                        similarity_score = self._calculate_keyword_similarity(chunk1, chunk2)
                        if similarity_score > 0.3:  # Порог схожести
                            semantic_repeats.append({
                                "type": category,
                                "chunks": [i, i + 1],
                                "pattern": pattern,
                                "similarity": similarity_score,
                                "description": f"Возможное семантическое повторение через '{pattern}'"
                            })

            # Проверяем повторение ключевых существительных и глаголов
            keywords1 = self._extract_keywords(chunk1)
            keywords2 = self._extract_keywords(chunk2)

            common_keywords = set(keywords1) & set(keywords2)
            if len(common_keywords) >= 3:  # Если 3+ общих ключевых слова
                semantic_repeats.append({
                    "type": "keyword_repetition",
                    "chunks": [i, i + 1],
                    "common_keywords": list(common_keywords),
                    "description": f"Повторение ключевых понятий: {', '.join(list(common_keywords)[:3])}"
                })

        return semantic_repeats

    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Вычислить схожесть текстов по ключевым словам

        Args:
            text1: Первый текст
            text2: Второй текст

        Returns:
            float: Коэффициент схожести 0-1
        """
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))

        if not keywords1 or not keywords2:
            return 0.0

        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)

        return intersection / union if union > 0 else 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Извлечь ключевые слова из текста (существительные, глаголы, прилагательные)

        Args:
            text: Текст для анализа

        Returns:
            List: Список ключевых слов
        """
        # Простая эвристика: слова длиннее 4 символов, исключая служебные
        stop_words = {
            'что', 'как', 'это', 'все', 'для', 'она', 'они', 'его', 'был', 'была',
            'были', 'есть', 'может', 'быть', 'сказал', 'сказала', 'один', 'одна',
            'тот', 'том', 'той', 'тем', 'при', 'над', 'под', 'без', 'через'
        }

        words = text.split()
        keywords = []

        for word in words:
            # Убираем пунктуацию
            clean_word = ''.join(c for c in word if c.isalpha()).lower()
            if (len(clean_word) > 4 and
                clean_word not in stop_words and
                clean_word.isalpha()):
                keywords.append(clean_word)

        return keywords

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычислить косинусное сходство

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            float: Косинусное сходство (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        water_percent = data.get('water_percentage', 0)
        info_density = data.get('info_density', 0)
        rating_ru = data.get('rating_ru', 'неизвестный')
        rating_emoji = data.get('rating_emoji', '💧')

        lines = [
            f'💧 **Анализ "воды" в тексте**\n',
            f"{rating_emoji} **Уровень воды**: {water_percent:.1f}%",
            f"📊 **Информационная плотность**: {info_density:.2f}",
            f"⭐ **Рейтинг**: {rating_ru}\n"
        ]

        # Интерпретация
        if water_percent < 20:
            interpretation = (
                "Очень лаконичный текст. Высокая информационная плотность. "
                "Каждое предложение несёт смысл. Отличный результат!"
            )
        elif water_percent < 35:
            interpretation = (
                "Хороший баланс между информацией и читабельностью. "
                "Умеренное количество повторов и описаний."
            )
        elif water_percent < 50:
            interpretation = (
                "Средний уровень воды. Есть повторы и избыточные описания, "
                "но они не критичны."
            )
        elif water_percent < 70:
            interpretation = (
                'Много "воды". Значительное количество повторов, избыточных '
                "описаний и малоинформативных фрагментов. Рекомендуется редактура."
            )
        else:
            interpretation = (
                "Очень высокий уровень воды. Текст перегружен повторами "
                "и избыточными описаниями. Требуется серьёзная редактура."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}")

        return '\n'.join(lines)
