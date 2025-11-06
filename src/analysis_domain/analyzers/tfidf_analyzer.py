"""
TF-IDF анализатор текста
"""
from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from src.text_domain.services.chunking_service import ChunkingService
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class TfidfAnalyzer(BaseAnalyzer):
    """
    Анализатор на основе TF-IDF (Term Frequency-Inverse Document Frequency)

    Статистический метод для определения важности слов в тексте.
    Не требует LLM, работает быстро.
    """

    def __init__(self):
        """Инициализация анализатора"""
        self.chunking_service = ChunkingService()

    @property
    def name(self) -> str:
        return "tfidf"

    @property
    def display_name(self) -> str:
        return "TF-IDF анализ"

    @property
    def description(self) -> str:
        return """Статистический анализ важности слов в тексте.
Выявляет ключевые термины и их значимость на основе частоты встречаемости."""

    @property
    def requires_llm(self) -> bool:
        return False

    @property
    def requires_embeddings(self) -> bool:
        return False

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Анализировать текст с помощью TF-IDF

        Args:
            text: Текст для анализа
            mode: Режим анализа
            chunking_strategy: Стратегия чанковки
            context: Дополнительный контекст

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            import time
            start_time = time.time()

            # Валидация режима
            self.validate_mode(mode)

            # Получение контента
            content = await text.get_content()

            if mode == AnalysisMode.FULL_TEXT:
                result_data = await self._analyze_full_text(content)
            else:
                if not chunking_strategy:
                    raise AnalysisError(
                        message="Chunking strategy required for CHUNKED mode",
                        details={"analyzer": self.name}
                    )
                result_data = await self._analyze_chunked(content, chunking_strategy)

            # Вычисление времени выполнения
            execution_time = (time.time() - start_time) * 1000

            # Создание результата
            result = AnalysisResult(
                text_id=text.id,
                analyzer_name=self.name,
                data=result_data,
                execution_time_ms=execution_time,
                mode=mode.value,
            )

            # Генерация интерпретации
            result.interpretation = self.interpret_results(result)

            logger.debug(
                f"TF-IDF анализ завершён для текста {text.id} "
                f"за {execution_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка TF-IDF анализа: {e}")
            raise AnalysisError(
                message=f"TF-IDF analysis failed: {e}",
                details={"text_id": text.id, "error": str(e)}
            )

    async def _analyze_full_text(self, content: str) -> Dict[str, Any]:
        """
        Анализ полного текста

        Args:
            content: Содержимое текста

        Returns:
            Dict: Результаты анализа
        """

        # Обогащенный список русских стоп-слов
        russian_stop_words = [
            # Предлоги
            'в', 'во', 'на', 'за', 'к', 'ко', 'с', 'со', 'по', 'из', 'от', 'до', 'для', 'при', 'о', 'об', 'под', 'над', 'между', 'перед',
            # Союзы
            'и', 'а', 'но', 'или', 'да', 'что', 'чтобы', 'если', 'когда', 'как', 'так', 'то', 'ни', 'либо', 'хотя', 'потому', 'поэтому',
            # Частицы
            'не', 'ни', 'же', 'ли', 'бы', 'вот', 'вон', 'ведь', 'уж', 'ну', 'даже', 'лишь', 'только', 'еще', 'уже', 'тоже', 'также',
            # Местоимения
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'себя', 'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их', 'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
            'мой', 'твой', 'свой', 'наш', 'ваш', 'этот', 'тот', 'такой', 'весь', 'всякий', 'каждый', 'любой', 'другой', 'иной', 'сам',
            'кто', 'что', 'какой', 'который', 'чей', 'где', 'куда', 'откуда', 'когда', 'зачем', 'почему', 'как', 'сколько',
            # Вспомогательные глаголы и связки
            'быть', 'есть', 'был', 'была', 'было', 'были', 'будет', 'будут', 'бывает', 'стать', 'стал', 'стала', 'стало', 'стали',
            # Наречия общего значения
            'очень', 'более', 'менее', 'самый', 'наиболее', 'наименее', 'слишком', 'довольно', 'весьма', 'почти', 'совсем', 'вполне',
            'здесь', 'там', 'тут', 'всюду', 'везде', 'нигде', 'куда-то', 'где-то', 'откуда-то', 'туда', 'сюда', 'оттуда', 'отсюда',
            'теперь', 'сейчас', 'тогда', 'всегда', 'никогда', 'иногда', 'часто', 'редко', 'скоро', 'давно', 'недавно', 'сегодня', 'вчера', 'завтра',
            # Модальные слова
            'может', 'может быть', 'наверное', 'вероятно', 'возможно', 'конечно', 'разумеется', 'безусловно', 'несомненно', 'действительно',
            # Вводные слова
            'кстати', 'например', 'впрочем', 'однако', 'наконец', 'итак', 'значит', 'следовательно', 'таким образом', 'во-первых', 'во-вторых',
            # Междометия
            'ах', 'ох', 'эх', 'ну', 'да', 'нет', 'ага', 'угу', 'ой', 'ай', 'эй',
            # Числительные
            'один', 'одна', 'одно', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
            'первый', 'второй', 'третий', 'последний', 'много', 'мало', 'несколько', 'столько', 'сколько',
            # Служебные конструкции
            'то есть', 'а именно', 'в том числе', 'в частности', 'в основном', 'в общем', 'в целом', 'по крайней мере',
            'так или иначе', 'так сказать', 'можно сказать', 'надо сказать', 'стоит отметить',
        ]

        # Создаём TF-IDF векторизатор
        vectorizer = TfidfVectorizer(
            max_features=100,  # Топ-100 слов
            stop_words=russian_stop_words,  # Обогащенные русские стоп-слова
            ngram_range=(1, 2),  # Униграммы и биграммы
            min_df=1,
            max_df=1.0,  # Для одного документа должно быть 1.0
            token_pattern=r'\b[а-яё]{2,}\b'  # Только русские слова длиной от 2 символов
        )

        # Получаем TF-IDF матрицу
        tfidf_matrix = vectorizer.fit_transform([content])
        feature_names = vectorizer.get_feature_names_out()

        # Получаем веса слов
        scores = tfidf_matrix.toarray()[0]

        # Сортируем по убыванию важности
        top_indices = scores.argsort()[-20:][::-1]
        top_terms = [
            {
                "term": feature_names[i],
                "score": float(scores[i]),
                "rank": rank + 1
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

        # Статистика
        total_terms = len([s for s in scores if s > 0])
        avg_score = float(np.mean(scores[scores > 0])) if total_terms > 0 else 0.0
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        return {
            "top_terms": top_terms,
            "total_terms": total_terms,
            "avg_score": avg_score,
            "max_score": max_score,
            "vocabulary_size": len(feature_names),
        }

    async def _analyze_chunked(
        self,
        content: str,
        strategy: ChunkingStrategy
    ) -> Dict[str, Any]:
        """
        Анализ по чанкам

        Args:
            content: Содержимое текста
            strategy: Стратегия чанковки

        Returns:
            Dict: Результаты анализа
        """
        # Разбиваем на чанки
        chunks = await self.chunking_service.chunk_text(content, strategy)

        # Анализируем каждый чанк
        chunk_results = []
        all_terms = {}

        for chunk in chunks:
            chunk_data = await self._analyze_full_text(chunk.content)
            chunk_results.append({
                "chunk_index": chunk.chunk_index,
                "top_terms": chunk_data["top_terms"][:5],  # Топ-5 для каждого чанка
            })

            # Агрегируем термины
            for term_data in chunk_data["top_terms"]:
                term = term_data["term"]
                score = term_data["score"]
                if term in all_terms:
                    all_terms[term] += score
                else:
                    all_terms[term] = score

        # Сортируем агрегированные термины
        sorted_terms = sorted(
            [{"term": term, "score": score} for term, score in all_terms.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:20]

        return {
            "chunk_count": len(chunks),
            "chunk_results": chunk_results,
            "aggregated_top_terms": sorted_terms,
            "total_unique_terms": len(all_terms),
        }

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты TF-IDF анализа

        Args:
            result: Результат анализа

        Returns:
            str: Текстовая интерпретация
        """
        data = result.data

        if result.mode == "full_text":
            top_terms = data.get("top_terms", [])
            if not top_terms:
                return "Ключевые термины не найдены."

            lines = ["📊 Ключевые термины и их значимость:\n"]

            for i, term_data in enumerate(top_terms[:10], 1):
                term = term_data["term"]
                score = term_data["score"]
                lines.append(f"{i}. '{term}' — {score:.3f}")

            lines.append(f"\nВсего уникальных терминов: {data.get('total_terms', 0)}")

            return "\n".join(lines)

        else:  # chunked mode
            chunk_count = data.get("chunk_count", 0)
            aggregated = data.get("aggregated_top_terms", [])

            if not aggregated:
                return f"Текст разбит на {chunk_count} чанков, но ключевые термины не найдены."

            lines = [
                f"📊 TF-IDF анализ по {chunk_count} чанкам:\n",
                "Топ-10 агрегированных терминов:\n"
            ]

            for i, term_data in enumerate(aggregated[:10], 1):
                term = term_data["term"]
                score = term_data["score"]
                lines.append(f"{i}. '{term}' — {score:.3f}")

            lines.append(f"\nУникальных терминов: {data.get('total_unique_terms', 0)}")

            return "\n".join(lines)

    def get_estimated_time(
        self,
        text_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """
        Оценка времени выполнения

        Args:
            text_length: Длина текста
            mode: Режим анализа

        Returns:
            float: Время в секундах
        """
        # TF-IDF быстрый
        base_time = 0.5
        time_per_1k = 0.05
        return base_time + (text_length / 1000) * time_per_1k
