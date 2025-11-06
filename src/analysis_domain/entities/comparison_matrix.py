"""
Матрица сравнений между текстами
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .comparison_result import ComparisonResult
from src.common.utils import now_utc


@dataclass
class ComparisonMatrix:
    """
    Матрица попарных сравнений текстов (N x N)

    Хранит результаты сравнения всех пар текстов
    """

    # Размер матрицы (количество текстов)
    size: int

    # ID текстов (порядок важен для индексации)
    text_ids: List[str]

    # Матрица результатов
    # Ключ: "text1_id:text2_id" -> ComparisonResult
    results: Dict[str, ComparisonResult] = field(default_factory=dict)

    # Временная метка
    created_at: datetime = field(default_factory=now_utc)

    def _get_key(self, text1_id: str, text2_id: str) -> str:
        """
        Получить ключ для пары текстов

        Args:
            text1_id: ID первого текста
            text2_id: ID второго текста

        Returns:
            str: Ключ
        """
        return f"{text1_id}:{text2_id}"

    def set(self, i: int, j: int, result: ComparisonResult) -> None:
        """
        Установить результат сравнения для пары (i, j)

        Args:
            i: Индекс первого текста
            j: Индекс второго текста
            result: Результат сравнения
        """
        if i >= self.size or j >= self.size:
            raise IndexError(f"Index out of bounds: ({i}, {j}) for size {self.size}")

        text1_id = self.text_ids[i]
        text2_id = self.text_ids[j]
        key = self._get_key(text1_id, text2_id)

        self.results[key] = result

    def get(self, i: int, j: int) -> Optional[ComparisonResult]:
        """
        Получить результат сравнения для пары (i, j)

        Args:
            i: Индекс первого текста
            j: Индекс второго текста

        Returns:
            Optional[ComparisonResult]: Результат или None
        """
        if i >= self.size or j >= self.size:
            return None

        text1_id = self.text_ids[i]
        text2_id = self.text_ids[j]
        key = self._get_key(text1_id, text2_id)

        return self.results.get(key)

    def get_by_ids(self, text1_id: str, text2_id: str) -> Optional[ComparisonResult]:
        """
        Получить результат сравнения по ID текстов

        Args:
            text1_id: ID первого текста
            text2_id: ID второго текста

        Returns:
            Optional[ComparisonResult]: Результат или None
        """
        key = self._get_key(text1_id, text2_id)
        return self.results.get(key)

    def get_similarity_matrix(self) -> List[List[float]]:
        """
        Получить матрицу оценок сходства (числовую)

        Returns:
            List[List[float]]: Матрица N x N с оценками сходства
        """
        matrix = [[0.0 for _ in range(self.size)] for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    matrix[i][j] = 1.0  # Текст идентичен самому себе
                else:
                    result = self.get(i, j)
                    if result:
                        matrix[i][j] = result.similarity_score

        return matrix

    def get_most_similar_pairs(self, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Получить топ-K наиболее похожих пар текстов

        Args:
            top_k: Количество пар

        Returns:
            List[Tuple[str, str, float]]: Список (text1_id, text2_id, score)
        """
        pairs = []

        for key, result in self.results.items():
            text1_id, text2_id = key.split(":")
            # Пропускаем сравнение текста с самим собой
            if text1_id != text2_id:
                pairs.append((text1_id, text2_id, result.similarity_score))

        # Сортировка по убыванию сходства
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[:top_k]

    def get_least_similar_pairs(self, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Получить топ-K наименее похожих пар текстов

        Args:
            top_k: Количество пар

        Returns:
            List[Tuple[str, str, float]]: Список (text1_id, text2_id, score)
        """
        pairs = []

        for key, result in self.results.items():
            text1_id, text2_id = key.split(":")
            if text1_id != text2_id:
                pairs.append((text1_id, text2_id, result.similarity_score))

        # Сортировка по возрастанию сходства
        pairs.sort(key=lambda x: x[2])

        return pairs[:top_k]

    def get_average_similarity(self) -> float:
        """
        Получить среднее сходство по всем парам

        Returns:
            float: Среднее сходство
        """
        if not self.results:
            return 0.0

        scores = []
        for key, result in self.results.items():
            text1_id, text2_id = key.split(":")
            # Пропускаем диагональ
            if text1_id != text2_id:
                scores.append(result.similarity_score)

        return sum(scores) / len(scores) if scores else 0.0

    def get_text_average_similarity(self, text_id: str) -> float:
        """
        Получить среднее сходство конкретного текста со всеми остальными

        Args:
            text_id: ID текста

        Returns:
            float: Среднее сходство
        """
        scores = []

        for other_id in self.text_ids:
            if other_id != text_id:
                result = self.get_by_ids(text_id, other_id)
                if result:
                    scores.append(result.similarity_score)

        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные матрицы
        """
        return {
            "size": self.size,
            "text_ids": self.text_ids,
            "total_comparisons": len(self.results),
            "similarity_matrix": self.get_similarity_matrix(),
            "average_similarity": self.get_average_similarity(),
            "most_similar_pairs": [
                {
                    "text1_id": t1,
                    "text2_id": t2,
                    "similarity": score
                }
                for t1, t2, score in self.get_most_similar_pairs(5)
            ],
            "least_similar_pairs": [
                {
                    "text1_id": t1,
                    "text2_id": t2,
                    "similarity": score
                }
                for t1, t2, score in self.get_least_similar_pairs(5)
            ],
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self) -> str:
        return f"ComparisonMatrix(size={self.size}x{self.size}, comparisons={len(self.results)})"

    def __repr__(self) -> str:
        return self.__str__()
