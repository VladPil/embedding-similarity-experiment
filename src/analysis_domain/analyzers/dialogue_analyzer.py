"""
Анализатор диалогов в тексте с использованием NLP
"""
import re
import json
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class DialogueAnalyzer(BaseAnalyzer):
    """
    Анализатор диалогов в тексте

    Выполняет анализ диалогов и речи персонажей:
    - Обнаружение прямой речи и диалогов
    - Анализ характеристик речи персонажей
    - Определение стилистики диалогов
    - Подсчет статистики диалогов
    - Анализ речевых паттернов
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        super().__init__(
            name="DialogueAnalyzer",
            description="Анализирует диалоги и прямую речь в тексте"
        )
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._get_default_prompt()

        # Паттерны для обнаружения прямой речи
        self.speech_patterns = [
            r'[—–]\s*([^—–\n]+?)(?=[.!?\n])',  # Тире
            r'["«]([^"»]+)["»]',                # Кавычки
            r':\s*[—–]\s*([^—–\n]+?)(?=[.!?\n])',  # Двоеточие + тире
        ]

        # Паттерны для обнаружения слов автора
        self.author_patterns = [
            r'(сказал|сказала|произнес|произнесла|ответил|ответила|спросил|спросила|воскликнул|воскликнула)',
            r'(проговорил|проговорила|пробормотал|пробормотала|прошептал|прошептала|крикнул|крикнула)',
            r'(заметил|заметила|добавил|добавила|продолжил|продолжила|уточнил|уточнила)',
            r'(промолвил|промолвила|буркнул|буркнула|бросил|бросила|выпалил|выпалила)'
        ]

        # Эмоциональные маркеры в речи
        self.emotion_markers = {
            'радость': ['ха-ха', 'хи-хи', 'ого', 'ура', 'прекрасно', 'замечательно', 'отлично'],
            'грусть': ['ох', 'ах', 'увы', 'печально', 'грустно', 'жаль'],
            'удивление': ['что?', 'как?', 'неужели', 'невероятно', 'ого', 'ба'],
            'гнев': ['черт', 'дьявол', 'проклятье', 'ненавижу', 'бесит', 'злит'],
            'страх': ['боже', 'ужас', 'страшно', 'боюсь', 'кошмар'],
            'сомнение': ['может', 'возможно', 'наверное', 'вроде', 'похоже']
        }

        # Речевые особенности
        self.speech_features = {
            'формальность': ['господин', 'госпожа', 'уважаемый', 'позвольте', 'извините'],
            'разговорность': ['ну', 'вот', 'типа', 'короче', 'слушай', 'блин'],
            'архаизмы': ['сие', 'оно', 'дабы', 'коли', 'ежели', 'посему'],
            'просторечие': ['чё', 'шо', 'када', 'тады', 'энтот']
        }

    def _get_default_prompt(self) -> PromptTemplate:
        """Получить промпт по умолчанию для LLM анализа"""
        template = """
        Проанализируй диалоги в данном тексте:
        1. Характеристики речи персонажей
        2. Эмоциональную окраску диалогов
        3. Стилистические особенности

        Текст: {text}

        Верни результат в формате JSON:
        {{
            "dialogue_quality": float (0-1),
            "character_voices": ["список характеристик голосов персонажей"],
            "dialogue_style": "formal|informal|mixed",
            "emotional_range": ["список эмоций в диалогах"],
            "authenticity_score": float (0-1),
            "recommendations": ["рекомендации по улучшению диалогов"]
        }}
        """
        return PromptTemplate(
            template=template,
            variables=["text"],
            name="dialogue_analysis_prompt"
        )

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        **kwargs
    ) -> AnalysisResult:
        """
        Анализ диалогов в тексте

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа диалогов
        """
        try:
            logger.info(f"Начинаем анализ диалогов в тексте: {text.title}")

            if mode == AnalysisMode.CHUNKED:
                return await self._analyze_chunks(text, **kwargs)
            else:
                return await self._analyze_full_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка анализа диалогов: {e}")
            raise AnalysisError(f"Не удалось проанализировать диалоги в тексте: {e}")

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

        # Извлечение диалогов
        dialogues = self._extract_dialogues(content)

        # Базовая статистика диалогов
        basic_stats = self._calculate_basic_dialogue_stats(content, dialogues)

        # Анализ структуры диалогов
        structure_analysis = self._analyze_dialogue_structure(dialogues, content)

        # Анализ речевых характеристик
        speech_analysis = self._analyze_speech_characteristics(dialogues)

        # Анализ эмоциональности
        emotion_analysis = self._analyze_dialogue_emotions(dialogues)

        # Анализ стилистики
        style_analysis = self._analyze_dialogue_style(dialogues)

        # Анализ аутентичности
        authenticity_analysis = self._analyze_dialogue_authenticity(dialogues)

        # Объединение всех результатов
        all_metrics = {
            **basic_stats,
            **structure_analysis,
            **speech_analysis,
            **emotion_analysis,
            **style_analysis,
            **authenticity_analysis
        }

        # Общая оценка качества диалогов
        overall_score = self._calculate_overall_dialogue_score(all_metrics)
        all_metrics["overall_dialogue_quality"] = overall_score
        all_metrics["dialogue_grade"] = self._get_dialogue_grade(overall_score)

        # Рекомендации
        all_metrics["automated_recommendations"] = self._generate_dialogue_recommendations(all_metrics)

        # LLM анализ если доступен
        if self.llm_service and dialogues:
            llm_analysis = await self._get_llm_dialogue_analysis(dialogues[:3])  # Анализируем первые 3 диалога
            all_metrics.update(llm_analysis)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data=all_metrics,
            metadata={
                "mode": "full_text",
                "text_length": len(content),
                "dialogues_found": len(dialogues)
            }
        )

    def _extract_dialogues(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение диалогов из текста"""
        dialogues = []

        # Разбиваем текст на абзацы
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            dialogue_data = self._analyze_paragraph_for_dialogue(paragraph, i)
            if dialogue_data:
                dialogues.append(dialogue_data)

        return dialogues

    def _analyze_paragraph_for_dialogue(self, paragraph: str, index: int) -> Optional[Dict[str, Any]]:
        """Анализ абзаца на предмет содержания диалога"""
        dialogue_info = {
            "paragraph_index": index,
            "text": paragraph,
            "speech_parts": [],
            "author_words": [],
            "has_dialogue": False
        }

        # Поиск прямой речи
        for pattern in self.speech_patterns:
            matches = re.finditer(pattern, paragraph)
            for match in matches:
                speech_text = match.group(1) if match.groups() else match.group(0)
                dialogue_info["speech_parts"].append({
                    "text": speech_text.strip(),
                    "position": match.span(),
                    "length": len(speech_text.strip())
                })
                dialogue_info["has_dialogue"] = True

        # Поиск слов автора
        for pattern in self.author_patterns:
            matches = re.finditer(pattern, paragraph, re.IGNORECASE)
            for match in matches:
                dialogue_info["author_words"].append({
                    "word": match.group(0),
                    "position": match.span()
                })

        return dialogue_info if dialogue_info["has_dialogue"] else None

    def _calculate_basic_dialogue_stats(self, text: str, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вычисление базовой статистики диалогов"""
        if not dialogues:
            return {
                "dialogue_count": 0,
                "dialogue_ratio": 0.0,
                "avg_dialogue_length": 0,
                "total_speech_words": 0,
                "speech_to_narrative_ratio": 0.0
            }

        total_speech_chars = sum(
            sum(part["length"] for part in d["speech_parts"])
            for d in dialogues
        )

        total_speech_words = sum(
            len(part["text"].split())
            for d in dialogues
            for part in d["speech_parts"]
        )

        total_text_words = len(text.split())
        speech_ratio = total_speech_words / max(total_text_words, 1)

        avg_dialogue_length = sum(
            len(part["text"].split())
            for d in dialogues
            for part in d["speech_parts"]
        ) / max(sum(len(d["speech_parts"]) for d in dialogues), 1)

        return {
            "dialogue_count": len(dialogues),
            "dialogue_ratio": round(speech_ratio, 3),
            "avg_dialogue_length": round(avg_dialogue_length, 1),
            "total_speech_words": total_speech_words,
            "total_speech_chars": total_speech_chars,
            "speech_to_narrative_ratio": round(speech_ratio / (1 - speech_ratio) if speech_ratio < 1 else float('inf'), 3)
        }

    def _analyze_dialogue_structure(self, dialogues: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Анализ структуры диалогов"""
        if not dialogues:
            return {"structure_score": 0}

        # Анализ распределения диалогов по тексту
        dialogue_positions = [d["paragraph_index"] for d in dialogues]
        total_paragraphs = len(text.split('\n\n'))

        if total_paragraphs > 1:
            # Равномерность распределения
            distribution_score = 1 - (max(dialogue_positions) - min(dialogue_positions)) / total_paragraphs
        else:
            distribution_score = 1.0

        # Анализ чередования диалога и повествования
        dialogue_paragraphs = set(d["paragraph_index"] for d in dialogues)
        alternation_score = 0

        consecutive_dialogues = 0
        max_consecutive = 0

        for i in range(max(dialogue_positions) + 1):
            if i in dialogue_paragraphs:
                consecutive_dialogues += 1
                max_consecutive = max(max_consecutive, consecutive_dialogues)
            else:
                consecutive_dialogues = 0

        # Чем больше подряд идущих диалогов, тем хуже структура
        alternation_score = max(0, 1 - max_consecutive / 10)

        # Наличие слов автора
        author_words_ratio = sum(len(d["author_words"]) for d in dialogues) / max(len(dialogues), 1)
        author_integration_score = min(author_words_ratio / 2, 1.0)  # Оптимум: 2 слова автора на диалог

        structure_score = (distribution_score + alternation_score + author_integration_score) / 3

        return {
            "structure_score": round(structure_score, 3),
            "distribution_score": round(distribution_score, 3),
            "alternation_score": round(alternation_score, 3),
            "author_integration_score": round(author_integration_score, 3),
            "max_consecutive_dialogues": max_consecutive
        }

    def _analyze_speech_characteristics(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ характеристик речи"""
        if not dialogues:
            return {"speech_diversity": 0, "character_voices": 0}

        all_speech_texts = [
            part["text"] for d in dialogues for part in d["speech_parts"]
        ]

        if not all_speech_texts:
            return {"speech_diversity": 0, "character_voices": 0}

        # Анализ длины высказываний
        speech_lengths = [len(speech.split()) for speech in all_speech_texts]
        avg_speech_length = sum(speech_lengths) / len(speech_lengths)
        speech_length_variance = sum((length - avg_speech_length) ** 2 for length in speech_lengths) / len(speech_lengths)

        # Лексическое разнообразие в речи
        all_speech_words = []
        for speech in all_speech_texts:
            words = re.findall(r'\b\w+\b', speech.lower())
            all_speech_words.extend(words)

        if all_speech_words:
            speech_diversity = len(set(all_speech_words)) / len(all_speech_words)
        else:
            speech_diversity = 0

        # Анализ речевых особенностей
        feature_scores = {}
        for feature, markers in self.speech_features.items():
            feature_count = sum(
                speech.lower().count(marker) for speech in all_speech_texts for marker in markers
            )
            feature_scores[f"{feature}_score"] = feature_count / max(len(all_speech_texts), 1)

        # Оценка разнообразия персонажей (эвристическая)
        # Основана на разнообразии речевых паттернов
        character_diversity = self._estimate_character_diversity(all_speech_texts)

        return {
            "speech_diversity": round(speech_diversity, 3),
            "avg_speech_length": round(avg_speech_length, 2),
            "speech_length_variance": round(speech_length_variance, 2),
            "estimated_character_count": character_diversity,
            **{key: round(value, 3) for key, value in feature_scores.items()}
        }

    def _estimate_character_diversity(self, speech_texts: List[str]) -> int:
        """Эвристическая оценка количества разных персонажей"""
        if not speech_texts:
            return 0

        # Группируем речь по стилистическим особенностям
        character_signatures = []

        for speech in speech_texts:
            signature = {
                'length': len(speech.split()),
                'formality': sum(speech.lower().count(marker) for marker in self.speech_features['формальность']),
                'colloquial': sum(speech.lower().count(marker) for marker in self.speech_features['разговорность']),
                'question_marks': speech.count('?'),
                'exclamations': speech.count('!'),
                'personal_pronouns': len(re.findall(r'\b(я|ты|мы|вы|он|она|они)\b', speech.lower()))
            }
            character_signatures.append(signature)

        # Простая кластеризация по похожести подписей
        clusters = []
        for signature in character_signatures:
            assigned = False
            for cluster in clusters:
                if self._signatures_similar(signature, cluster[0], threshold=0.7):
                    cluster.append(signature)
                    assigned = True
                    break

            if not assigned:
                clusters.append([signature])

        return len(clusters)

    def _signatures_similar(self, sig1: Dict[str, int], sig2: Dict[str, int], threshold: float = 0.7) -> bool:
        """Проверка похожести речевых подписей"""
        similarity_score = 0
        total_features = 0

        for key in sig1:
            if key in sig2:
                val1, val2 = sig1[key], sig2[key]
                if val1 == val2 == 0:
                    similarity_score += 1
                elif max(val1, val2) > 0:
                    similarity_score += 1 - abs(val1 - val2) / max(val1, val2, 1)
                total_features += 1

        return (similarity_score / max(total_features, 1)) >= threshold

    def _analyze_dialogue_emotions(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ эмоциональности диалогов"""
        if not dialogues:
            return {"emotional_diversity": 0}

        all_speech_texts = [
            part["text"] for d in dialogues for part in d["speech_parts"]
        ]

        emotion_counts = {}
        total_emotional_markers = 0

        for emotion, markers in self.emotion_markers.items():
            count = sum(
                speech.lower().count(marker) for speech in all_speech_texts for marker in markers
            )
            emotion_counts[emotion] = count
            total_emotional_markers += count

        # Эмоциональное разнообразие
        emotions_present = sum(1 for count in emotion_counts.values() if count > 0)
        emotional_diversity = emotions_present / len(self.emotion_markers)

        # Эмоциональная интенсивность
        emotional_intensity = total_emotional_markers / max(len(all_speech_texts), 1)

        # Распределение эмоций
        emotion_distribution = {}
        if total_emotional_markers > 0:
            for emotion, count in emotion_counts.items():
                emotion_distribution[f"{emotion}_ratio"] = round(count / total_emotional_markers, 3)

        return {
            "emotional_diversity": round(emotional_diversity, 3),
            "emotional_intensity": round(emotional_intensity, 3),
            "total_emotional_markers": total_emotional_markers,
            **emotion_distribution
        }

    def _analyze_dialogue_style(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ стилистики диалогов"""
        if not dialogues:
            return {"style_consistency": 0}

        all_speech_texts = [
            part["text"] for d in dialogues for part in d["speech_parts"]
        ]

        # Анализ пунктуации
        punctuation_analysis = self._analyze_dialogue_punctuation(all_speech_texts)

        # Анализ синтаксиса
        syntax_analysis = self._analyze_dialogue_syntax(all_speech_texts)

        # Анализ лексики
        lexical_analysis = self._analyze_dialogue_lexicon(all_speech_texts)

        return {
            **punctuation_analysis,
            **syntax_analysis,
            **lexical_analysis
        }

    def _analyze_dialogue_punctuation(self, speech_texts: List[str]) -> Dict[str, float]:
        """Анализ пунктуации в диалогах"""
        if not speech_texts:
            return {}

        total_questions = sum(speech.count('?') for speech in speech_texts)
        total_exclamations = sum(speech.count('!') for speech in speech_texts)
        total_ellipsis = sum(speech.count('...') + speech.count('…') for speech in speech_texts)

        return {
            "question_density": round(total_questions / len(speech_texts), 3),
            "exclamation_density": round(total_exclamations / len(speech_texts), 3),
            "ellipsis_density": round(total_ellipsis / len(speech_texts), 3)
        }

    def _analyze_dialogue_syntax(self, speech_texts: List[str]) -> Dict[str, float]:
        """Анализ синтаксиса диалогов"""
        if not speech_texts:
            return {}

        # Простые предложения (без запятых)
        simple_sentences = sum(1 for speech in speech_texts if ',' not in speech)
        simple_ratio = simple_sentences / len(speech_texts)

        # Неполные предложения (эллиптические)
        incomplete_pattern = r'^[а-я].*[^.!?]$'
        incomplete_sentences = sum(1 for speech in speech_texts
                                 if re.match(incomplete_pattern, speech.strip(), re.IGNORECASE))
        incomplete_ratio = incomplete_sentences / len(speech_texts)

        return {
            "simple_sentences_ratio": round(simple_ratio, 3),
            "incomplete_sentences_ratio": round(incomplete_ratio, 3)
        }

    def _analyze_dialogue_lexicon(self, speech_texts: List[str]) -> Dict[str, float]:
        """Анализ лексики диалогов"""
        if not speech_texts:
            return {}

        all_words = []
        for speech in speech_texts:
            words = re.findall(r'\b\w+\b', speech.lower())
            all_words.extend(words)

        if not all_words:
            return {}

        # Разговорные слова
        colloquial_markers = ['ну', 'вот', 'блин', 'типа', 'короче', 'слушай', 'смотри']
        colloquial_count = sum(all_words.count(marker) for marker in colloquial_markers)
        colloquial_ratio = colloquial_count / len(all_words)

        # Междометия
        interjections = ['ах', 'ох', 'эх', 'ух', 'фу', 'ба', 'ого', 'ага', 'угу']
        interjection_count = sum(all_words.count(interj) for interj in interjections)
        interjection_ratio = interjection_count / len(all_words)

        return {
            "colloquial_ratio": round(colloquial_ratio, 3),
            "interjection_ratio": round(interjection_ratio, 3)
        }

    def _analyze_dialogue_authenticity(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ аутентичности диалогов"""
        if not dialogues:
            return {"authenticity_score": 0}

        all_speech_texts = [
            part["text"] for d in dialogues for part in d["speech_parts"]
        ]

        authenticity_factors = []

        # Фактор 1: Естественная длина высказываний (не слишком длинные)
        speech_lengths = [len(speech.split()) for speech in all_speech_texts]
        avg_length = sum(speech_lengths) / len(speech_lengths)
        length_score = max(0, 1 - (max(0, avg_length - 20) / 30))  # Оптимум до 20 слов
        authenticity_factors.append(length_score)

        # Фактор 2: Наличие разговорных элементов
        colloquial_score = min(1, sum(
            speech.lower().count(marker) for speech in all_speech_texts
            for marker in ['ну', 'вот', 'да', 'нет', 'ага']
        ) / len(all_speech_texts))
        authenticity_factors.append(colloquial_score)

        # Фактор 3: Эмоциональная выразительность
        emotional_punctuation = sum(
            speech.count('!') + speech.count('?') + speech.count('...')
            for speech in all_speech_texts
        )
        emotion_score = min(1, emotional_punctuation / len(all_speech_texts) / 2)
        authenticity_factors.append(emotion_score)

        # Фактор 4: Неформальность структуры
        incomplete_sentences = sum(
            1 for speech in all_speech_texts
            if not speech.strip().endswith(('.', '!', '?'))
        )
        informality_score = min(1, incomplete_sentences / len(all_speech_texts) * 2)
        authenticity_factors.append(informality_score)

        authenticity_score = sum(authenticity_factors) / len(authenticity_factors)

        return {
            "authenticity_score": round(authenticity_score, 3),
            "length_authenticity": round(length_score, 3),
            "colloquial_authenticity": round(colloquial_score, 3),
            "emotional_authenticity": round(emotion_score, 3),
            "structural_authenticity": round(informality_score, 3)
        }

    def _calculate_overall_dialogue_score(self, metrics: Dict[str, Any]) -> float:
        """Вычисление общей оценки качества диалогов"""
        if metrics.get("dialogue_count", 0) == 0:
            return 0.0

        # Веса для различных компонентов
        components = [
            ("structure_score", 0.25),
            ("speech_diversity", 0.2),
            ("emotional_diversity", 0.2),
            ("authenticity_score", 0.25),
            ("dialogue_ratio", 0.1)  # Небольшой вес за наличие диалогов
        ]

        score = 0
        total_weight = 0

        for metric, weight in components:
            if metric in metrics:
                value = metrics[metric]
                if metric == "dialogue_ratio":
                    # Оптимальный ratio диалогов 0.2-0.4 (20-40% текста)
                    optimal_ratio = 0.3
                    ratio_score = 1 - abs(value - optimal_ratio) / optimal_ratio
                    ratio_score = max(0, ratio_score)
                    score += ratio_score * weight
                else:
                    score += value * weight
                total_weight += weight

        return round(score / total_weight if total_weight > 0 else 0, 3)

    def _get_dialogue_grade(self, score: float) -> str:
        """Определение оценки качества диалогов"""
        if score >= 0.8:
            return "отличный"
        elif score >= 0.6:
            return "хороший"
        elif score >= 0.4:
            return "удовлетворительный"
        elif score >= 0.2:
            return "слабый"
        else:
            return "неудовлетворительный"

    def _generate_dialogue_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению диалогов"""
        recommendations = []

        dialogue_ratio = metrics.get("dialogue_ratio", 0)
        if dialogue_ratio < 0.1:
            recommendations.append("Добавьте больше диалогов для оживления повествования")
        elif dialogue_ratio > 0.6:
            recommendations.append("Сбалансируйте диалоги с повествованием")

        authenticity_score = metrics.get("authenticity_score", 0)
        if authenticity_score < 0.5:
            recommendations.append("Сделайте диалоги более естественными: добавьте разговорные элементы")

        emotional_diversity = metrics.get("emotional_diversity", 0)
        if emotional_diversity < 0.3:
            recommendations.append("Добавьте эмоциональное разнообразие в диалоги")

        structure_score = metrics.get("structure_score", 0)
        if structure_score < 0.5:
            recommendations.append("Улучшите структуру диалогов: добавьте слова автора и разнообразьте подачу")

        if not recommendations:
            recommendations.append("Диалоги имеют хорошее качество")

        return recommendations

    async def _get_llm_dialogue_analysis(self, sample_dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ диалогов с помощью LLM"""
        try:
            # Формируем текст для анализа
            sample_text = "\n".join([
                f"Диалог {i+1}: " + " ".join([part["text"] for part in d["speech_parts"]])
                for i, d in enumerate(sample_dialogues)
            ])

            prompt = self.prompt_template.format(text=sample_text)
            response = await self.llm_service.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Ошибка LLM анализа диалогов: {e}")
            return {}

    # Методы для анализа по чанкам

    async def _analyze_chunks(self, text: BaseText, **kwargs) -> AnalysisResult:
        """Анализ диалогов по чанкам"""
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
        aggregated_metrics = self._aggregate_chunk_dialogues(chunk_results)

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
        """Анализ диалогов в одном чанке"""
        dialogues = self._extract_dialogues(chunk_text)
        basic_stats = self._calculate_basic_dialogue_stats(chunk_text, dialogues)

        if dialogues:
            speech_analysis = self._analyze_speech_characteristics(dialogues)
            emotion_analysis = self._analyze_dialogue_emotions(dialogues)
            overall_score = self._calculate_overall_dialogue_score({
                **basic_stats, **speech_analysis, **emotion_analysis
            })
        else:
            speech_analysis = {}
            emotion_analysis = {}
            overall_score = 0

        return {
            "chunk_index": chunk_index,
            "overall_dialogue_score": overall_score,
            "dialogue_grade": self._get_dialogue_grade(overall_score),
            **basic_stats,
            **speech_analysis,
            **emotion_analysis
        }

    def _aggregate_chunk_dialogues(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирование результатов анализа диалогов по чанкам"""
        if not chunk_results:
            return {}

        # Суммарная статистика
        total_dialogues = sum(chunk.get("dialogue_count", 0) for chunk in chunk_results)
        chunks_with_dialogues = sum(1 for chunk in chunk_results if chunk.get("dialogue_count", 0) > 0)

        # Средние значения метрик
        numeric_metrics = [
            "overall_dialogue_score", "dialogue_ratio", "speech_diversity", "emotional_diversity"
        ]

        aggregated = {
            "total_dialogues": total_dialogues,
            "chunks_with_dialogues": chunks_with_dialogues,
            "dialogue_distribution": round(chunks_with_dialogues / len(chunk_results), 3)
        }

        for metric in numeric_metrics:
            values = [chunk.get(metric, 0) for chunk in chunk_results if chunk.get(metric, 0) > 0]
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 3)
                aggregated[f"max_{metric}"] = round(max(values), 3)

        # Общая оценка диалогов
        avg_score = aggregated.get("avg_overall_dialogue_score", 0)
        aggregated["overall_dialogue_grade"] = self._get_dialogue_grade(avg_score)

        return aggregated

    def estimate_time(self, text: BaseText, mode: AnalysisMode = AnalysisMode.FULL_TEXT) -> float:
        """Оценка времени анализа диалогов"""
        base_time = 0.8  # Базовое время для извлечения и анализа диалогов

        # Время зависит от количества потенциальных диалогов
        potential_dialogues = text.content.count('—') + text.content.count('"') + text.content.count('«')
        dialogue_factor = potential_dialogues / 100  # 100 диалогов = 1 секунда

        if mode == AnalysisMode.CHUNKED:
            chunking_strategy = ChunkingStrategy()
            estimated_chunks = max(1, len(text.content) // chunking_strategy.chunk_size)
            return (base_time + dialogue_factor) * estimated_chunks * 0.3
        else:
            llm_time = 2.0 if self.llm_service and potential_dialogues > 0 else 0
            return base_time + dialogue_factor + llm_time

    def get_supported_modes(self) -> list[AnalysisMode]:
        """Получить поддерживаемые режимы анализа"""
        return [AnalysisMode.FULL_TEXT, AnalysisMode.CHUNKED]

    # Реализация абстрактных методов BaseAnalyzer

    @property
    def name(self) -> str:
        return "dialogue"

    @property
    def display_name(self) -> str:
        return "Анализ диалогов"

    @property
    def description(self) -> str:
        return "Анализирует диалоги и прямую речь в тексте"

    @property
    def requires_llm(self) -> bool:
        return False

    def interpret_results(self, result: AnalysisResult) -> str:
        data = result.result_data
        if "error" in data:
            return f"Ошибка анализа диалогов: {data['error']}"

        dialogue_count = data.get("dialogue_count", 0)
        dialogue_ratio = data.get("dialogue_ratio", 0)
        overall_quality = data.get("overall_dialogue_quality", 0)
        dialogue_grade = data.get("dialogue_grade", "неопределенная")

        interpretation = f"Диалоги в тексте: {dialogue_count} фрагментов\n"
        interpretation += f"Доля диалогов: {dialogue_ratio:.1%}\n"
        interpretation += f"Качество диалогов: {dialogue_grade} ({overall_quality:.3f})\n"

        authenticity = data.get("authenticity_score", 0)
        if authenticity > 0:
            interpretation += f"Аутентичность: {authenticity:.3f}\n"

        return interpretation.strip()