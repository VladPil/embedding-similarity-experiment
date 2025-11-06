"""
Анализатор описательных элементов текста с использованием NLP
"""
import re
import json
from typing import Optional, Dict, Any, List, Set
from collections import Counter, defaultdict
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class DescriptionAnalyzer(BaseAnalyzer):
    """
    Анализатор описательных элементов текста

    Анализирует описания в тексте:
    - Выявление описательных пассажей
    - Анализ типов описаний (портреты, пейзажи, интерьеры, действия)
    - Оценка детализации и выразительности
    - Анализ изобразительных средств
    - Баланс описания и повествования
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        super().__init__(
            name="DescriptionAnalyzer",
            description="Анализирует описательные элементы и изобразительные средства в тексте"
        )
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._get_default_prompt()

        # Обогащенные маркеры описательных элементов
        self.description_markers = {
            'внешность': [
                # Лицо и черты
                'лицо', 'глаза', 'волосы', 'рост', 'фигура', 'черты', 'взгляд', 'улыбка',
                'брови', 'губы', 'нос', 'щеки', 'подбородок', 'лоб', 'руки', 'пальцы',
                'ресницы', 'веки', 'скулы', 'морщины', 'родинка', 'шрам', 'ямочка',
                'борода', 'усы', 'прическа', 'локоны', 'кудри', 'косы', 'челка',
                # Телосложение
                'плечи', 'грудь', 'талия', 'бедра', 'ноги', 'походка', 'осанка', 'жесты',
                'движения', 'мимика', 'выражение', 'кожа', 'загар', 'бледность', 'румянец'
            ],
            'одежда': [
                # Основная одежда
                'платье', 'костюм', 'рубашка', 'блузка', 'свитер', 'кофта', 'джемпер',
                'брюки', 'джинсы', 'юбка', 'шорты', 'пальто', 'куртка', 'пиджак', 'жилет',
                'футболка', 'майка', 'топ', 'халат', 'сарафан', 'кимоно', 'мундир',
                # Обувь и аксессуары
                'обувь', 'туфли', 'ботинки', 'сапоги', 'кроссовки', 'сандалии', 'тапочки',
                'шляпа', 'кепка', 'берет', 'шапка', 'платок', 'шарф', 'галстук', 'бант',
                'перчатки', 'варежки', 'пояс', 'ремень', 'сумка', 'портфель', 'рюкзак',
                # Украшения
                'украшения', 'кольцо', 'браслет', 'цепочка', 'ожерелье', 'серьги', 'брошь',
                'часы', 'очки', 'линзы', 'значок', 'булавка', 'запонки', 'медальон'
            ],
            'природа': [
                # Небо и атмосфера
                'небо', 'облака', 'тучи', 'солнце', 'луна', 'звезды', 'рассвет', 'закат',
                'ветер', 'бриз', 'ураган', 'буря', 'дождь', 'ливень', 'морось', 'снег',
                'метель', 'град', 'туман', 'роса', 'иней', 'радуга', 'молния', 'гром',
                # Растительность
                'лес', 'роща', 'чаща', 'поляна', 'деревья', 'кусты', 'трава', 'мох',
                'цветы', 'бутоны', 'лепестки', 'листья', 'ветки', 'корни', 'ствол',
                'крона', 'хвоя', 'шишки', 'ягоды', 'грибы', 'папоротник', 'плющ',
                # Водоемы и рельеф
                'река', 'ручей', 'озеро', 'пруд', 'море', 'океан', 'залив', 'бухта',
                'волны', 'прибой', 'течение', 'водопад', 'родник', 'болото', 'топь',
                'горы', 'холмы', 'скалы', 'утес', 'пещера', 'долина', 'овраг', 'поле',
                'луг', 'степь', 'пустыня', 'оазис', 'тропа', 'дорога', 'тропинка'
            ],
            'интерьер': [
                # Архитектурные элементы
                'комната', 'зал', 'спальня', 'гостиная', 'кухня', 'кабинет', 'библиотека',
                'стены', 'стена', 'окно', 'окна', 'дверь', 'двери', 'потолок', 'пол',
                'порог', 'подоконник', 'карниз', 'плинтус', 'балка', 'колонна', 'арка',
                # Мебель
                'мебель', 'стол', 'столик', 'стул', 'кресло', 'диван', 'кровать', 'кушетка',
                'шкаф', 'комод', 'тумбочка', 'полка', 'стеллаж', 'буфет', 'сервант',
                'трюмо', 'секретер', 'этажерка', 'табурет', 'скамья', 'банкетка',
                # Декор и освещение
                'картина', 'портрет', 'зеркало', 'рама', 'ваза', 'статуэтка', 'свеча',
                'лампа', 'люстра', 'торшер', 'бра', 'ковер', 'дорожка', 'гобелен',
                'занавески', 'шторы', 'жалюзи', 'покрывало', 'плед', 'подушки', 'скатерть'
            ],
            'архитектура': [
                # Здания и сооружения
                'дом', 'особняк', 'усадьба', 'дворец', 'замок', 'крепость', 'башня',
                'здание', 'строение', 'сооружение', 'храм', 'церковь', 'собор', 'мечеть',
                'синагога', 'монастырь', 'театр', 'музей', 'школа', 'больница', 'вокзал',
                # Элементы архитектуры
                'фасад', 'портал', 'крыша', 'кровля', 'черепица', 'купол', 'шпиль',
                'балкон', 'терраса', 'веранда', 'крыльцо', 'лестница', 'ступени', 'перила',
                'колонна', 'пилястра', 'капитель', 'фронтон', 'карниз', 'арка', 'свод',
                'ворота', 'калитка', 'забор', 'ограда', 'решетка', 'парапет', 'аттик',
                # Городская среда
                'сад', 'парк', 'сквер', 'аллея', 'бульвар', 'двор', 'дворик', 'патио',
                'площадь', 'улица', 'переулок', 'проспект', 'набережная', 'мост', 'мостик',
                'фонтан', 'беседка', 'павильон', 'ротонда', 'обелиск', 'памятник', 'стела'
            ],
            'атмосфера': [
                # Эмоциональная окраска
                'уют', 'комфорт', 'тепло', 'холод', 'мрак', 'свет', 'яркость', 'блеск',
                'тень', 'полумрак', 'сумрак', 'темнота', 'освещение', 'подсветка',
                'торжественность', 'величие', 'роскошь', 'богатство', 'бедность', 'скромность',
                'запустение', 'ветхость', 'новизна', 'старина', 'античность', 'современность'
            ]
        }

        # Сенсорные модальности
        self.sensory_words = {
            'зрение': {
                'цвет': ['красный', 'синий', 'зеленый', 'желтый', 'черный', 'белый', 'серый',
                        'яркий', 'тусклый', 'светлый', 'темный', 'блестящий', 'матовый'],
                'форма': ['круглый', 'квадратный', 'длинный', 'короткий', 'широкий', 'узкий',
                         'высокий', 'низкий', 'прямой', 'изогнутый', 'острый', 'тупой'],
                'размер': ['большой', 'маленький', 'огромный', 'крошечный', 'средний',
                          'гигантский', 'миниатюрный', 'массивный', 'компактный']
            },
            'слух': ['тихий', 'громкий', 'шепот', 'крик', 'мелодия', 'шум', 'звук', 'голос',
                    'эхо', 'тишина', 'гул', 'треск', 'шорох', 'скрип', 'звон'],
            'осязание': ['мягкий', 'твердый', 'гладкий', 'шершавый', 'холодный', 'теплый',
                        'горячий', 'влажный', 'сухой', 'скользкий', 'липкий', 'колючий'],
            'обоняние': ['запах', 'аромат', 'вонь', 'благоухание', 'пахнуть', 'душистый',
                        'ароматный', 'вонючий', 'свежий', 'затхлый', 'острый'],
            'вкус': ['сладкий', 'кислый', 'горький', 'соленый', 'острый', 'пресный',
                    'вкусный', 'невкусный', 'приятный', 'отвратительный']
        }

        # Художественные приемы в описаниях
        self.literary_devices = {
            'сравнение': [r'как\s+\w+', r'словно\s+\w+', r'будто\s+\w+', r'подобно\s+\w+'],
            'метафора': [r'\w+\s+—\s+\w+', r'\w+\s+есть\s+\w+'],
            'эпитеты': ['прекрасный', 'великолепный', 'чудесный', 'удивительный', 'восхитительный',
                       'таинственный', 'волшебный', 'загадочный', 'мрачный', 'печальный'],
            'олицетворение': [r'(солнце|луна|ветер|дождь|море|река)\s+(говорит|смеется|плачет|думает)']
        }

        # Временные маркеры для описаний
        self.temporal_markers = {
            'утро': ['рассвет', 'утром', 'восход', 'раннее утро', 'утренний'],
            'день': ['днем', 'полдень', 'дневной', 'солнечный день'],
            'вечер': ['вечером', 'закат', 'сумерки', 'вечерний', 'заходящее солнце'],
            'ночь': ['ночью', 'полночь', 'ночной', 'темнота', 'звездная ночь']
        }

    def _get_default_prompt(self) -> PromptTemplate:
        """Получить промпт по умолчанию для LLM анализа"""
        template = """
        Проанализируй описательные элементы в данном тексте:
        1. Качество и детализация описаний
        2. Образность и выразительность
        3. Использование художественных приемов
        4. Сенсорное восприятие в описаниях

        Текст: {text}

        Верни результат в формате JSON:
        {{
            "description_quality": float (0-1),
            "imagery_strength": float (0-1),
            "sensory_richness": float (0-1),
            "artistic_techniques": ["список использованных приемов"],
            "description_types": ["типы описаний в тексте"],
            "recommendations": ["рекомендации по улучшению описаний"]
        }}
        """
        return PromptTemplate(
            template=template,
            variables=["text"],
            name="description_analysis_prompt"
        )

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        **kwargs
    ) -> AnalysisResult:
        """
        Анализ описательных элементов в тексте

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа описаний
        """
        try:
            logger.info(f"Начинаем анализ описательных элементов в тексте: {text.title}")

            if mode == AnalysisMode.CHUNKED:
                return await self._analyze_chunks(text, **kwargs)
            else:
                return await self._analyze_full_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка анализа описаний: {e}")
            raise AnalysisError(f"Не удалось проанализировать описательные элементы: {e}")

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

        # Выявление описательных пассажей
        descriptive_passages = self._extract_descriptive_passages(content)

        # Базовая статистика описаний
        basic_stats = self._calculate_basic_description_stats(content, descriptive_passages)

        # Анализ типов описаний
        type_analysis = self._analyze_description_types(descriptive_passages)

        # Анализ сенсорности
        sensory_analysis = self._analyze_sensory_elements(descriptive_passages)

        # Анализ художественных приемов
        literary_analysis = self._analyze_literary_devices(descriptive_passages)

        # Анализ детализации
        detail_analysis = self._analyze_description_detail(descriptive_passages)

        # Анализ динамики описаний
        dynamic_analysis = self._analyze_description_dynamics(content, descriptive_passages)

        # Объединение результатов
        all_metrics = {
            **basic_stats,
            **type_analysis,
            **sensory_analysis,
            **literary_analysis,
            **detail_analysis,
            **dynamic_analysis
        }

        # Общая оценка качества описаний
        overall_score = self._calculate_overall_description_score(all_metrics)
        all_metrics["overall_description_quality"] = overall_score
        all_metrics["description_grade"] = self._get_description_grade(overall_score)

        # Рекомендации
        all_metrics["automated_recommendations"] = self._generate_description_recommendations(all_metrics)

        # LLM анализ если доступен
        if self.llm_service and descriptive_passages:
            llm_analysis = await self._get_llm_description_analysis(descriptive_passages[:3])
            all_metrics.update(llm_analysis)

        return AnalysisResult(
            text_id=text.id,
            analyzer_name=self.name,
            data=all_metrics,
            metadata={
                "mode": "full_text",
                "text_length": len(content),
                "descriptive_passages_found": len(descriptive_passages)
            }
        )

    def _extract_descriptive_passages(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение описательных пассажей из текста"""
        sentences = self._split_sentences(text)
        descriptive_passages = []

        for i, sentence in enumerate(sentences):
            description_score = self._score_sentence_descriptiveness(sentence)

            if description_score > 0.3:  # Порог описательности
                passage_info = {
                    "sentence_index": i,
                    "text": sentence.strip(),
                    "description_score": description_score,
                    "length": len(sentence.split()),
                    "types": self._classify_description_type(sentence)
                }
                descriptive_passages.append(passage_info)

        return descriptive_passages

    def _score_sentence_descriptiveness(self, sentence: str) -> float:
        """Оценка описательности предложения"""
        words = sentence.lower().split()
        if not words:
            return 0

        descriptive_score = 0

        # Проверяем наличие описательных маркеров
        for category, markers in self.description_markers.items():
            marker_count = sum(1 for word in words if word in markers)
            descriptive_score += marker_count * 0.1

        # Проверяем сенсорные слова
        for sense, sense_words in self.sensory_words.items():
            if isinstance(sense_words, dict):
                for sub_category, sub_words in sense_words.items():
                    marker_count = sum(1 for word in words if word in sub_words)
                    descriptive_score += marker_count * 0.15
            else:
                marker_count = sum(1 for word in words if word in sense_words)
                descriptive_score += marker_count * 0.15

        # Проверяем прилагательные (эвристически)
        adjective_patterns = [r'\w+(ный|ная|ное|ые)$', r'\w+(ий|ая|ое|ие)$']
        adjective_count = sum(
            len(re.findall(pattern, sentence, re.IGNORECASE))
            for pattern in adjective_patterns
        )
        descriptive_score += (adjective_count / len(words)) * 0.5

        # Нормализуем оценку
        return min(descriptive_score, 1.0)

    def _classify_description_type(self, sentence: str) -> List[str]:
        """Классификация типа описания"""
        sentence_lower = sentence.lower()
        types = []

        for desc_type, markers in self.description_markers.items():
            if any(marker in sentence_lower for marker in markers):
                types.append(desc_type)

        return types

    def _calculate_basic_description_stats(self, text: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вычисление базовой статистики описаний"""
        if not passages:
            return {
                "description_count": 0,
                "description_ratio": 0.0,
                "avg_description_length": 0,
                "total_description_words": 0,
                "description_density": 0.0
            }

        total_description_words = sum(p["length"] for p in passages)
        total_text_words = len(text.split())

        description_ratio = total_description_words / max(total_text_words, 1)
        avg_length = total_description_words / len(passages)

        # Плотность описаний (описания на 1000 слов)
        description_density = (len(passages) / max(total_text_words, 1)) * 1000

        return {
            "description_count": len(passages),
            "description_ratio": round(description_ratio, 3),
            "avg_description_length": round(avg_length, 1),
            "total_description_words": total_description_words,
            "description_density": round(description_density, 2)
        }

    def _analyze_description_types(self, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ типов описаний"""
        if not passages:
            return {"type_diversity": 0}

        # Подсчет типов описаний
        type_counts = defaultdict(int)
        for passage in passages:
            for desc_type in passage["types"]:
                type_counts[desc_type] += 1

        # Разнообразие типов
        types_present = len(type_counts)
        total_types_possible = len(self.description_markers)
        type_diversity = types_present / total_types_possible

        # Распределение типов
        total_typed = sum(type_counts.values())
        type_distribution = {}
        if total_typed > 0:
            for desc_type, count in type_counts.items():
                type_distribution[f"{desc_type}_ratio"] = round(count / total_typed, 3)

        return {
            "type_diversity": round(type_diversity, 3),
            "types_count": types_present,
            **type_distribution
        }

    def _analyze_sensory_elements(self, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ сенсорных элементов в описаниях"""
        if not passages:
            return {"sensory_richness": 0}

        all_text = " ".join([p["text"] for p in passages]).lower()

        sensory_scores = {}
        total_sensory_words = 0

        for sense, words in self.sensory_words.items():
            if isinstance(words, dict):
                # Для зрения, которое разбито на подкатегории
                sense_count = 0
                for sub_category, sub_words in words.items():
                    count = sum(all_text.count(word) for word in sub_words)
                    sense_count += count
                    sensory_scores[f"{sense}_{sub_category}_count"] = count
                sensory_scores[f"{sense}_total"] = sense_count
            else:
                # Для остальных чувств
                count = sum(all_text.count(word) for word in words)
                sensory_scores[f"{sense}_count"] = count

            total_sensory_words += sensory_scores.get(f"{sense}_total", sensory_scores.get(f"{sense}_count", 0))

        # Сенсорное разнообразие
        senses_used = sum(1 for sense in self.sensory_words.keys()
                         if sensory_scores.get(f"{sense}_total", sensory_scores.get(f"{sense}_count", 0)) > 0)
        sensory_diversity = senses_used / len(self.sensory_words)

        # Сенсорная плотность
        description_words = sum(p["length"] for p in passages)
        sensory_density = total_sensory_words / max(description_words, 1)

        return {
            "sensory_diversity": round(sensory_diversity, 3),
            "sensory_density": round(sensory_density, 3),
            "total_sensory_words": total_sensory_words,
            **sensory_scores
        }

    def _analyze_literary_devices(self, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ художественных приемов в описаниях"""
        if not passages:
            return {"literary_richness": 0}

        all_text = " ".join([p["text"] for p in passages])

        device_counts = {}
        total_devices = 0

        for device, patterns in self.literary_devices.items():
            count = 0
            for pattern in patterns:
                if isinstance(pattern, str) and not pattern.startswith('r'):
                    # Простой поиск по словам
                    count += all_text.lower().count(pattern)
                else:
                    # Регулярные выражения
                    regex_pattern = pattern[2:] if pattern.startswith('r') else pattern
                    matches = re.findall(regex_pattern, all_text.lower())
                    count += len(matches)

            device_counts[f"{device}_count"] = count
            total_devices += count

        # Художественное разнообразие
        devices_used = sum(1 for count in device_counts.values() if count > 0)
        literary_diversity = devices_used / len(self.literary_devices)

        # Плотность художественных приемов
        description_words = sum(p["length"] for p in passages)
        literary_density = total_devices / max(description_words, 1) * 100  # на 100 слов

        return {
            "literary_diversity": round(literary_diversity, 3),
            "literary_density": round(literary_density, 2),
            "total_literary_devices": total_devices,
            **device_counts
        }

    def _analyze_description_detail(self, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ детализации описаний"""
        if not passages:
            return {"detail_level": 0}

        # Анализ длины описаний как показатель детализации
        lengths = [p["length"] for p in passages]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)

        # Вариативность детализации
        length_variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        length_std = length_variance ** 0.5

        # Уровень детализации (основан на длине и описательности)
        description_scores = [p["description_score"] for p in passages]
        avg_descriptiveness = sum(description_scores) / len(description_scores)

        # Комбинированный показатель детализации
        detail_level = min((avg_length / 20) * avg_descriptiveness, 1.0)  # 20 слов = хорошая детализация

        # Консистентность детализации
        detail_consistency = max(0, 1 - (length_std / max(avg_length, 1)))

        return {
            "detail_level": round(detail_level, 3),
            "detail_consistency": round(detail_consistency, 3),
            "avg_description_length": round(avg_length, 1),
            "max_description_length": max_length,
            "detail_variance": round(length_variance, 2)
        }

    def _analyze_description_dynamics(self, text: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ динамики и распределения описаний в тексте"""
        if not passages:
            return {"distribution_quality": 0}

        sentences = self._split_sentences(text)
        total_sentences = len(sentences)

        # Позиции описаний в тексте
        positions = [p["sentence_index"] for p in passages]

        if total_sentences <= 1:
            return {"distribution_quality": 1.0}

        # Равномерность распределения
        # Разбиваем текст на сегменты и проверяем наличие описаний в каждом
        segments = 5  # Разбиваем на 5 частей
        segment_size = total_sentences // segments
        segment_coverage = 0

        for i in range(segments):
            segment_start = i * segment_size
            segment_end = (i + 1) * segment_size if i < segments - 1 else total_sentences

            # Проверяем есть ли описания в этом сегменте
            descriptions_in_segment = sum(1 for pos in positions
                                        if segment_start <= pos < segment_end)
            if descriptions_in_segment > 0:
                segment_coverage += 1

        distribution_quality = segment_coverage / segments

        # Анализ концентрации описаний
        # Ищем области с высокой концентрацией описаний
        concentration_areas = 0
        window_size = max(total_sentences // 10, 1)  # Окно 10% от текста

        for i in range(0, total_sentences - window_size + 1, window_size // 2):
            window_descriptions = sum(1 for pos in positions
                                    if i <= pos < i + window_size)
            # Если в окне больше 3 описаний - это область концентрации
            if window_descriptions >= 3:
                concentration_areas += 1

        # Временные маркеры в описаниях
        temporal_analysis = self._analyze_temporal_markers(passages)

        return {
            "distribution_quality": round(distribution_quality, 3),
            "segment_coverage": segment_coverage,
            "concentration_areas": concentration_areas,
            **temporal_analysis
        }

    def _analyze_temporal_markers(self, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ временных маркеров в описаниях"""
        if not passages:
            return {}

        all_text = " ".join([p["text"] for p in passages]).lower()

        temporal_counts = {}
        for time_period, markers in self.temporal_markers.items():
            count = sum(all_text.count(marker) for marker in markers)
            temporal_counts[f"{time_period}_descriptions"] = count

        # Временное разнообразие
        periods_used = sum(1 for count in temporal_counts.values() if count > 0)
        temporal_diversity = periods_used / len(self.temporal_markers)

        return {
            "temporal_diversity": round(temporal_diversity, 3),
            **temporal_counts
        }

    def _calculate_overall_description_score(self, metrics: Dict[str, Any]) -> float:
        """Вычисление общей оценки качества описаний"""
        if metrics.get("description_count", 0) == 0:
            return 0.0

        # Веса компонентов
        components = [
            ("description_ratio", 0.15),        # Наличие описаний
            ("type_diversity", 0.2),           # Разнообразие типов
            ("sensory_diversity", 0.2),        # Сенсорное богатство
            ("literary_diversity", 0.15),      # Художественные приемы
            ("detail_level", 0.15),           # Уровень детализации
            ("distribution_quality", 0.15)     # Качество распределения
        ]

        score = 0
        total_weight = 0

        for metric, weight in components:
            if metric in metrics:
                value = metrics[metric]
                if metric == "description_ratio":
                    # Оптимальное соотношение описаний: 0.15-0.3
                    optimal_ratio = 0.225
                    ratio_score = 1 - abs(value - optimal_ratio) / optimal_ratio
                    ratio_score = max(0, min(ratio_score, 1))
                    score += ratio_score * weight
                else:
                    score += value * weight
                total_weight += weight

        return round(score / total_weight if total_weight > 0 else 0, 3)

    def _get_description_grade(self, score: float) -> str:
        """Определение оценки качества описаний"""
        if score >= 0.8:
            return "превосходный"
        elif score >= 0.65:
            return "хороший"
        elif score >= 0.5:
            return "удовлетворительный"
        elif score >= 0.3:
            return "слабый"
        else:
            return "недостаточный"

    def _generate_description_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению описаний"""
        recommendations = []

        description_ratio = metrics.get("description_ratio", 0)
        if description_ratio < 0.1:
            recommendations.append("Добавьте больше описательных элементов для создания атмосферы")
        elif description_ratio > 0.4:
            recommendations.append("Сократите описания, чтобы не замедлять повествование")

        type_diversity = metrics.get("type_diversity", 0)
        if type_diversity < 0.4:
            recommendations.append("Разнообразьте типы описаний: добавьте портреты, пейзажи, интерьеры")

        sensory_diversity = metrics.get("sensory_diversity", 0)
        if sensory_diversity < 0.4:
            recommendations.append("Используйте больше сенсорных деталей: звуки, запахи, тактильные ощущения")

        literary_diversity = metrics.get("literary_diversity", 0)
        if literary_diversity < 0.3:
            recommendations.append("Добавьте художественные приемы: сравнения, метафоры, эпитеты")

        detail_level = metrics.get("detail_level", 0)
        if detail_level < 0.4:
            recommendations.append("Увеличьте детализацию описаний для большей выразительности")

        distribution_quality = metrics.get("distribution_quality", 0)
        if distribution_quality < 0.5:
            recommendations.append("Распределите описания более равномерно по тексту")

        if not recommendations:
            recommendations.append("Описания имеют хорошее качество и баланс")

        return recommendations

    async def _get_llm_description_analysis(self, sample_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ описаний с помощью LLM"""
        try:
            # Формируем текст для анализа
            sample_text = "\n".join([
                f"Описание {i+1}: {passage['text']}"
                for i, passage in enumerate(sample_passages)
            ])

            prompt = self.prompt_template.format(text=sample_text)
            response = await self.llm_service.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Ошибка LLM анализа описаний: {e}")
            return {}

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивка текста на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    # Методы для анализа по чанкам

    async def _analyze_chunks(self, text: BaseText, **kwargs) -> AnalysisResult:
        """Анализ описаний по чанкам"""
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
        aggregated_metrics = self._aggregate_chunk_descriptions(chunk_results)

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
        """Анализ описаний в одном чанке"""
        passages = self._extract_descriptive_passages(chunk_text)
        basic_stats = self._calculate_basic_description_stats(chunk_text, passages)

        if passages:
            type_analysis = self._analyze_description_types(passages)
            sensory_analysis = self._analyze_sensory_elements(passages)
            overall_score = self._calculate_overall_description_score({
                **basic_stats, **type_analysis, **sensory_analysis
            })
        else:
            type_analysis = {}
            sensory_analysis = {}
            overall_score = 0

        return {
            "chunk_index": chunk_index,
            "overall_description_score": overall_score,
            "description_grade": self._get_description_grade(overall_score),
            **basic_stats,
            **type_analysis,
            **sensory_analysis
        }

    def _aggregate_chunk_descriptions(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирование результатов анализа описаний по чанкам"""
        if not chunk_results:
            return {}

        # Суммарная статистика
        total_descriptions = sum(chunk.get("description_count", 0) for chunk in chunk_results)
        chunks_with_descriptions = sum(1 for chunk in chunk_results
                                     if chunk.get("description_count", 0) > 0)

        # Средние значения
        numeric_metrics = [
            "overall_description_score", "description_ratio", "type_diversity", "sensory_diversity"
        ]

        aggregated = {
            "total_descriptions": total_descriptions,
            "chunks_with_descriptions": chunks_with_descriptions,
            "description_distribution": round(chunks_with_descriptions / len(chunk_results), 3)
        }

        for metric in numeric_metrics:
            values = [chunk.get(metric, 0) for chunk in chunk_results if chunk.get(metric, 0) > 0]
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 3)
                aggregated[f"max_{metric}"] = round(max(values), 3)

        # Общая оценка описаний
        avg_score = aggregated.get("avg_overall_description_score", 0)
        aggregated["overall_description_grade"] = self._get_description_grade(avg_score)

        return aggregated

    def estimate_time(self, text: BaseText, mode: AnalysisMode = AnalysisMode.FULL_TEXT) -> float:
        """Оценка времени анализа описаний"""
        base_time = 1.2  # Базовое время для анализа описательных элементов

        # Время зависит от количества предложений (потенциальных описаний)
        estimated_sentences = text.content.count('.') + text.content.count('!') + text.content.count('?')
        sentence_factor = estimated_sentences / 200  # 200 предложений = 1 секунда

        if mode == AnalysisMode.CHUNKED:
            chunking_strategy = ChunkingStrategy()
            estimated_chunks = max(1, len(text.content) // chunking_strategy.chunk_size)
            return (base_time + sentence_factor) * estimated_chunks * 0.4
        else:
            llm_time = 1.8 if self.llm_service else 0
            return base_time + sentence_factor + llm_time

    def get_supported_modes(self) -> list[AnalysisMode]:
        """Получить поддерживаемые режимы анализа"""
        return [AnalysisMode.FULL_TEXT, AnalysisMode.CHUNKED]

    # Реализация абстрактных методов BaseAnalyzer

    @property
    def name(self) -> str:
        return "description"

    @property
    def display_name(self) -> str:
        return "Анализ описаний"

    @property
    def description(self) -> str:
        return "Анализирует описательные элементы и изобразительные средства в тексте"

    @property
    def requires_llm(self) -> bool:
        return False

    def interpret_results(self, result: AnalysisResult) -> str:
        data = result.result_data
        if "error" in data:
            return f"Ошибка анализа описаний: {data['error']}"

        description_count = data.get("description_count", 0)
        description_ratio = data.get("description_ratio", 0)
        overall_quality = data.get("overall_description_quality", 0)
        description_grade = data.get("description_grade", "недостаточный")

        interpretation = f"Описания в тексте: {description_count} фрагментов\n"
        interpretation += f"Доля описаний: {description_ratio:.1%}\n"
        interpretation += f"Качество описаний: {description_grade} ({overall_quality:.3f})\n"

        sensory_diversity = data.get("sensory_diversity", 0)
        if sensory_diversity > 0:
            interpretation += f"Сенсорное разнообразие: {sensory_diversity:.3f}\n"

        return interpretation.strip()