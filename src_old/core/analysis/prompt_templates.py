"""
Optimized prompt templates for book analysis using 7B/8B LLM models.
All prompts designed for JSON output and short, focused responses.
"""

from typing import Dict


class PromptTemplates:
    """Collection of optimized prompts for book analysis."""

    # ==========================================================================
    # GENRE CLASSIFICATION
    # ==========================================================================

    GENRE_CLASSIFICATION = """Определи жанр текста. Выбери один основной и до двух дополнительных жанров.

Доступные жанры:
- фэнтези (волшебство, магия, вымышленные существа)
- sci-fi (технологии будущего, космос, роботы, ИИ)
- детектив (расследование преступления, сбор улик)
- триллер (постоянная угроза, напряжение, твисты)
- ужасы (страх, ужас, сверхъестественное)
- романтика (любовная линия как центр сюжета)
- исторический (реальный исторический период)
- приключения (путешествия, квесты, исследования)
- драма (глубокие эмоции, конфликты характеров)
- антиутопия (мрачное будущее, тоталитаризм)

ТЕКСТ:
{text}

Ответь СТРОГО в формате JSON:
{{
  "main_genre": "название_жанра",
  "sub_genres": ["жанр1", "жанр2"],
  "confidence": 0.85,
  "reasoning": "краткое обоснование 1-2 предложения"
}}"""

    # ==========================================================================
    # CHARACTER ANALYSIS
    # ==========================================================================

    CHARACTER_ANALYSIS = """Проанализируй персонажа в этом отрывке.

ТЕКСТ:
{text}

КОНТЕКСТ (если есть):
{context}

Определи:
1. Имя персонажа (если упоминается)
2. 3-5 ключевых черт характера
3. Роль: "main" (главный), "secondary" (второстепенный), "episodic" (эпизодический)

Ответь СТРОГО в формате JSON:
{{
  "name": "Имя Персонажа",
  "traits": [
    {{"trait": "храбрость", "evidence": "бросился в бой без колебаний"}},
    {{"trait": "верность", "evidence": "не предал друга даже под угрозой"}},
    {{"trait": "импульсивность", "evidence": "действует не подумав"}}
  ],
  "role": "main",
  "age_group": "adult",
  "brief_description": "краткое описание 1 предложение"
}}"""

    # ==========================================================================
    # TENSION ANALYSIS
    # ==========================================================================

    TENSION_ANALYSIS = """В этом отрывке обнаружен высокий уровень напряжения. Опиши источник КРАТКО.

ТЕКСТ:
{text}

Источники напряжения:
- physical_danger (физическая опасность, битва, угроза жизни)
- interpersonal_conflict (конфликт между персонажами)
- moral_dilemma (сложный моральный выбор)
- mystery (тайна, неизвестность, загадка)
- time_pressure (цейтнот, дедлайн, спешка)
- internal_struggle (внутренний конфликт героя)

Ответь СТРОГО в формате JSON:
{{
  "source": "physical_danger",
  "description": "краткое описание 1-2 предложения",
  "excerpt": "ключевая цитата из текста"
}}"""

    # ==========================================================================
    # EVENT EXTRACTION
    # ==========================================================================

    EVENT_EXTRACTION = """Определи является ли этот отрывок ключевым сюжетным событием.

ТЕКСТ:
{text}

Ключевое событие = что-то важное произошло: решение, действие, открытие, поворот.
НЕ ключевое = описания, размышления, обыденные действия.

Если ключевое - опиши кратко ЧТО произошло и ПОЧЕМУ это важно.

Ответь СТРОГО в формате JSON:
{{
  "is_key_event": true,
  "event_type": "conflict_start",
  "description": "краткое описание события",
  "importance": 8
}}

Типы событий: conflict_start, decision, discovery, turning_point, climax, resolution"""

    # ==========================================================================
    # THEME EXTRACTION
    # ==========================================================================

    THEME_EXTRACTION = """Определи главную тему этого отрывка.

ТЕКСТ:
{text}

Возможные темы:
- дружба, верность
- любовь, романтика
- добро vs зло
- взросление, поиск себя
- жертва, самопожертвование
- свобода, выбор
- власть, коррупция
- семья, родство
- предательство, доверие
- смерть, потеря

Ответь СТРОГО в формате JSON:
{{
  "theme": "дружба",
  "confidence": 0.9,
  "evidence": "ключевая цитата подтверждающая тему"
}}"""

    # ==========================================================================
    # RELATIONSHIP ANALYSIS
    # ==========================================================================

    RELATIONSHIP_ANALYSIS = """Определи отношения между персонажами в этом отрывке.

ТЕКСТ:
{text}

ИЗВЕСТНЫЕ ПЕРСОНАЖИ:
{known_characters}

Типы отношений:
- friendship (дружба)
- romance (романтические)
- family (семейные)
- rivalry (соперничество)
- mentor_student (наставник-ученик)
- enemies (враждебные)
- allies (союзники)

Ответь СТРОГО в формате JSON:
{{
  "relationships": [
    {{
      "character1": "Гарри",
      "character2": "Рон",
      "type": "friendship",
      "strength": 9,
      "evidence": "цитата"
    }}
  ]
}}"""

    # ==========================================================================
    # PACE ASSESSMENT
    # ==========================================================================

    PACE_ASSESSMENT = """Оцени темп повествования в этом отрывке по шкале 1-10.

ТЕКСТ:
{text}

Критерии:
- 1-3: Очень медленно (описания, размышления, детали)
- 4-6: Средне (смесь действий и описаний)
- 7-9: Быстро (события, диалоги, действия)
- 10: Очень быстро (экшен, битва, погоня)

Ответь СТРОГО в формате JSON:
{{
  "pace_score": 7,
  "reasoning": "много действий и коротких предложений"
}}"""

    # ==========================================================================
    # WORLDBUILDING ASSESSMENT
    # ==========================================================================

    WORLDBUILDING_ASSESSMENT = """Оцени проработанность мира в этом отрывке.

ТЕКСТ:
{text}

Аспекты worldbuilding:
- география, локации
- история мира, лор
- социальная структура
- магическая/технологическая система
- культура, традиции

Ответь СТРОГО в формате JSON:
{{
  "has_worldbuilding": true,
  "aspects": ["география", "магическая система"],
  "detail_level": 7,
  "excerpt": "ключевой пример worldbuilding"
}}"""

    # ==========================================================================
    # AUDIENCE DETECTION
    # ==========================================================================

    AUDIENCE_DETECTION = """Определи целевую аудиторию книги по этому отрывку.

ТЕКСТ:
{text}

Категории:
- children (0-12): простой язык, добро vs зло, без насилия
- young_adult (12-18): темы взросления, приключения, умеренное насилие
- new_adult (18-25): темы самостоятельности, отношений
- adult (25+): сложные темы, философия, графический контент допустим

Ответь СТРОГО в формате JSON:
{{
  "target_audience": "young_adult",
  "confidence": 0.85,
  "indicators": ["протагонист-подросток", "темы взросления", "умеренное насилие"]
}}"""

    # ==========================================================================
    # TRIGGER WARNING DETECTION
    # ==========================================================================

    TRIGGER_WARNING = """Проверь текст на наличие триггерного контента.

ТЕКСТ:
{text}

Триггеры:
- violence (насилие, кровь, пытки)
- sexual_content (сексуальные сцены)
- death (смерть, суицид)
- abuse (насилие над детьми, домашнее насилие)
- mental_health (депрессия, паника, психические расстройства)
- drugs (наркотики, алкоголь)

Ответь СТРОГО в формате JSON:
{{
  "triggers": [
    {{
      "type": "violence",
      "severity": 7,
      "description": "описание сцены битвы"
    }}
  ]
}}

Если триггеров нет: {{"triggers": []}}"""

    # ==========================================================================
    # STYLE ANALYSIS
    # ==========================================================================

    STYLE_ANALYSIS = """Опиши стиль письма автора в этом отрывке.

ТЕКСТ:
{text}

Аспекты стиля:
- формальность (formal/informal)
- образность (много метафор / прямолинейно)
- динамика (быстрые короткие предложения / длинные развёрнутые)
- тон (серьёзный, ироничный, драматичный, лёгкий)

Ответь СТРОГО в формате JSON:
{{
  "formality": "informal",
  "imagery_level": 7,
  "sentence_structure": "dynamic",
  "tone": "ironic",
  "description": "краткое описание стиля 1-2 предложения"
}}"""

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    @staticmethod
    def format_genre_prompt(text: str, max_chars: int = 4000) -> str:
        """Format genre classification prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.GENRE_CLASSIFICATION.format(text=text)

    @staticmethod
    def format_character_prompt(text: str, context: str = "", max_chars: int = 3000) -> str:
        """Format character analysis prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        context = context[:1000] if context and len(context) > 1000 else context
        return PromptTemplates.CHARACTER_ANALYSIS.format(text=text, context=context or "Нет")

    @staticmethod
    def format_tension_prompt(text: str, max_chars: int = 2000) -> str:
        """Format tension analysis prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.TENSION_ANALYSIS.format(text=text)

    @staticmethod
    def format_event_prompt(text: str, max_chars: int = 2000) -> str:
        """Format event extraction prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.EVENT_EXTRACTION.format(text=text)

    @staticmethod
    def format_theme_prompt(text: str, max_chars: int = 3000) -> str:
        """Format theme extraction prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.THEME_EXTRACTION.format(text=text)

    @staticmethod
    def format_relationship_prompt(text: str, known_characters: list, max_chars: int = 2500) -> str:
        """Format relationship analysis prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        chars = ", ".join(known_characters[:10])  # Limit to first 10
        return PromptTemplates.RELATIONSHIP_ANALYSIS.format(text=text, known_characters=chars)

    @staticmethod
    def format_pace_prompt(text: str, max_chars: int = 2000) -> str:
        """Format pace assessment prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.PACE_ASSESSMENT.format(text=text)

    @staticmethod
    def format_worldbuilding_prompt(text: str, max_chars: int = 3000) -> str:
        """Format worldbuilding assessment prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.WORLDBUILDING_ASSESSMENT.format(text=text)

    @staticmethod
    def format_audience_prompt(text: str, max_chars: int = 3000) -> str:
        """Format audience detection prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.AUDIENCE_DETECTION.format(text=text)

    @staticmethod
    def format_trigger_prompt(text: str, max_chars: int = 2500) -> str:
        """Format trigger warning detection prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.TRIGGER_WARNING.format(text=text)

    @staticmethod
    def format_style_prompt(text: str, max_chars: int = 2500) -> str:
        """Format style analysis prompt."""
        text = text[:max_chars] if len(text) > max_chars else text
        return PromptTemplates.STYLE_ANALYSIS.format(text=text)


# Module-level constants for backward compatibility
COMPARISON_PROMPT = """Compare these two texts and identify their similarities and differences:

Text 1:
{text1}

Text 2:
{text2}

Please provide a detailed comparison including:
- Main themes
- Writing style
- Tone and mood
- Key differences
- Overall similarity score (0-100%)"""

SUMMARY_PROMPT = """Summarize the following text concisely:

{text}

Provide a comprehensive summary that captures the main points, key themes, and essential information."""

THEMES_PROMPT = """Identify and analyze the main themes in this text:

{text}

List the primary themes and provide brief explanations for each."""

CHARACTER_PROMPT = PromptTemplates.CHARACTER_ANALYSIS
STYLE_PROMPT = PromptTemplates.STYLE_ANALYSIS
EMOTION_PROMPT = """Analyze the emotional content of this text:

{text}

Identify the primary emotions and their intensity."""
GENRE_PROMPT = PromptTemplates.GENRE_CLASSIFICATION
WATER_ANALYSIS_PROMPT = """Analyze this text for 'water content' (unnecessary filler):

{text}

Identify any excessive or unnecessary content."""
PACE_ANALYSIS_PROMPT = PromptTemplates.PACE_ASSESSMENT
TENSION_ANALYSIS_PROMPT = PromptTemplates.TENSION_ANALYSIS
BOOK_REPORT_PROMPT = """Create a book report comparing these texts:

Text 1:
{text1}

Text 2:
{text2}

Include analysis of themes, style, and overall quality."""
MULTILINGUAL_PROMPT = """Perform a multilingual analysis:

{text}

Identify language patterns and cross-linguistic features."""
ERROR_CORRECTION_PROMPT = """Analyze this text for errors and corrections needed:

{text}

Identify any grammatical, spelling, or stylistic errors."""

FACT_CHECK_PROMPT = """Fact-check the following text:

{text}

Identify any claims that may need verification and assess their accuracy."""

SENTIMENT_PROMPT = """Analyze the sentiment of this text:

{text}

Determine if the overall sentiment is positive, negative, or neutral, and explain why."""


# Helper functions
def create_comparison_prompt(text1: str, text2: str) -> str:
    """Create a comparison prompt for two texts."""
    return COMPARISON_PROMPT.format(text1=text1, text2=text2)


def create_summary_prompt(text: str) -> str:
    """Create a summary prompt for a text."""
    return SUMMARY_PROMPT.format(text=text)


def create_analysis_prompt(text: str, analysis_type: str = "general") -> str:
    """Create an analysis prompt based on type."""
    if analysis_type == "theme":
        return THEMES_PROMPT.format(text=text)
    elif analysis_type == "character":
        return CHARACTER_PROMPT.format(text=text, context="")
    elif analysis_type == "style":
        return STYLE_PROMPT.format(text=text)
    else:
        return f"Analyze this text:\n\n{text}"


def format_prompt_with_context(template: str, **kwargs) -> str:
    """Format a prompt template with context."""
    return template.format(**kwargs)


def validate_prompt_length(prompt: str, min_length: int = 50, max_length: int = 10000) -> bool:
    """Validate that prompt length is within acceptable bounds."""
    return min_length <= len(prompt) <= max_length


def create_custom_prompt(template: str, **kwargs) -> str:
    """Create a custom prompt from a template."""
    return template.format(**kwargs)
