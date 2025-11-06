"""
Общие Enum типы для всего приложения
"""
from enum import Enum


# ===== Сессия анализа =====

class SessionStatus(str, Enum):
    """Статус сессии анализа"""
    DRAFT = "draft"              # Создана, настраивается
    QUEUED = "queued"            # В очереди
    RUNNING = "running"          # Выполняется
    COMPLETED = "completed"      # Завершена
    FAILED = "failed"            # Ошибка
    CANCELLED = "cancelled"      # Отменена
    PARTIAL = "partial"          # Частично завершена


class AnalysisMode(str, Enum):
    """Режим анализа"""
    FULL_TEXT = "full_text"      # Анализ всего текста целиком
    CHUNKED = "chunked"          # Анализ по чанкам


class ChunkedComparisonStrategy(str, Enum):
    """Стратегия сравнения в chunked режиме"""
    AGGREGATE_FIRST = "aggregate_first"  # Сначала агрегируем → потом сравниваем
    CHUNK_TO_CHUNK = "chunk_to_chunk"    # Сравниваем каждый чанк с каждым
    HYBRID = "hybrid"                    # Гибридный подход


# ===== Тексты =====

class TextType(str, Enum):
    """Тип текста"""
    PLAIN = "plain"              # Обычный текст
    FB2 = "fb2"                  # FB2 книга


class TextStorageType(str, Enum):
    """Тип хранения текста"""
    DATABASE = "database"        # В БД (для коротких)
    FILE = "file"                # В файле (для длинных)


# ===== Анализаторы =====

class AnalyzerType(str, Enum):
    """Типы анализаторов"""
    GENRE = "genre"              # Жанр
    STYLE = "style"              # Стиль
    EMOTION = "emotion"          # Эмоции
    THEME = "theme"              # Темы
    CHARACTER = "character"      # Персонажи
    PACE = "pace"                # Темп повествования
    TENSION = "tension"          # Напряжение
    WATER = "water"              # Вода (стилистика)
    TFIDF = "tfidf"              # TF-IDF
    SIMILARITY = "similarity"    # Поиск похожих


# ===== Компараторы =====

class ComparatorType(str, Enum):
    """Типы компараторов"""
    COSINE = "cosine"            # Косинусное сходство
    SEMANTIC = "semantic"        # Семантическое
    EUCLIDEAN = "euclidean"      # Евклидово расстояние
    JACCARD = "jaccard"          # Жаккар
    HYBRID = "hybrid"            # Гибридное
    LLM = "llm"                  # Через LLM


# ===== Модели =====

class ModelType(str, Enum):
    """Тип модели"""
    LLM = "llm"                  # LLM модель
    EMBEDDING = "embedding"      # Embedding модель


class ModelStatus(str, Enum):
    """Статус модели"""
    NOT_DOWNLOADED = "not_downloaded"  # Не скачана
    DOWNLOADING = "downloading"        # Скачивается
    DOWNLOADED = "downloaded"          # Скачана
    LOADING = "loading"                # Загружается в память
    LOADED = "loaded"                  # Загружена и готова
    UNLOADING = "unloading"            # Выгружается
    ERROR = "error"                    # Ошибка


class QuantizationType(str, Enum):
    """Тип квантизации модели"""
    NONE = "none"                # Без квантизации
    INT8 = "int8"                # 8-bit
    INT4 = "int4"                # 4-bit
    FP16 = "fp16"                # Half precision


# ===== FAISS индексы =====

class FaissIndexType(str, Enum):
    """Тип FAISS индекса"""
    FLAT = "flat"                # IndexFlat - точный поиск
    IVF_FLAT = "ivf_flat"        # IndexIVFFlat - инвертированный индекс
    HNSW = "hnsw"                # IndexHNSW - граф
    IVF_PQ = "ivf_pq"            # IndexIVFPQ - с квантизацией


class IndexStatus(str, Enum):
    """Статус индекса"""
    BUILDING = "building"        # Строится
    READY = "ready"              # Готов
    REBUILDING = "rebuilding"    # Перестраивается
    ERROR = "error"              # Ошибка


# ===== Экспорт =====

class ExportFormat(str, Enum):
    """Формат экспорта"""
    JSON = "json"                # JSON
    CSV = "csv"                  # CSV
    PDF = "pdf"                  # PDF с графиками


# ===== Задачи =====

class TaskType(str, Enum):
    """Тип задачи"""
    TEXT_ANALYSIS = "text_analysis"           # Анализ текста
    SESSION_EXECUTION = "session_execution"   # Выполнение сессии
    INDEX_BUILD = "index_build"               # Построение индекса
    MODEL_DOWNLOAD = "model_download"         # Скачивание модели
    EXPORT = "export"                         # Экспорт результатов


class TaskStatus(str, Enum):
    """Статус задачи"""
    PENDING = "pending"          # Ожидает
    RUNNING = "running"          # Выполняется
    COMPLETED = "completed"      # Завершена
    FAILED = "failed"            # Ошибка
    CANCELLED = "cancelled"      # Отменена


# ===== Устройства =====

class DeviceType(str, Enum):
    """Тип устройства для вычислений"""
    CPU = "cpu"                  # CPU
    CUDA = "cuda"                # NVIDIA GPU
    MPS = "mps"                  # Apple Silicon
