"""
Сервис для оркестрации анализа текстов
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
from datetime import datetime

from src.analysis_domain.entities import AnalysisSession, AnalysisResult, BaseAnalyzer
from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.plain_text import PlainText
from src.text_domain.services.chunking_service import ChunkingService
from src.infrastructure.database.repositories import SessionRepository, TextRepository
from src.infrastructure.queue.progress_broadcaster import ProgressBroadcaster
from src.model_management.services.llm_service import LLMService
from src.model_management.services.embedding_service import EmbeddingService
from src.common.exceptions import AnalysisError
from src.common.utils import now_utc
from sqlalchemy.ext.asyncio import AsyncSession


class AnalysisService:
    """Сервис для выполнения анализа текстов"""

    def __init__(
        self,
        db_session: AsyncSession,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        progress_broadcaster: Optional[ProgressBroadcaster] = None
    ):
        """
        Args:
            db_session: Сессия базы данных
            llm_service: Сервис для работы с LLM
            embedding_service: Сервис для embeddings
            progress_broadcaster: Broadcaster для WebSocket обновлений
        """
        self.db_session = db_session
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.progress_broadcaster = progress_broadcaster

        self.session_repo = SessionRepository(db_session)
        self.text_repo = TextRepository(db_session)

        # Реестр анализаторов
        self._analyzer_registry: Dict[str, type] = {}
        self._register_analyzers()

    def _register_analyzers(self):
        """Регистрация всех доступных анализаторов"""
        from src.analysis_domain.analyzers.genre_analyzer import GenreAnalyzer
        from src.analysis_domain.analyzers.style_analyzer import StyleAnalyzer
        from src.analysis_domain.analyzers.emotion_analyzer import EmotionAnalyzer
        from src.analysis_domain.analyzers.complexity_analyzer import ComplexityAnalyzer
        from src.analysis_domain.analyzers.readability_analyzer import ReadabilityAnalyzer
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer
        from src.analysis_domain.analyzers.dialogue_analyzer import DialogueAnalyzer
        from src.analysis_domain.analyzers.description_analyzer import DescriptionAnalyzer
        from src.analysis_domain.analyzers.structure_analyzer import StructureAnalyzer

        self._analyzer_registry = {
            "GenreAnalyzer": GenreAnalyzer,
            "StyleAnalyzer": StyleAnalyzer,
            "EmotionAnalyzer": EmotionAnalyzer,
            "ComplexityAnalyzer": ComplexityAnalyzer,
            "ReadabilityAnalyzer": ReadabilityAnalyzer,
            "CharacterAnalyzer": CharacterAnalyzer,
            "TensionAnalyzer": TensionAnalyzer,
            "PaceAnalyzer": PaceAnalyzer,
            "WaterAnalyzer": WaterAnalyzer,
            "ThemeAnalyzer": ThemeAnalyzer,
            "DialogueAnalyzer": DialogueAnalyzer,
            "DescriptionAnalyzer": DescriptionAnalyzer,
            "StructureAnalyzer": StructureAnalyzer,
        }

    def get_available_analyzers(self) -> List[str]:
        """Получить список доступных анализаторов"""
        return list(self._analyzer_registry.keys())

    async def run_session(self, session_id: str) -> None:
        """
        Запустить сессию анализа

        Args:
            session_id: ID сессии
        """
        try:
            logger.info(f"🚀 Начало выполнения сессии {session_id}")

            # Обновляем статус
            await self.session_repo.update_status(
                session_id, "running", progress=0, progress_message="Инициализация..."
            )
            await self._broadcast_progress(session_id, 0, "Инициализация...")

            # Получаем сессию
            session_model = await self.session_repo.get_by_id(session_id, load_relations=True)
            if not session_model:
                raise AnalysisError(f"Session {session_id} not found")

            # Получаем тексты и анализаторы
            text_ids = await self.session_repo.get_text_ids(session_id)
            analyzer_names = await self.session_repo.get_analyzer_names(session_id)

            if not text_ids:
                raise AnalysisError("No texts in session")
            if not analyzer_names:
                raise AnalysisError("No analyzers in session")

            logger.info(f"Тексты: {len(text_ids)}, анализаторы: {len(analyzer_names)}")

            # Общее количество задач
            total_tasks = len(text_ids) * len(analyzer_names)
            completed_tasks = 0

            # Анализируем каждый текст
            for text_idx, text_id in enumerate(text_ids):
                logger.info(f"📖 Обработка текста {text_idx + 1}/{len(text_ids)}: {text_id}")

                # Получаем текст
                content = await self.text_repo.get_content(text_id)
                if not content:
                    logger.warning(f"Текст {text_id} не найден, пропускаем")
                    completed_tasks += len(analyzer_names)
                    continue

                text = PlainText(
                    id=text_id,
                    title=f"Text {text_id}",
                    content=content,
                    storage_type="database"
                )

                # Чанкуем если нужно
                chunks = None
                if session_model.mode == "chunked":
                    await self._broadcast_progress(
                        session_id,
                        int((completed_tasks / total_tasks) * 100),
                        f"Чанкинг текста {text_idx + 1}/{len(text_ids)}..."
                    )
                    chunks = await self._chunk_text(text)

                # Запускаем анализаторы
                for analyzer_idx, analyzer_name in enumerate(analyzer_names):
                    try:
                        progress_msg = f"Анализ {analyzer_name} для текста {text_idx + 1}/{len(text_ids)}"
                        logger.info(f"🔍 {progress_msg}")

                        await self._broadcast_progress(
                            session_id,
                            int((completed_tasks / total_tasks) * 100),
                            progress_msg
                        )

                        # Создаем и запускаем анализатор
                        start_time = now_utc()
                        result = await self._run_analyzer(
                            analyzer_name, text, chunks, session_model.mode
                        )
                        execution_time = (now_utc() - start_time).total_seconds() * 1000

                        # Сохраняем результат
                        await self.session_repo.save_result(
                            session_id=session_id,
                            text_id=text_id,
                            analyzer_name=analyzer_name,
                            result_data=result.data,
                            interpretation=self._get_interpretation(analyzer_name, result),
                            execution_time_ms=execution_time
                        )

                        completed_tasks += 1

                    except Exception as e:
                        logger.error(f"Ошибка в анализаторе {analyzer_name}: {e}")
                        completed_tasks += 1
                        # Продолжаем с другими анализаторами

            # Завершено успешно
            await self.session_repo.update_status(
                session_id, "completed", progress=100, progress_message="Анализ завершён"
            )
            await self._broadcast_progress(session_id, 100, "Анализ завершён ✅")

            logger.info(f"✅ Сессия {session_id} завершена успешно")

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения сессии {session_id}: {e}")
            await self.session_repo.update_status(
                session_id, "failed", error=str(e)
            )
            await self._broadcast_progress(session_id, 0, f"Ошибка: {str(e)}")
            raise

    async def _chunk_text(self, text: BaseText) -> List[Any]:
        """Разбить текст на чанки"""
        from src.text_domain.entities import ChunkingStrategy

        strategy = ChunkingStrategy(
            base_chunk_size=2000,
            min_chunk_size=500,
            max_chunk_size=4000,
            overlap_percentage=0.1,
            use_sentence_boundaries=True,
            use_paragraph_boundaries=True
        )

        chunker = ChunkingService(strategy)
        return chunker.chunk_text(text.content)

    async def _run_analyzer(
        self,
        analyzer_name: str,
        text: BaseText,
        chunks: Optional[List[Any]],
        mode: str
    ) -> dict:
        """
        Запустить анализатор

        Args:
            analyzer_name: Название анализатора
            text: Текст
            chunks: Чанки (если режим chunked)
            mode: Режим анализа

        Returns:
            Результаты анализа
        """
        analyzer_class = self._analyzer_registry.get(analyzer_name)
        if not analyzer_class:
            raise AnalysisError(f"Unknown analyzer: {analyzer_name}")

        # Создаем экземпляр анализатора
        # TODO: Передать правильные зависимости (промпт-шаблоны из БД)
        analyzer = analyzer_class(
            llm_service=self.llm_service,
            prompt_template="default"  # Заглушка
        )

        # Запускаем анализ
        result = await analyzer.analyze(text, mode, chunks=chunks)

        return result

    def _get_interpretation(self, analyzer_name: str, result: AnalysisResult) -> str:
        """Получить интерпретацию результата"""
        analyzer_class = self._analyzer_registry.get(analyzer_name)
        if not analyzer_class:
            return str(result)

        try:
            # Создаем временный экземпляр для интерпретации
            analyzer = analyzer_class(
                llm_service=self.llm_service,
                prompt_template="default"
            )
            return analyzer.interpret_results(result)
        except Exception as e:
            logger.error(f"Ошибка интерпретации результата: {e}")
            return str(result)

    async def _broadcast_progress(
        self,
        session_id: str,
        progress: int,
        message: str
    ) -> None:
        """Отправить обновление прогресса через WebSocket"""
        if self.progress_broadcaster:
            try:
                await self.progress_broadcaster.broadcast_progress(
                    task_id=session_id,
                    status="running",
                    progress=progress,
                    current_step=message
                )
            except Exception as e:
                logger.error(f"Ошибка broadcast: {e}")

    async def cancel_session(self, session_id: str) -> bool:
        """
        Отменить выполнение сессии

        Args:
            session_id: ID сессии

        Returns:
            True если отменена
        """
        try:
            await self.session_repo.update_status(
                session_id, "cancelled", progress_message="Отменено пользователем"
            )
            await self._broadcast_progress(session_id, 0, "Отменено ❌")
            return True
        except Exception as e:
            logger.error(f"Ошибка отмены сессии: {e}")
            return False
