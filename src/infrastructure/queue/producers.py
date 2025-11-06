"""
Producers для отправки задач в очередь
"""
from typing import Optional, List
from loguru import logger

from .broker import broker
from .schemas import (
    TextAnalysisMessage,
    SessionExecutionMessage,
    IndexBuildMessage,
    ModelDownloadMessage,
    ExportMessage,
)


class TaskProducer:
    """Producer для отправки задач в очередь"""

    def __init__(self):
        """Инициализация producer"""
        self.broker = broker

    async def submit_text_analysis(
        self,
        task_id: str,
        text_id: str,
        text_title: str,
        text_content: str,
        analyzer_name: str,
        mode: str = "full_text",
        chunk_size: int = 2000,
        chunking_strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Отправить задачу анализа текста

        Args:
            task_id: ID задачи
            text_id: ID текста
            text_title: Название текста
            text_content: Содержимое текста
            analyzer_name: Название анализатора
            mode: Режим анализа
            chunk_size: Размер чанка
            chunking_strategy_id: ID стратегии чанковки

        Returns:
            bool: True если успешно
        """
        try:
            message = TextAnalysisMessage(
                task_id=task_id,
                text_id=text_id,
                text_title=text_title,
                text_content=text_content,
                analyzer_name=analyzer_name,
                mode=mode,
                chunk_size=chunk_size,
                chunking_strategy_id=chunking_strategy_id,
            )

            await self.broker.publish(
                message=message.model_dump(),
                channel="text_analysis_queue",
            )

            logger.info(f"📤 Задача анализа отправлена: {task_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки задачи анализа: {e}")
            return False

    async def submit_session_execution(
        self,
        task_id: str,
        session_id: str,
        session_name: str,
        text_ids: List[str],
        analyzer_names: List[str],
        comparator_name: Optional[str] = None,
        mode: str = "full_text",
        chunking_strategy_id: Optional[str] = None,
        chunked_comparison_strategy: str = "aggregate_first",
        use_faiss_search: bool = False,
        faiss_index_id: Optional[str] = None,
        similarity_top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> bool:
        """
        Отправить задачу выполнения сессии анализа

        Args:
            task_id: ID задачи
            session_id: ID сессии
            session_name: Название сессии
            text_ids: Список ID текстов
            analyzer_names: Список анализаторов
            comparator_name: Название компаратора
            mode: Режим анализа
            chunking_strategy_id: ID стратегии чанковки
            chunked_comparison_strategy: Стратегия сравнения в chunked режиме
            use_faiss_search: Использовать FAISS поиск
            faiss_index_id: ID FAISS индекса
            similarity_top_k: Количество похожих
            similarity_threshold: Порог сходства

        Returns:
            bool: True если успешно
        """
        try:
            message = SessionExecutionMessage(
                task_id=task_id,
                session_id=session_id,
                session_name=session_name,
                text_ids=text_ids,
                analyzer_names=analyzer_names,
                comparator_name=comparator_name,
                mode=mode,
                chunking_strategy_id=chunking_strategy_id,
                chunked_comparison_strategy=chunked_comparison_strategy,
                use_faiss_search=use_faiss_search,
                faiss_index_id=faiss_index_id,
                similarity_top_k=similarity_top_k,
                similarity_threshold=similarity_threshold,
            )

            await self.broker.publish(
                message=message.model_dump(),
                channel="session_execution_queue",
            )

            logger.info(f"📤 Задача выполнения сессии отправлена: {session_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки задачи сессии: {e}")
            return False

    async def submit_index_build(
        self,
        task_id: str,
        index_id: str,
        index_name: str,
        model_name: str,
        index_type: str,
        text_ids: List[str],
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        hnsw_m: Optional[int] = None,
        pq_m: Optional[int] = None,
        pq_nbits: Optional[int] = None,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ) -> bool:
        """
        Отправить задачу построения индекса

        Args:
            task_id: ID задачи
            index_id: ID индекса
            index_name: Название индекса
            model_name: Название модели
            index_type: Тип индекса
            text_ids: Список ID текстов
            nlist: Параметр для IVF
            nprobe: Параметр для IVF
            hnsw_m: Параметр для HNSW
            pq_m: Параметр для PQ
            pq_nbits: Параметр для PQ
            use_gpu: Использовать GPU
            gpu_id: ID GPU

        Returns:
            bool: True если успешно
        """
        try:
            message = IndexBuildMessage(
                task_id=task_id,
                index_id=index_id,
                index_name=index_name,
                model_name=model_name,
                index_type=index_type,
                text_ids=text_ids,
                nlist=nlist,
                nprobe=nprobe,
                hnsw_m=hnsw_m,
                pq_m=pq_m,
                pq_nbits=pq_nbits,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
            )

            await self.broker.publish(
                message=message.model_dump(),
                channel="index_build_queue",
            )

            logger.info(f"📤 Задача построения индекса отправлена: {index_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки задачи индекса: {e}")
            return False

    async def submit_model_download(
        self,
        task_id: str,
        model_id: str,
        model_name: str,
        model_type: str,
    ) -> bool:
        """
        Отправить задачу скачивания модели

        Args:
            task_id: ID задачи
            model_id: ID модели
            model_name: Название модели
            model_type: Тип модели

        Returns:
            bool: True если успешно
        """
        try:
            message = ModelDownloadMessage(
                task_id=task_id,
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
            )

            await self.broker.publish(
                message=message.model_dump(),
                channel="model_download_queue",
            )

            logger.info(f"📤 Задача скачивания модели отправлена: {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки задачи скачивания модели: {e}")
            return False

    async def submit_export(
        self,
        task_id: str,
        session_id: str,
        export_format: str,
        include_graphs: bool = True,
    ) -> bool:
        """
        Отправить задачу экспорта результатов

        Args:
            task_id: ID задачи
            session_id: ID сессии
            export_format: Формат экспорта (json, csv, pdf)
            include_graphs: Включить графики (для PDF)

        Returns:
            bool: True если успешно
        """
        try:
            message = ExportMessage(
                task_id=task_id,
                session_id=session_id,
                export_format=export_format,
                include_graphs=include_graphs,
            )

            await self.broker.publish(
                message=message.model_dump(),
                channel="export_queue",
            )

            logger.info(f"📤 Задача экспорта отправлена: {session_id} -> {export_format}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки задачи экспорта: {e}")
            return False


# Singleton instance
task_producer = TaskProducer()
