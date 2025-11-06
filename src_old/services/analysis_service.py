"""
Service for text analysis operations.
Implements Strategy pattern for different analysis types.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import logging
import uuid

from server.services.base import BaseService
from server.services.embedding_service import EmbeddingService
from server.services.text_service import TextService
from server.repositories.task_repository import TaskRepository, AnalysisHistoryRepository
from server.services.strategies import (
    AnalysisStrategy,
    SemanticAnalysisStrategy,
    StyleAnalysisStrategy,
    TFIDFAnalysisStrategy,
    EmotionAnalysisStrategy,
    ChunkedAnalysisStrategy,
    CombinedAnalysisStrategy,
    LLMAnalysisStrategy
)

logger = logging.getLogger(__name__)


class AnalysisService(BaseService):
    """
    Main analysis service that coordinates different strategies.
    Implements SOLID principles with strategy pattern.
    """

    def __init__(self, db: Session):
        """Initialize analysis service."""
        super().__init__(db)
        self.text_service = TextService(db)
        self.embedding_service = EmbeddingService(db)
        self.task_repo = TaskRepository(db)
        self.history_repo = AnalysisHistoryRepository(db)

        # Initialize basic strategies
        self.strategies: Dict[str, AnalysisStrategy] = {
            "semantic": SemanticAnalysisStrategy(self.embedding_service),
            "style": StyleAnalysisStrategy(),
            "tfidf": TFIDFAnalysisStrategy(),
            "emotion": EmotionAnalysisStrategy(),
            "chunked": ChunkedAnalysisStrategy(self.embedding_service),
            "llm": LLMAnalysisStrategy()
        }

        # Initialize combined strategy with access to all strategies
        self.strategies["combined"] = CombinedAnalysisStrategy(self.strategies)

    async def initialize(self):
        """Initialize service."""
        await self.embedding_service.initialize()
        await self.text_service.initialize()
        self.log_info("Analysis service initialized")

    async def cleanup(self):
        """Cleanup service resources."""
        await self.embedding_service.cleanup()
        await self.text_service.cleanup()
        self.log_info("Analysis service cleaned up")

    async def analyze(
        self,
        analysis_type: str,
        text1_id: str,
        text2_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform analysis using appropriate strategy.

        Args:
            analysis_type: Type of analysis
            text1_id: First text ID
            text2_id: Second text ID
            params: Additional parameters

        Returns:
            Analysis results
        """
        if analysis_type not in self.strategies:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Get text contents
        text1_content = await self.text_service.get_text_content(text1_id)
        text2_content = await self.text_service.get_text_content(text2_id)

        if not text1_content or not text2_content:
            raise ValueError("One or both texts not found")

        # Add text IDs to params
        if params is None:
            params = {}
        params["text1_id"] = text1_id
        params["text2_id"] = text2_id

        # Execute strategy
        strategy = self.strategies[analysis_type]
        result = await strategy.analyze(text1_content, text2_content, params)

        # Save to history
        text1 = self.text_service.get_text(text1_id)
        text2 = self.text_service.get_text(text2_id)

        self.history_repo.create(
            type=analysis_type,
            text1_title=text1.title if text1 else "Unknown",
            text2_title=text2.title if text2 else "Unknown",
            similarity=result.get("similarity"),
            interpretation=result.get("interpretation"),
            details=result
        )

        return result

    def analyze_sync(
        self,
        analysis_type: str,
        text1_id: str,
        text2_id: str,
        task_id: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Execute analysis synchronously (for background tasks).

        Args:
            analysis_type: Type of analysis
            text1_id: First text ID
            text2_id: Second text ID
            task_id: Task ID
            params: Additional parameters
        """
        try:
            # Get text contents using sync methods
            text1_content = self.text_service.get_text_content_sync(text1_id)
            text2_content = self.text_service.get_text_content_sync(text2_id)

            if not text1_content or not text2_content:
                raise ValueError("One or both texts not found")

            # Add text IDs to params
            if params is None:
                params = {}
            params["text1_id"] = text1_id
            params["text2_id"] = text2_id

            # Execute strategy synchronously
            strategy = self.strategies[analysis_type]

            # Use asyncio to run async analyze in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(strategy.analyze(text1_content, text2_content, params))
            finally:
                loop.close()

            # Save to history
            text1 = self.text_service.get_text(text1_id)
            text2 = self.text_service.get_text(text2_id)

            self.history_repo.create(
                type=analysis_type,
                text1_title=text1.title if text1 else "Unknown",
                text2_title=text2.title if text2 else "Unknown",
                similarity=result.get("similarity"),
                interpretation=result.get("interpretation"),
                details=result
            )

            # Mark as completed
            self.task_repo.complete_task(task_id, result)
            self.log_info(f"Task {task_id} completed successfully")

        except Exception as e:
            # Mark task as failed
            self.task_repo.fail_task(task_id, str(e))
            self.log_error(f"Task {task_id} failed: {e}")

    async def analyze_async(
        self,
        analysis_type: str,
        text1_id: str,
        text2_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start async analysis task.

        Args:
            analysis_type: Type of analysis
            text1_id: First text ID
            text2_id: Second text ID
            params: Additional parameters

        Returns:
            Task ID
        """
        # Create task and ENSURE it's saved
        task_id = str(uuid.uuid4())

        try:
            # Create task in DB
            task = self.task_repo.create_task(
                task_id=task_id,
                name=f"{analysis_type.title()} Analysis",
                task_type=analysis_type,
                params=params or {},
                text1_id=text1_id,
                text2_id=text2_id
            )

            if not task:
                raise ValueError("Failed to create task in database")

            # Start task AFTER confirming it's created
            self.task_repo.start_task(task_id)

            # Verify task exists before returning
            created_task = self.task_repo.get_task(task_id)
            if not created_task:
                raise ValueError(f"Task {task_id} was not found after creation")

            # IMPORTANT: Just return task_id, actual execution happens via BackgroundTasks
            return task_id

        except Exception as e:
            # Mark task as failed if it exists
            try:
                if self.task_repo.get_task(task_id):
                    self.task_repo.fail_task(task_id, str(e))
                    self.db.commit()
            except:
                pass
            raise

    async def get_analysis_history(
        self,
        limit: int = 10,
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get analysis history.

        Args:
            limit: Maximum results
            analysis_type: Filter by type

        Returns:
            List of analysis results
        """
        analyses = self.history_repo.get_recent_analyses(limit, analysis_type)
        return [
            {
                "id": a.id,
                "type": a.type,
                "text1_title": a.text1_title,
                "text2_title": a.text2_title,
                "similarity": a.similarity,
                "interpretation": a.interpretation,
                "created_at": a.created_at.isoformat()
            }
            for a in analyses
        ]