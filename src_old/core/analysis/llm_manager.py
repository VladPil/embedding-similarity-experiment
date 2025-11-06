"""
LLM Manager for text analysis operations.
"""

from typing import Dict, Any, Optional
from loguru import logger

from server.core.analysis.llm import LLMAnalyzer


class LLMManager:
    """Manager class for LLM operations."""

    def __init__(self, model_key: str = 'qwen2.5-1.5b', device: Optional[str] = None):
        """
        Initialize LLM Manager.

        Args:
            model_key: Model identifier
            device: Device to use ('cuda', 'cpu', or None for settings)
        """
        self.model_key = model_key
        self.analyzer = LLMAnalyzer(model_key=model_key, device=device)
        logger.info(f"LLMManager initialized with model {model_key}")

    async def execute_task(
        self,
        task_type: str,
        text1: str,
        text2: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute LLM task based on task type.

        Args:
            task_type: Type of task to execute
            text1: First text
            text2: Second text (optional)
            **kwargs: Additional parameters

        Returns:
            Dictionary with task results
        """
        try:
            # Handle different task types
            if task_type == "compare":
                if not text2:
                    raise ValueError("Second text is required for comparison")
                result = self.analyzer.compare_texts(text1, text2)

            elif task_type == "summary":
                max_words = kwargs.get("max_words", 100)
                result = self.analyzer.quick_summary(text1, max_words=max_words)

            elif task_type == "themes":
                result = self.analyzer.extract_key_themes(text1)

            elif task_type == "sentiment":
                result = self.analyzer.quick_sentiment(text1)

            elif task_type == "differences":
                if not text2:
                    raise ValueError("Second text is required for differences")
                result = self.analyzer.extract_key_differences(text1, text2)

            elif task_type == "quick_compare":
                if not text2:
                    raise ValueError("Second text is required for comparison")
                result = self.analyzer.compare_quick(text1, text2)

            elif task_type == "report":
                if not text2:
                    raise ValueError("Second text is required for report")
                text1_title = kwargs.get("text1_title", "Text 1")
                text2_title = kwargs.get("text2_title", "Text 2")
                result = self.analyzer.generate_report(
                    text1, text2,
                    text1_title=text1_title,
                    text2_title=text2_title
                )

            elif task_type == "single_report":
                text_title = kwargs.get("text1_title", "Text")
                result = self.analyzer.generate_single_text_report(
                    text1,
                    text_title=text_title
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Add task type to result
            result["task_type"] = task_type

            return result

        except Exception as e:
            logger.error(f"LLM task execution failed: {e}")
            return {
                "error": str(e),
                "task_type": task_type,
                "success": False
            }

    def load_model(self):
        """Load the LLM model."""
        self.analyzer.load_model()

    def unload_model(self):
        """Unload the LLM model to free memory."""
        self.analyzer.unload_model()

    def check_gpu_memory(self) -> Dict[str, Any]:
        """Check GPU memory usage."""
        return self.analyzer.check_gpu_memory()

    @staticmethod
    def list_models():
        """List available models."""
        return LLMAnalyzer.list_models()


# Global instance
_llm_manager: Optional[LLMManager] = None


async def get_llm_manager(model_key: str = 'qwen2.5-1.5b', device: Optional[str] = None) -> LLMManager:
    """
    Get or create global LLM manager instance.

    Args:
        model_key: Model identifier
        device: Device to use

    Returns:
        LLMManager instance
    """
    global _llm_manager

    if _llm_manager is None:
        _llm_manager = LLMManager(model_key=model_key, device=device)

    return _llm_manager