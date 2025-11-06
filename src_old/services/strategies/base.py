"""
Base strategy for analysis operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class AnalysisStrategy(ABC):
    """Abstract strategy for analysis."""

    @abstractmethod
    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters

        Returns:
            Dictionary with analysis results
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Get analysis type identifier."""
        pass
