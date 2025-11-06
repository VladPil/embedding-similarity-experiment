"""
Book Analysis Service Module.

Modular monolith architecture with clear service layers.
"""

from server.services.book_analysis.strategies import *
from server.services.book_analysis.builder import AnalysisBuilder
from server.services.book_analysis.factory import StrategyFactory
from server.services.book_analysis.service import BookAnalysisService

__all__ = [
    'AnalysisBuilder',
    'StrategyFactory',
    'BookAnalysisService',
]
