"""
Analysis strategies - each implements a specific type of book analysis.
Following Strategy pattern and Single Responsibility Principle.
"""

from server.services.book_analysis.strategies.genre_strategy import GenreAnalysisStrategy
from server.services.book_analysis.strategies.character_strategy import CharacterAnalysisStrategy
from server.services.book_analysis.strategies.tension_strategy import TensionAnalysisStrategy
from server.services.book_analysis.strategies.pace_strategy import PaceAnalysisStrategy
from server.services.book_analysis.strategies.water_strategy import WaterAnalysisStrategy
from server.services.book_analysis.strategies.theme_strategy import ThemeAnalysisStrategy

__all__ = [
    'GenreAnalysisStrategy',
    'CharacterAnalysisStrategy',
    'TensionAnalysisStrategy',
    'PaceAnalysisStrategy',
    'WaterAnalysisStrategy',
    'ThemeAnalysisStrategy',
]
