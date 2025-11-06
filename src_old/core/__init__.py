"""
Core business logic modules.
"""

# Legacy imports - kept for backward compatibility
from server.core.similarity_calc import SimilarityCalculator
# from server.core.embeddings.manager import EmbeddingManager
# from server.core.chunks.manager import ChunkManager
# from server.core.analysis.style import StyleAnalyzer
# from server.core.analysis.tfidf import TFIDFAnalyzer
# from server.core.analysis.emotion import EmotionAnalyzer

# New architecture exports
from server.core.analysis import base as analysis_base
from server.core.similarity import base as similarity_base

__all__ = [
    "SimilarityCalculator",  # Legacy - kept for backward compatibility
    # "EmbeddingManager",
    # "ChunkManager",
    # "StyleAnalyzer",
    # "TFIDFAnalyzer",
    # "EmotionAnalyzer",
    "analysis_base",
    "similarity_base",
]