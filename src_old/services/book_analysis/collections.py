"""
Text collections for scalable comparison.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TextItem:
    """Single text item with metadata."""

    text_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis results cache
    analysis_results: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None  # {model_name: embedding_vector}


@dataclass
class ComparisonStrategy:
    """Configuration for comparison strategy."""

    # Analysis types to run
    analyses: List[str] = field(default_factory=lambda: [
        "genre", "character", "pace", "tension", "water", "theme"
    ])

    # Embedding comparison settings
    embedding_method: str = "hybrid"  # cosine | semantic | hybrid
    embedding_model: str = "multilingual-e5-small"

    # Chunk settings
    chunk_size: int = 2000

    # Comparison matrix settings
    compare_all_pairs: bool = True  # If False, compare only sequential pairs
    include_self_comparison: bool = False


@dataclass
class TextCollection:
    """Collection of texts for batch comparison."""

    collection_id: str
    name: str
    texts: List[TextItem]
    strategy: ComparisonStrategy

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Comparison results
    comparison_matrix: Optional[Dict[str, Dict[str, Any]]] = None

    def __len__(self) -> int:
        """Get number of texts in collection."""
        return len(self.texts)

    def get_text(self, text_id: str) -> Optional[TextItem]:
        """Get text by ID."""
        for text in self.texts:
            if text.text_id == text_id:
                return text
        return None

    def add_text(self, text: TextItem):
        """Add text to collection."""
        # Check if already exists
        existing = self.get_text(text.text_id)
        if existing:
            raise ValueError(f"Text with id {text.text_id} already exists in collection")

        self.texts.append(text)

    def remove_text(self, text_id: str) -> bool:
        """Remove text from collection."""
        for i, text in enumerate(self.texts):
            if text.text_id == text_id:
                self.texts.pop(i)
                return True
        return False

    def get_comparison_pairs(self) -> List[tuple[str, str]]:
        """
        Get list of text pairs to compare based on strategy.

        Returns:
            List of (text_id1, text_id2) tuples
        """
        pairs = []

        if self.strategy.compare_all_pairs:
            # All pairs comparison (matrix)
            for i, text1 in enumerate(self.texts):
                start_j = i if self.strategy.include_self_comparison else i + 1
                for j in range(start_j, len(self.texts)):
                    text2 = self.texts[j]
                    pairs.append((text1.text_id, text2.text_id))
        else:
            # Sequential pairs only
            for i in range(len(self.texts) - 1):
                pairs.append((self.texts[i].text_id, self.texts[i + 1].text_id))

        return pairs

    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary."""
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "text_count": len(self.texts),
            "texts": [
                {
                    "text_id": text.text_id,
                    "title": text.title,
                    "length": len(text.content),
                    "metadata": text.metadata,
                }
                for text in self.texts
            ],
            "strategy": {
                "analyses": self.strategy.analyses,
                "embedding_method": self.strategy.embedding_method,
                "embedding_model": self.strategy.embedding_model,
                "chunk_size": self.strategy.chunk_size,
                "compare_all_pairs": self.strategy.compare_all_pairs,
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two texts."""

    text1_id: str
    text2_id: str
    text1_title: str
    text2_title: str

    # Overall similarity (0-1)
    overall_similarity: float

    # Per-analyzer results
    analysis_comparisons: Dict[str, Any] = field(default_factory=dict)

    # Embedding similarity
    embedding_similarity: Optional[float] = None

    # Summary and interpretation
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text1_id": self.text1_id,
            "text2_id": self.text2_id,
            "text1_title": self.text1_title,
            "text2_title": self.text2_title,
            "overall_similarity": self.overall_similarity,
            "analysis_comparisons": self.analysis_comparisons,
            "embedding_similarity": self.embedding_similarity,
            "summary": self.summary,
        }


@dataclass
class ComparisonMatrix:
    """Matrix of all pairwise comparisons in a collection."""

    collection_id: str
    results: Dict[str, ComparisonResult] = field(default_factory=dict)

    # Statistics
    total_comparisons: int = 0
    completed_comparisons: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_result(self, result: ComparisonResult):
        """Add comparison result to matrix."""
        # Create key from sorted IDs to avoid duplicates
        key = self._make_key(result.text1_id, result.text2_id)
        self.results[key] = result
        self.completed_comparisons = len(self.results)

    def get_result(self, text1_id: str, text2_id: str) -> Optional[ComparisonResult]:
        """Get comparison result between two texts."""
        key = self._make_key(text1_id, text2_id)
        return self.results.get(key)

    def _make_key(self, text1_id: str, text2_id: str) -> str:
        """Create consistent key for text pair."""
        # Sort to ensure (A, B) and (B, A) have same key
        ids = sorted([text1_id, text2_id])
        return f"{ids[0]}:{ids[1]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to dictionary."""
        return {
            "collection_id": self.collection_id,
            "total_comparisons": self.total_comparisons,
            "completed_comparisons": self.completed_comparisons,
            "progress": (
                self.completed_comparisons / self.total_comparisons * 100
                if self.total_comparisons > 0 else 0
            ),
            "results": {
                key: result.to_dict()
                for key, result in self.results.items()
            },
            "created_at": self.created_at.isoformat(),
        }
