"""
Simple ChunkIndexer implementation for test compatibility.
This is a simpler version that works with FAISS for vector similarity search.
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

try:
    import faiss
except ImportError:
    faiss = None


class ChunkIndexer:
    """Simple chunk indexer with FAISS integration."""

    def __init__(self, dimension: int = 384, index_type: str = 'Flat', nlist: int = 100):
        """Initialize the chunk indexer.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.chunks = []

    def build_index(self, embeddings: np.ndarray, chunks: List[str]):
        """Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of shape (n_chunks, dimension)
            chunks: List of text chunks
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')

        self.chunks = chunks

        if faiss is None:
            # Fallback without FAISS
            self.embeddings = embeddings
            self.index = None
            return

        # Create appropriate FAISS index
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.index.train(embeddings)
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)

    def add_chunks(self, chunks: List[str], embeddings: List[np.ndarray]):
        """Add new chunks to the index.

        Args:
            chunks: List of text chunks
            embeddings: List of embeddings
        """
        if not embeddings:
            return

        embeddings_array = np.array(embeddings).astype('float32')

        if self.index is None:
            # Initialize index if not exists
            self.build_index(embeddings_array, chunks)
        else:
            # Add to existing index
            self.chunks.extend(chunks)
            if faiss and self.index:
                self.index.add(embeddings_array)
            else:
                # Fallback: concatenate embeddings
                if hasattr(self, 'embeddings'):
                    self.embeddings = np.vstack([self.embeddings, embeddings_array])
                else:
                    self.embeddings = embeddings_array

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of dicts with 'chunk', 'score', and 'index' keys
        """
        if query_embedding is None or k <= 0:
            return []

        if not self.chunks:
            return []

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype('float32')

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if faiss and self.index:
            # Use FAISS search
            distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < len(self.chunks):
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(distances[0][i]),
                        'index': int(idx)
                    })
        else:
            # Fallback: brute force search
            if hasattr(self, 'embeddings'):
                distances = np.sum((self.embeddings - query_embedding) ** 2, axis=1)
                sorted_indices = np.argsort(distances)[:k]
                results = []
                for idx in sorted_indices:
                    if idx < len(self.chunks):
                        results.append({
                            'chunk': self.chunks[idx],
                            'score': float(distances[idx]),
                            'index': int(idx)
                        })
            else:
                results = []

        return results

    def search_batch(self, queries: List[np.ndarray], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Search for multiple queries.

        Args:
            queries: List of query embeddings
            k: Number of results per query

        Returns:
            List of search results for each query
        """
        if not queries:
            return []

        results = []
        for query in queries:
            results.append(self.search(query, k))
        return results

    def get_chunk(self, index: int) -> Optional[str]:
        """Get chunk by index.

        Args:
            index: Chunk index

        Returns:
            Chunk text or None if invalid index
        """
        if index < 0 or index >= len(self.chunks):
            return None
        return self.chunks[index]

    def get_chunks_by_indices(self, indices: List[int]) -> List[str]:
        """Get multiple chunks by indices.

        Args:
            indices: List of chunk indices

        Returns:
            List of chunk texts
        """
        chunks = []
        for idx in indices:
            chunk = self.get_chunk(idx)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def clear(self):
        """Clear the index and chunks."""
        self.index = None
        self.chunks = []
        if hasattr(self, 'embeddings'):
            delattr(self, 'embeddings')

    def save(self, filepath: str):
        """Save index to file.

        Args:
            filepath: Path to save the index
        """
        # Save chunks
        chunks_path = filepath.replace('.idx', '_chunks.json')
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        # Save index
        if faiss and self.index:
            faiss.write_index(self.index, filepath)
        elif hasattr(self, 'embeddings'):
            np.save(filepath.replace('.idx', '.npy'), self.embeddings)

    def load(self, filepath: str):
        """Load index from file.

        Args:
            filepath: Path to load the index from
        """
        # Load chunks
        chunks_path = filepath.replace('.idx', '_chunks.json')
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Load index
        if faiss and os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
        else:
            npy_path = filepath.replace('.idx', '.npy')
            if os.path.exists(npy_path):
                self.embeddings = np.load(npy_path, allow_pickle=True)

    @property
    def ntotal(self) -> int:
        """Get total number of indexed items."""
        if faiss and self.index:
            return self.index.ntotal
        return len(self.chunks)