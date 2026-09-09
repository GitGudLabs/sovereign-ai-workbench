import math
import logging
from typing import List, Optional, Tuple
from backend.models.document import Chunk
from backend.rag.interfaces import VectorStore, SearchResult, BaseEmbedder

logger = logging.getLogger(__name__)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class InMemoryVectorStore(VectorStore):
    """
    A sovereign, local in-memory vector store supporting cosine similarity search.
    """
    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder
        self.items: List[Tuple[Chunk, List[float]]] = []

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> bool:
        if not chunks:
            return True

        if embeddings is None:
            if self.embedder is None:
                logger.warning("No embedder or pre-computed embeddings provided. Storing chunks with empty embeddings.")
                for chunk in chunks:
                    self.items.append((chunk, []))
                return True
            embeddings = self.embedder.embed_chunks(chunks)

        if len(chunks) != len(embeddings):
            raise ValueError("Length of chunks and embeddings must match.")

        for chunk, vector in zip(chunks, embeddings):
            self.items.append((chunk, vector))

        return True

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if not query_embedding or not self.items:
            return []

        results = []
        for chunk, vector in self.items:
            if not vector:
                continue
            score = cosine_similarity(query_embedding, vector)
            results.append(SearchResult(chunk=chunk, score=score))

        # Sort descending by similarity score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
