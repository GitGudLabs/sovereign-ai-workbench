import logging
from typing import List
from backend.rag.interfaces import VectorStore, BaseEmbedder, SearchResult

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    Retrieves top-k relevant chunks from a vector store for a given query text.
    """
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not query.strip():
            return []

        query_vector = self.embedder.embed_text(query)
        if not query_vector:
            logger.warning("Query embedding returned empty vector.")
            return []

        return self.vector_store.search(query_vector, top_k=top_k)
