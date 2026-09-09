from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from backend.models.document import Chunk

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add chunks (and optional pre-computed embeddings) to the vector store."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for top_k most similar chunks given a query embedding vector."""
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single string."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate vector embeddings for a list of Chunks."""
        pass

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text response from LLM."""
        pass
