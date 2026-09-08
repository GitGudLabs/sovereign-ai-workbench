from abc import ABC, abstractmethod
from typing import List
from backend.models.document import Chunk

class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """Add chunks to the vector store."""
        pass
