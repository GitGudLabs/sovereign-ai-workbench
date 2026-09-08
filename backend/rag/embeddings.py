import logging
from typing import List
from backend.models.document import Chunk
from backend.rag.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)

class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding generator using Ollama Python API.
    """
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        try:
            import ollama
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"Error generating embedding via Ollama: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        embeddings = []
        for chunk in chunks:
            vector = self.embed_text(chunk.text)
            embeddings.append(vector)
        return embeddings
