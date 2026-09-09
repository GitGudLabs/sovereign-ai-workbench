from dataclasses import dataclass
from typing import List
from backend.rag.interfaces import SearchResult

@dataclass
class SourceCitation:
    doc_id: str
    filename: str
    page_number: int
    chunk_id: str
    score: float

@dataclass
class RAGContext:
    formatted_context: str
    citations: List[SourceCitation]

class ContextBuilder:
    """
    Builds structured, bounded LLM context strings with source citations.
    """
    def __init__(self, max_context_chars: int = 4000):
        self.max_context_chars = max_context_chars

    def build_context(self, search_results: List[SearchResult]) -> RAGContext:
        formatted_chunks = []
        citations = []
        total_chars = 0

        for idx, result in enumerate(search_results, start=1):
            chunk = result.chunk
            filename = chunk.metadata.get("filename", "Unknown Document")
            
            chunk_header = f"[Source #{idx} | Doc: {filename} | Page: {chunk.page_number} | Chunk ID: {chunk.chunk_id}]"
            chunk_body = f"{chunk_header}\n{chunk.text}\n"

            if total_chars + len(chunk_body) > self.max_context_chars:
                break

            formatted_chunks.append(chunk_body)
            total_chars += len(chunk_body)

            citations.append(
                SourceCitation(
                    doc_id=chunk.doc_id,
                    filename=filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    score=result.score
                )
            )

        formatted_text = "\n---\n".join(formatted_chunks) if formatted_chunks else "No relevant context found."
        return RAGContext(formatted_context=formatted_text, citations=citations)
