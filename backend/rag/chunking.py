from typing import List
from backend.models.document import Document, Chunk

class BasicChunker:
    """
    A simple chunker that splits text into fixed-size chunks based on character count.
    In a real RAG scenario, this might use semantic splitting or tokenization.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        chunk_idx = 0
        
        for page in document.pages:
            text = page.text
            if not text:
                continue
                
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                
                # If we're not at the end of the text, try to find a space to split at
                if end < text_len:
                    # Look for the last space within the chunk
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + self.chunk_size // 2:
                        end = last_space
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            doc_id=document.id,
                            chunk_id=f"{document.id}_p{page.page_number}_c{chunk_idx}",
                            page_number=page.page_number,
                            text=chunk_text,
                            metadata={"filename": document.filename}
                        )
                    )
                    chunk_idx += 1
                
                start = end - self.overlap
                if start < 0:
                    start = 0
                if end == text_len:
                    break
                    
        return chunks
