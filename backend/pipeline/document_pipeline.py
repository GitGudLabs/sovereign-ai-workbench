import os
import uuid
import logging
from typing import Optional, List

from backend.ocr.ocr import ocr_document
from backend.models.document import Document, Page, Chunk
from backend.rag.chunking import BasicChunker
from backend.rag.interfaces import VectorStore, BaseEmbedder

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    pass

class OCRError(PipelineError):
    pass

class DocumentPipeline:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[BasicChunker] = None,
        embedder: Optional[BaseEmbedder] = None
    ):
        self.vector_store = vector_store
        self.chunker = chunker or BasicChunker()
        self.embedder = embedder

    def process_file(self, file_path: str) -> Document:
        """
        Process a file end-to-end: Validate -> OCR -> Normalize -> Chunk -> Embed & Store
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. OCR processing
        try:
            ocr_results = ocr_document(file_path)
        except Exception as e:
            logger.error(f"Unexpected error during OCR execution.")
            raise OCRError("OCR execution failed due to an internal error.")

        # 2. Validation & Normalization
        doc_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        document = Document(id=doc_id, filename=filename)

        if not ocr_results:
            raise OCRError("OCR returned empty results.")

        for item in ocr_results:
            page_text = item.get("text", "")
            
            # Catch OCR error strings
            if page_text.startswith("[ERROR:"):
                logger.error(f"OCR module returned an error for file {filename}")
                raise OCRError(f"OCR failed processing page: {page_text}")
                
            page_number = item.get("page", 1)
            document.pages.append(Page(page_number=page_number, text=page_text))

        # 3. Chunking
        try:
            chunks = self.chunker.chunk_document(document)
        except Exception as e:
            logger.error("Error during document chunking.")
            raise PipelineError("Failed to chunk document.")

        # 4. Storage / RAG Indexing
        if self.vector_store:
            try:
                embeddings = None
                if self.embedder:
                    embeddings = self.embedder.embed_chunks(chunks)
                self.vector_store.add_chunks(chunks, embeddings=embeddings)
            except Exception as e:
                logger.error("Error storing chunks to vector store.")
                raise PipelineError("Failed to store document in vector store.")

        return document
