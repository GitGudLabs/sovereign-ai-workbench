import unittest
from unittest.mock import patch
import os
import tempfile

from backend.pipeline.document_pipeline import DocumentPipeline, OCRError, PipelineError
from backend.rag.interfaces import VectorStore

class MockVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks, embeddings=None):
        self.chunks.extend(chunks)
        return True

    def search(self, query_embedding, top_k=5):
        return []

class TestDocumentPipeline(unittest.TestCase):
    def setUp(self):
        self.vector_store = MockVectorStore()
        self.pipeline = DocumentPipeline(vector_store=self.vector_store)

    @patch('backend.pipeline.document_pipeline.ocr_document')
    def test_successful_processing(self, mock_ocr):
        # Mock OCR output
        mock_ocr.return_value = [
            {"page": 1, "text": "This is a test document with some text."},
            {"page": 2, "text": "Second page with more text for chunking."}
        ]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name

        try:
            doc = self.pipeline.process_file(tmp_path)
            
            self.assertEqual(len(doc.pages), 2)
            self.assertEqual(doc.pages[0].text, "This is a test document with some text.")
            
            # Check chunks were generated and stored
            self.assertGreater(len(self.vector_store.chunks), 0)
            self.assertEqual(self.vector_store.chunks[0].page_number, 1)
        finally:
            os.unlink(tmp_path)

    @patch('backend.pipeline.document_pipeline.ocr_document')
    def test_ocr_error_string(self, mock_ocr):
        mock_ocr.return_value = [
            {"page": 1, "text": "[ERROR: OCR failed for this page]"}
        ]
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with self.assertRaises(OCRError):
                self.pipeline.process_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process_file("nonexistent_file.pdf")

if __name__ == '__main__':
    unittest.main()
