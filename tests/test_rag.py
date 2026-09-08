import unittest
from backend.models.document import Chunk
from backend.rag.interfaces import BaseEmbedder, BaseLLM
from backend.rag.vector_store import InMemoryVectorStore, cosine_similarity
from backend.rag.retriever import VectorRetriever
from backend.rag.context import ContextBuilder
from backend.rag.pipeline import RAGQueryEngine

class MockEmbedder(BaseEmbedder):
    def embed_text(self, text: str):
        if "apple" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "banana" in text.lower():
            return [0.0, 1.0, 0.0]
        else:
            return [0.5, 0.5, 0.5]

    def embed_chunks(self, chunks):
        return [self.embed_text(c.text) for c in chunks]

class MockLLM(BaseLLM):
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return "Fruit information generated from context."

class TestRAGComponents(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.vector_store = InMemoryVectorStore(embedder=self.embedder)
        self.chunk1 = Chunk(doc_id="doc1", chunk_id="c1", page_number=1, text="Apples are red fruits.", metadata={"filename": "fruits.pdf"})
        self.chunk2 = Chunk(doc_id="doc1", chunk_id="c2", page_number=2, text="Bananas are yellow fruits.", metadata={"filename": "fruits.pdf"})

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        v3 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_vector_store_add_and_search(self):
        self.vector_store.add_chunks([self.chunk1, self.chunk2])
        
        query_vector = self.embedder.embed_text("apple")
        results = self.vector_store.search(query_vector, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertGreater(results[0].score, results[1].score)

    def test_retriever(self):
        self.vector_store.add_chunks([self.chunk1, self.chunk2])
        retriever = VectorRetriever(vector_store=self.vector_store, embedder=self.embedder)
        
        results = retriever.retrieve("tell me about banana", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_context_builder(self):
        self.vector_store.add_chunks([self.chunk1, self.chunk2])
        retriever = VectorRetriever(vector_store=self.vector_store, embedder=self.embedder)
        results = retriever.retrieve("apple", top_k=2)

        context_builder = ContextBuilder(max_context_chars=1000)
        rag_context = context_builder.build_context(results)

        self.assertIn("fruits.pdf", rag_context.formatted_context)
        self.assertEqual(len(rag_context.citations), 2)
        self.assertEqual(rag_context.citations[0].chunk_id, "c1")

    def test_rag_query_engine(self):
        self.vector_store.add_chunks([self.chunk1, self.chunk2])
        mock_llm = MockLLM()
        engine = RAGQueryEngine(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm=mock_llm
        )

        response = engine.query("What color are apples?")
        self.assertEqual(response.query, "What color are apples?")
        self.assertEqual(response.answer, "Fruit information generated from context.")
        self.assertGreater(len(response.sources), 0)
        self.assertEqual(response.sources[0].filename, "fruits.pdf")

if __name__ == '__main__':
    unittest.main()
