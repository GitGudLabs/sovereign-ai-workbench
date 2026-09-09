from dataclasses import dataclass
from typing import List, Optional
from backend.rag.interfaces import VectorStore, BaseEmbedder, BaseLLM
from backend.rag.retriever import VectorRetriever
from backend.rag.context import ContextBuilder, SourceCitation

@dataclass
class RAGResponse:
    query: str
    answer: str
    context: str
    sources: List[SourceCitation]

class RAGQueryEngine:
    """
    End-to-End RAG Query Engine: Query -> Embedding -> Retrieval -> Context Construction -> LLM -> Grounded Answer + Sources
    """
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        top_k: int = 3,
        max_context_chars: int = 4000
    ):
        self.retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        self.context_builder = ContextBuilder(max_context_chars=max_context_chars)
        self.llm = llm
        self.top_k = top_k

    def query(self, user_query: str, system_prompt: Optional[str] = None) -> RAGResponse:
        # 1. Retrieve relevant chunks
        search_results = self.retriever.retrieve(user_query, top_k=self.top_k)

        # 2. Build context and citations
        rag_context = self.context_builder.build_context(search_results)

        # 3. Construct prompt
        default_system_prompt = (
            "You are a helpful sovereign AI assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, state clearly that the information is not available in the documents. "
            "Do not make up facts."
        )
        sys_prompt = system_prompt or default_system_prompt
        
        full_prompt = f"Context:\n{rag_context.formatted_context}\n\nQuestion: {user_query}\nAnswer:"

        # 4. Generate answer from LLM
        answer = self.llm.generate(prompt=full_prompt, system_prompt=sys_prompt)

        return RAGResponse(
            query=user_query,
            answer=answer,
            context=rag_context.formatted_context,
            sources=rag_context.citations
        )
