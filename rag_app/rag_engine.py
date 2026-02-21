"""
RAG Engine — builds the vector store and answers questions.
"""

from typing import List, Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TEMPERATURE,
    TOP_K,
)
from document_loader import load_files, chunk_documents


class RAGEngine:
    """Encapsulates the full RAG pipeline: ingest → embed → retrieve → answer."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
        self.vectorstore: Optional[FAISS] = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.file_count = 0
        self.chunk_count = 0

    # ── Ingestion ─────────────────────────────────────────────
    def ingest(self, file_paths: List[str]) -> str:
        """Load files, chunk them, build/update the vector store."""
        # Load & chunk
        raw_docs = load_files(file_paths)
        if not raw_docs:
            return "❌ No documents could be loaded. Check file types."

        chunks = chunk_documents(raw_docs)
        self.chunk_count = len(chunks)
        self.file_count = len(file_paths)

        # Build (or rebuild) vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Reset conversation memory for fresh documents
        self.memory.clear()

        # Build the QA chain
        self._build_chain()

        return (
            f"✅ Ingested **{self.file_count}** file(s) → "
            f"**{self.chunk_count}** chunks indexed."
        )

    # ── Chain construction ────────────────────────────────────
    def _build_chain(self):
        if self.vectorstore is None:
            return
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
        )

    # ── Question-answering ────────────────────────────────────
    def ask(self, question: str) -> str:
        """Ask a question against the ingested documents."""
        if self.chain is None:
            return "⚠️ Please upload and ingest documents first."

        result = self.chain.invoke({"question": question})
        answer = result["answer"]

        # Append source references
        sources = result.get("source_documents", [])
        if sources:
            unique_sources = sorted(
                {doc.metadata.get("source", "unknown") for doc in sources}
            )
            answer += "\n\n📄 **Sources:** " + ", ".join(unique_sources)

        return answer

    # ── Reset ─────────────────────────────────────────────────
    def reset(self):
        """Clear all state so the user can start fresh."""
        self.vectorstore = None
        self.chain = None
        self.memory.clear()
        self.file_count = 0
        self.chunk_count = 0
        return "🗑️ All documents and chat history cleared."
