"""
RAG Engine for Executive Document Analyzer.

Provides:
  - Document ingestion with vector indexing
  - Conversational Q&A grounded in documents
  - Auto-extraction of executive insights & highlights
"""

from typing import List, Optional, Dict, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TEMPERATURE,
    TOP_K,
)
from document_loader import load_files, chunk_documents


# ── Prompt templates for executive analysis ───────────────────

EXECUTIVE_SUMMARY_PROMPT = """You are a senior executive analyst. Analyze the following document content and produce a structured executive briefing.

DOCUMENT CONTENT:
{context}

Produce the following sections. Be specific, cite numbers/dates/names when available:

## Executive Summary
A 3-5 sentence high-level overview of what these documents contain.

## Key Findings
Bullet-point list of the most important facts, figures, and conclusions.

## Action Items & Decisions Required
Any tasks, deadlines, or decisions that require executive attention.

## Risks & Concerns
Potential risks, red flags, or areas of concern identified in the documents.

## Financial Highlights
Any revenue, cost, budget, or financial data mentioned (if applicable, otherwise state "No financial data found").

## Important Dates & Deadlines
Chronological list of dates and deadlines mentioned (if applicable).

## People & Organizations
Key people, teams, or organizations referenced and their roles/relevance.

Be precise and factual. Only report what is in the documents — do not speculate."""


HIGHLIGHTS_PROMPT = """You are an executive document analyst. Review the following content and extract the MOST CRITICAL pieces of information that an executive must see immediately.

DOCUMENT CONTENT:
{context}

Return a JSON array of highlight objects. Each object must have:
- "type": one of "action_item", "deadline", "financial", "risk", "decision", "key_metric", "important_fact"
- "text": the exact highlight text (1-2 sentences)
- "urgency": "high", "medium", or "low"
- "source": the source document name if known

Return ONLY the JSON array, no other text. Example:
[
  {{"type": "deadline", "text": "Board meeting scheduled for March 15th — presentation required", "urgency": "high", "source": "memo.pdf"}},
  {{"type": "financial", "text": "Q3 revenue increased 12% YoY to $4.2M", "urgency": "medium", "source": "report.xlsx"}}
]"""


QA_SYSTEM_PROMPT = """You are an executive-level document analyst. Answer questions based ONLY on the provided document context. Your responses should be:

1. **Direct and concise** — executives value brevity
2. **Data-driven** — cite specific numbers, dates, and facts from the documents
3. **Actionable** — when relevant, suggest what actions might follow
4. **Honest** — if the documents don't contain the answer, say so clearly

If asked for opinions or analysis, ground them strictly in the document content.

Context from uploaded documents:
{context}"""


class RAGEngine:
    """Executive-grade RAG pipeline: ingest → embed → analyze → answer."""

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
        self.analysis_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=4000,
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
        self.file_names: List[str] = []
        self.raw_docs: List[Document] = []

    # ── Ingestion ─────────────────────────────────────────────

    def ingest(self, file_paths: List[str]) -> Dict[str, Any]:
        """Load files, chunk them, build the vector store.
        Returns a dict with status info and any loading errors."""
        raw_docs, errors = load_files(file_paths)
        if not raw_docs:
            return {
                "success": False,
                "message": "No documents could be loaded. Check file types.",
                "errors": errors,
            }

        self.raw_docs = raw_docs
        chunks = chunk_documents(raw_docs)
        self.chunk_count = len(chunks)
        self.file_count = len(file_paths)
        self.file_names = [p.split("/")[-1] for p in file_paths]

        # Build vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Reset conversation memory
        self.memory.clear()

        # Build QA chain
        self._build_chain()

        return {
            "success": True,
            "message": (
                f"Ingested {self.file_count} file(s) → "
                f"{self.chunk_count} chunks indexed."
            ),
            "files": self.file_names,
            "chunks": self.chunk_count,
            "errors": errors,
        }

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

    # ── Executive Summary ─────────────────────────────────────

    def generate_executive_summary(self) -> str:
        """Generate a structured executive briefing from all ingested docs."""
        if not self.raw_docs:
            return "⚠️ No documents ingested yet."

        # Combine raw document text (truncate to fit context window)
        combined = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in self.raw_docs
        )
        # Truncate if too long (leave room for prompt)
        if len(combined) > 80_000:
            combined = combined[:80_000] + "\n\n[... content truncated for analysis ...]"

        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a senior executive analyst."),
             ("human", EXECUTIVE_SUMMARY_PROMPT)]
        )
        chain = prompt | self.analysis_llm
        result = chain.invoke({"context": combined})
        return result.content

    # ── Key Highlights ────────────────────────────────────────

    def extract_highlights(self) -> str:
        """Extract critical highlights as structured JSON."""
        if not self.raw_docs:
            return "[]"

        combined = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in self.raw_docs
        )
        if len(combined) > 80_000:
            combined = combined[:80_000] + "\n\n[... content truncated ...]"

        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are an executive document analyst."),
             ("human", HIGHLIGHTS_PROMPT)]
        )
        chain = prompt | self.analysis_llm
        result = chain.invoke({"context": combined})
        return result.content

    # ── Question-answering ────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question against the ingested documents."""
        if self.chain is None:
            return {
                "answer": "⚠️ Please upload and ingest documents first.",
                "sources": [],
            }

        result = self.chain.invoke({"question": question})
        answer = result["answer"]

        sources = result.get("source_documents", [])
        unique_sources = sorted(
            {doc.metadata.get("source", "unknown") for doc in sources}
        )

        return {
            "answer": answer,
            "sources": unique_sources,
        }

    # ── Quick-ask (single-shot, no memory) ────────────────────

    def quick_analyze(self, question: str) -> str:
        """Run a single analytical query against all documents (no memory)."""
        if not self.raw_docs:
            return "⚠️ No documents ingested yet."

        combined = "\n\n".join(d.page_content for d in self.raw_docs)
        if len(combined) > 80_000:
            combined = combined[:80_000]

        prompt = ChatPromptTemplate.from_messages(
            [("system", QA_SYSTEM_PROMPT),
             ("human", "{question}")]
        )
        chain = prompt | self.analysis_llm
        result = chain.invoke({"context": combined, "question": question})
        return result.content

    # ── Reset ─────────────────────────────────────────────────

    def reset(self):
        """Clear all state."""
        self.vectorstore = None
        self.chain = None
        self.memory.clear()
        self.file_count = 0
        self.chunk_count = 0
        self.file_names = []
        self.raw_docs = []
        return "All documents and chat history cleared."
