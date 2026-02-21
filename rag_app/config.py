"""
Configuration for the RAG application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

# ── Chunking Settings ─────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval Settings ────────────────────────────────────────
TOP_K = 5  # number of chunks to retrieve

# ── Supported File Types ──────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md"}
