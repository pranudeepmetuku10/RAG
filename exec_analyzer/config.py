"""
Configuration for the Executive Document Analyzer.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM Settings ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.15  # Low temperature for factual executive analysis

# ── Chunking Settings ─────────────────────────────────────────
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# ── Retrieval Settings ────────────────────────────────────────
TOP_K = 8  # Retrieve more context for executive-grade answers

# ── Supported File Types ──────────────────────────────────────
# Text-based formats
TEXT_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md", ".xlsx"}
# Image formats (OCR will be applied)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
# All supported
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS

# ── OCR Settings ──────────────────────────────────────────────
TESSERACT_LANG = "eng"  # Tesseract language pack
OCR_DPI = 300  # DPI for PDF-to-image conversion
