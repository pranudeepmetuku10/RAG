"""
Document loading and chunking utilities.
Supports: PDF, TXT, DOCX, CSV, Markdown
"""

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


# Map file extensions to their respective LangChain loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_single_file(file_path: str) -> List[Document]:
    """Load a single file and return a list of Document objects."""
    ext = Path(file_path).suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )
    loader = LOADER_MAP[ext](file_path)
    docs = loader.load()
    # Tag every document with its source filename
    for doc in docs:
        doc.metadata["source"] = Path(file_path).name
    return docs


def load_files(file_paths: List[str]) -> List[Document]:
    """Load multiple files and return a combined list of Documents."""
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(load_single_file(fp))
        except Exception as e:
            print(f"⚠️  Skipping {fp}: {e}")
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller, overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
