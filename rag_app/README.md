# 📄 RAG Document Q&A App

A **Gradio-powered web application** that lets you upload documents and ask questions about them. Answers are generated using **Retrieval-Augmented Generation (RAG)** — the AI reads your documents and responds based only on their content.

---

## Table of Contents

- [What This App Does](#what-this-app-does)
- [Features](#features)
- [Project Structure](#project-structure)
- [How RAG Works (Step by Step)](#how-rag-works-step-by-step)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
- [Running the App](#running-the-app)
- [Using the App](#using-the-app)
- [File-by-File Explanation](#file-by-file-explanation)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## What This App Does

You upload documents (PDF, Word, text files, CSVs, or Markdown), the app processes them into searchable chunks, and then you can ask any question in natural language. The AI retrieves the most relevant pieces of your documents and generates an accurate answer grounded in your content — not from its general training data.

---

## Features

- **Multi-format upload** — supports PDF, TXT, DOCX, CSV, and Markdown files
- **Intelligent chunking** — documents are split into overlapping segments so context is never lost at boundaries
- **Vector search** — chunks are embedded and indexed with FAISS for fast similarity retrieval
- **Conversational memory** — ask follow-up questions; the AI remembers the conversation context
- **Source attribution** — every answer shows which uploaded files were used
- **Clean web UI** — drag-and-drop file upload + chat interface powered by Gradio

---

## Project Structure

```
rag_app/
├── __init__.py            # Makes this a Python package
├── app.py                 # Gradio web UI — handles uploads, chat, and display
├── rag_engine.py          # Core RAG pipeline — ingestion, embedding, retrieval, answering
├── document_loader.py     # File loading (PDF/TXT/DOCX/CSV/MD) and text chunking
├── config.py              # Central configuration — models, chunk sizes, API keys
├── requirements.txt       # Python package dependencies
├── .env.example           # Template for your OpenAI API key
└── README.md              # This file
```

---

## How RAG Works (Step by Step)

RAG (Retrieval-Augmented Generation) combines a **search step** with a **generation step** to produce accurate, document-grounded answers:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌───────────┐
│  Upload   │ →  │  Load &  │ →  │  Embed   │ →  │  Store in │ →  │  Ready!   │
│  Files    │    │  Chunk   │    │  Chunks  │    │  FAISS    │    │           │
└──────────┘    └──────────┘    └──────────┘    └───────────┘    └───────────┘

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌───────────┐
│  Ask a   │ →  │  Embed   │ →  │ Retrieve │ →  │  Send to  │ →  │  Answer   │
│ Question │    │ Question │    │ Top-K    │    │   LLM     │    │ + Sources │
└──────────┘    └──────────┘    └──────────┘    └───────────┘    └───────────┘
```

### Detailed steps:

1. **Upload** — User drags files into the Gradio interface
2. **Load** — Each file is read using a format-specific loader (PyPDF for PDFs, docx2txt for Word, etc.)
3. **Chunk** — The full text is split into ~1000-character overlapping chunks using `RecursiveCharacterTextSplitter`. Overlap (200 chars) ensures no information is lost at split boundaries
4. **Embed** — Each chunk is converted into a numerical vector (embedding) using OpenAI's `text-embedding-3-small` model
5. **Index** — All embeddings are stored in a **FAISS** vector index for fast nearest-neighbor search
6. **Query** — When the user asks a question, the question is also embedded into a vector
7. **Retrieve** — FAISS finds the top-5 most similar chunks to the question
8. **Generate** — The retrieved chunks + the question are sent to OpenAI's `gpt-4o-mini` LLM, which generates a grounded answer
9. **Display** — The answer is shown in the chat UI with source file attribution

---

## Prerequisites

Before setting up, make sure you have:

- **Python 3.9+** installed ([download here](https://www.python.org/downloads/))
- **An OpenAI API key** ([get one here](https://platform.openai.com/api-keys))
- **pip** (comes with Python) or **conda** for package management
- **~500 MB disk space** for dependencies (FAISS, LangChain, Gradio, etc.)

---

## Setup Guide

### Step 1: Clone or navigate to the project

```bash
cd /path/to/rag_app
```

### Step 2: (Recommended) Create a virtual environment

This keeps dependencies isolated from your system Python:

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `langchain` | Orchestrates the RAG pipeline |
| `langchain-openai` | OpenAI embeddings & chat models |
| `langchain-community` | Document loaders & FAISS integration |
| `faiss-cpu` | Fast vector similarity search |
| `gradio` | Web UI framework |
| `pypdf` | PDF file reading |
| `docx2txt` | Word document reading |
| `unstructured` | Markdown file parsing |
| `python-dotenv` | Loads API key from `.env` file |

### Step 4: Set up your OpenAI API key

**Option A — `.env` file (recommended):**

```bash
# Copy the template
cp .env.example .env

# Edit it with your real key
nano .env   # or open in any text editor
```

Set the contents to:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**Option B — Environment variable:**

```bash
# macOS / Linux
export OPENAI_API_KEY=sk-proj-your-actual-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-proj-your-actual-key-here"
```

### Step 5: Verify the setup

```bash
python -c "from config import OPENAI_API_KEY; print('Key loaded!' if OPENAI_API_KEY else 'Key missing!')"
```

You should see `Key loaded!`.

---

## Running the App

```bash
python app.py
```

You'll see output like:

```
Running on local URL:  http://0.0.0.0:7860
```

Open **http://localhost:7860** in your browser.

To make the app accessible from other devices on your network, it's already bound to `0.0.0.0`. To create a public shareable link, change `share=False` to `share=True` in `app.py`.

---

## Using the App

### Step 1: Upload documents

- Click the **Upload Documents** area on the left panel
- Drag and drop files, or click to browse
- You can upload **multiple files at once**
- Supported formats: `.pdf`, `.txt`, `.docx`, `.csv`, `.md`

### Step 2: Ingest documents

- Click the **"📥 Ingest Documents"** button
- The app will load, chunk, embed, and index your files
- You'll see a status message like: `✅ Ingested 3 file(s) → 47 chunks indexed.`

### Step 3: Ask questions

- Type your question in the text box at the bottom right
- Press **Enter** or click **Send**
- The AI will answer based on your uploaded content
- Each answer includes **source file references**

### Step 4: Follow-up questions

- The app maintains **conversation memory**
- You can ask follow-up questions like "Can you elaborate on that?" or "What about the second point?"
- The AI remembers the previous Q&A context

### Step 5: Start over

- Click **"🗑️ Clear All"** to remove all documents, embeddings, and chat history
- Upload new documents and start fresh

---

## File-by-File Explanation

### `config.py` — Central Configuration

All tunable settings live here. Loads the OpenAI API key from a `.env` file using `python-dotenv`. You can change the LLM model, embedding model, chunk sizes, and retrieval depth without touching any other file.

### `document_loader.py` — File Loading & Chunking

- **`load_single_file(path)`** — Detects the file extension and uses the appropriate LangChain loader (PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, or UnstructuredMarkdownLoader)
- **`load_files(paths)`** — Loops through multiple files, loading each one and collecting all `Document` objects. Gracefully skips files that fail to load.
- **`chunk_documents(docs)`** — Uses `RecursiveCharacterTextSplitter` to break documents into ~1000-char chunks with 200-char overlap. The splitter tries to break at paragraph boundaries first, then sentences, then words.

### `rag_engine.py` — The RAG Pipeline

The `RAGEngine` class ties everything together:

- **`__init__()`** — Initializes the OpenAI embedding model, the ChatOpenAI LLM, conversation memory, and empty placeholders for the vector store and chain.
- **`ingest(file_paths)`** — Loads files → chunks them → creates FAISS vector store from embeddings → builds the conversational retrieval chain. Returns a status string.
- **`_build_chain()`** — Creates a `ConversationalRetrievalChain` that combines the FAISS retriever with the LLM and conversation memory.
- **`ask(question)`** — Runs the question through the chain, gets the answer and source documents, formats source attribution, and returns the full response.
- **`reset()`** — Clears all state (vector store, chain, memory, counters).

### `app.py` — Gradio Web Interface

- Sets up a two-column layout: left for uploads/controls, right for the chatbot
- **`handle_upload(files)`** — Extracts file paths from uploaded files and passes them to `engine.ingest()`
- **`handle_question(question, history)`** — Sends the question to `engine.ask()` and appends the Q&A to chat history
- **`handle_clear()`** — Resets the engine and clears the chatbot display
- Launches on port 7860, accessible from any network interface

---

## Configuration Reference

Edit `config.py` to customize behavior:

| Setting | Default | What It Controls |
|---------|---------|------------------|
| `OPENAI_API_KEY` | from `.env` | Your OpenAI API key for embeddings & LLM |
| `LLM_MODEL` | `gpt-4o-mini` | The OpenAI model used to generate answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | The model used to create vector embeddings |
| `TEMPERATURE` | `0.2` | LLM creativity (0 = deterministic, 1 = creative) |
| `CHUNK_SIZE` | `1000` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlapping characters between adjacent chunks |
| `TOP_K` | `5` | Number of relevant chunks retrieved per question |

### Tips:
- **Increase `CHUNK_SIZE`** (e.g., 2000) for documents with long, continuous passages
- **Increase `TOP_K`** (e.g., 10) if answers seem to miss relevant info
- **Lower `TEMPERATURE`** (e.g., 0.0) for more factual, less creative answers
- **Switch `LLM_MODEL`** to `gpt-4o` for higher quality (more expensive) answers

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Key missing!` when verifying setup | Make sure `.env` file is in the `rag_app/` folder and contains a valid key |
| `ModuleNotFoundError: No module named 'faiss'` | Run `pip install faiss-cpu` |
| `openai.AuthenticationError` | Your API key is invalid or expired — get a new one from OpenAI |
| App loads but answers are wrong | Try increasing `TOP_K` to retrieve more context, or reduce `CHUNK_SIZE` for finer granularity |
| PDF upload fails | Ensure `pypdf` is installed: `pip install pypdf` |
| DOCX upload fails | Ensure `docx2txt` is installed: `pip install docx2txt` |
| Port 7860 already in use | Change the port in `app.py`: `app.launch(server_port=7861)` |
| Want a public shareable link | Set `share=True` in `app.py`'s `app.launch()` call |

---

## License

This project is for educational and personal use.
