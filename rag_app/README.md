# 📄 RAG Document Q&A App

A Gradio-powered app that lets you **upload documents** and **ask questions** — answers are generated using Retrieval-Augmented Generation (RAG) with OpenAI.

## Features

- **Multi-format support** — PDF, TXT, DOCX, CSV, Markdown
- **Conversational memory** — follow-up questions understand context
- **Source attribution** — see which documents informed each answer
- **Clean UI** — drag-and-drop upload + chat interface

## Architecture

```
rag_app/
├── app.py               # Gradio UI
├── rag_engine.py         # RAG pipeline (embed → retrieve → answer)
├── document_loader.py    # File loading & chunking
├── config.py             # All settings in one place
├── requirements.txt      # Python dependencies
└── README.md             # You are here
```

**Pipeline:** Upload → Load → Chunk → Embed (OpenAI) → FAISS Index → Retrieve → LLM Answer

## Quick Start

### 1. Install dependencies

```bash
cd rag_app
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the `rag_app/` folder:

```
OPENAI_API_KEY=sk-your-key-here
```

Or export it directly:

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the app

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

## Usage

1. **Upload** one or more files using the left panel
2. Click **"📥 Ingest Documents"** to process them
3. **Ask questions** in the chat — answers come from your uploaded content
4. Click **"🗑️ Clear All"** to reset and start over

## Configuration

Edit `config.py` to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for answering |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved |
| `TEMPERATURE` | `0.2` | LLM creativity (0 = focused) |
