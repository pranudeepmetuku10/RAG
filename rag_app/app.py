"""
Gradio UI for the RAG Document Q&A App.
Upload documents, ask questions, get AI-powered answers.
"""

import gradio as gr
from rag_engine import RAGEngine

# ── Instantiate engine ────────────────────────────────────────
engine = RAGEngine()


# ── Callbacks ─────────────────────────────────────────────────
def handle_upload(files):
    """Ingest uploaded files into the RAG pipeline."""
    if not files:
        return "⚠️ No files uploaded."
    file_paths = [f.name for f in files]
    status = engine.ingest(file_paths)
    return status


def handle_question(question, chat_history):
    """Send a question to the RAG engine and return the answer."""
    if not question.strip():
        return chat_history, ""
    answer = engine.ask(question)
    chat_history.append((question, answer))
    return chat_history, ""


def handle_clear():
    """Reset everything."""
    msg = engine.reset()
    return [], msg  # clear chatbot, update status


# ── Gradio Interface ─────────────────────────────────────────
with gr.Blocks(
    title="📄 RAG Document Q&A",
    theme=gr.themes.Soft(),
    css="""
        .contain { max-width: 900px; margin: auto; }
        footer { display: none !important; }
    """,
) as app:

    gr.Markdown(
        """
        # 📄 RAG Document Q&A
        Upload your documents (PDF, TXT, DOCX, CSV, Markdown), then ask any
        question — the AI will answer **based only on your uploaded content**.
        """
    )

    with gr.Row():
        # ── Left column: upload & controls ────────────────────
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".docx", ".csv", ".md"],
                type="filepath",
            )
            upload_btn = gr.Button("📥 Ingest Documents", variant="primary")
            status_box = gr.Markdown(value="*No documents loaded yet.*")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

        # ── Right column: chat ────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                show_copy_button=True,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # ── Wire up events ────────────────────────────────────────
    upload_btn.click(
        fn=handle_upload,
        inputs=[file_upload],
        outputs=[status_box],
    )

    send_btn.click(
        fn=handle_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )
    question_box.submit(
        fn=handle_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    clear_btn.click(
        fn=handle_clear,
        inputs=[],
        outputs=[chatbot, status_box],
    )


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
