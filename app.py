"""Simple local interface for document uploads and cited answers."""

from pathlib import Path

import streamlit as st

from config import UPLOAD_DIR, validate_keys
from ingest import ingest_file
from rag import answer_question

st.set_page_config(page_title="AI Knowledge Base", page_icon="📚")
st.title("📚 AI Knowledge Base")
st.caption("Upload documents, then ask a complete question about them.")

try:
    validate_keys()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Preserve displayed messages between Streamlit's script reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_message(message):
    """Display an answer together with its numbered source excerpts."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander("Retrieved sources — check citations here"):
                for source in sources:
                    st.text(f"[{source['id']}] {source['label']}")
                    st.text(source["text"])
                    st.divider()


with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader(
        "Choose PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    st.caption("Document text is sent to OpenAI for processing.")
    # Index only on a button click, not whenever the script reruns.
    if st.button("Index documents", disabled=not uploads):
        for uploaded in uploads:
            filename = Path(uploaded.name).name
            try:
                with st.spinner(f"Indexing {filename}..."):
                    path = UPLOAD_DIR / filename
                    path.write_bytes(uploaded.getvalue())
                    added = ingest_file(path)
                st.success(f"{filename}: {added} new chunks.")
            except Exception as exc:
                st.error(f"{filename}: {exc}")

    if st.button("Clear chat"):
        st.session_state.messages = []
    st.caption("Clearing chat keeps your indexed documents.")


for message in st.session_state.messages:
    show_message(message)

if question := st.chat_input("Ask about your documents"):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)
    try:
        with st.spinner("Finding sources and writing an answer..."):
            result = answer_question(question)
        reply = {"role": "assistant", "content": result["answer"],
                 "sources": result["sources"]}
        st.session_state.messages.append(reply)
        show_message(reply)
    except Exception as exc:
        st.error(f"Could not answer: {exc}")