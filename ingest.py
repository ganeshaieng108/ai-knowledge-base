"""Read files, split text, and save embeddings in local Chroma."""

import hashlib
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, validate_keys,
)


def get_store():
    """Reuse the same database and embedding model for ingestion and QA."""
    validate_keys()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )


def load_documents(path):
    """Keep PDF pages separate so chunks retain their page number."""
    if path.suffix.lower() == ".pdf":
        pages = PdfReader(str(path)).pages
        docs = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": path.name, "page": number},
            )
            for number, page in enumerate(pages, start=1)
        ]
    elif path.suffix.lower() == ".txt":
        docs = [Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )]
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return [doc for doc in docs if doc.page_content.strip()]


def ingest_file(file_path):
    path = Path(file_path)
    docs = load_documents(path)
    if not docs:
        raise ValueError("No readable text found. Scanned PDFs need OCR.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    # Stable IDs let us skip chunks already embedded on a previous run.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = [f"{path.name}:{digest}:{i}" for i in range(len(chunks))]
    store = get_store()
    existing = set(store.get(where={"source": path.name})["ids"])
    pending = [(doc, key) for doc, key in zip(chunks, ids)
               if key not in existing]
    # Small batches also allow interrupted imports to resume.
    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        store.add_documents(
            documents=[doc for doc, _ in batch],
            ids=[key for _, key in batch],
        )
    # Replace an older version with the same filename after success.
    stale = list(existing - set(ids))
    if stale:
        store.delete(ids=stale)
    return len(pending)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python ingest.py "path/to/file.pdf"')
    print(f"Added {ingest_file(sys.argv[1])} new chunks.")