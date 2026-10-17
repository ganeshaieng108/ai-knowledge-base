
"""Shared settings and local storage paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to this file, regardless of the terminal location.
BASE_DIR = Path(__file__).resolve().parent

# Load secrets from the project's .env file.
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "knowledge_base"


def validate_keys():
    """Catch missing keys before starting ingestion or chat."""
    required = ("OPENAI_API_KEY" ,)
    missing = [
        name for name in required
        if not os.getenv(name, "").strip()
        or os.getenv(name, "").startswith("replace_with_")
    ]
    if missing:
        raise ValueError(f"Set these keys in .env: {', '.join(missing)}")


if __name__ == "__main__":
    validate_keys()
    print("Configuration OK: keys are present and folders are ready.")