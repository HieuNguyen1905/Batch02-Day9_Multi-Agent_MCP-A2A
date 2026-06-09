"""Task 4 - chunk Markdown documents and build a local vector index."""

from __future__ import annotations

from pathlib import Path

from .local_rag_utils import (
    REAL_EMBEDDING_DIM,
    REAL_EMBEDDING_MODEL,
    VECTOR_INDEX_PATH,
    chunk_documents as split_documents,
    load_markdown_documents,
    neural_embeddings,
    save_vector_index,
)


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Recursive character chunking is robust for mixed legal/news Markdown.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Real multilingual neural embeddings for demo and retrieval.
EMBEDDING_MODEL = REAL_EMBEDDING_MODEL
EMBEDDING_DIM = REAL_EMBEDDING_DIM
VECTOR_STORE = "local-json-sentence-transformers"


def load_documents() -> list[dict]:
    """Read all Markdown files from data/standardized/."""
    return load_markdown_documents(STANDARDIZED_DIR)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    return split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add real multilingual sentence-transformer embeddings to every chunk."""
    texts = [chunk.get("content", "") for chunk in chunks]
    vectors = neural_embeddings(texts, EMBEDDING_MODEL)

    embedded: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        item = {**chunk, "metadata": dict(chunk.get("metadata", {}))}
        item["embedding"] = vector
        item["metadata"]["embedding_model"] = EMBEDDING_MODEL
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist the local index as JSON for inspection and reuse."""
    return save_vector_index(chunks, VECTOR_INDEX_PATH)


def run_pipeline() -> Path:
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    return index_to_vectorstore(embedded)


if __name__ == "__main__":
    output = run_pipeline()
    print(f"Indexed to {output}")
