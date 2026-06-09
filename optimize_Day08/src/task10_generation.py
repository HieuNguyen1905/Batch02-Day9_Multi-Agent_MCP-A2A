"""Task 10 - RAG generation with citations."""

from __future__ import annotations

from .local_rag_utils import source_label
from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese. Use only the provided context and cite
every factual claim with the source label in brackets. If evidence is missing,
say that the information cannot be verified from the current sources."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place the strongest chunk first and the second strongest near the end."""
    if len(chunks) <= 2:
        return list(chunks)
    front = list(chunks[0::2])
    back = list(reversed(chunks[1::2]))
    return front + back


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _snippet(text: str, max_len: int = 260) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Return an extractive answer with citations and source metadata."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    _ = format_context(reordered)

    if not reordered:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": [],
            "retrieval_source": "none",
        }

    answer_parts: list[str] = []
    for chunk in reordered[: min(3, len(reordered))]:
        label = source_label(chunk)
        answer_parts.append(f"{_snippet(chunk.get('content', ''))} [{label}]")

    answer = (
        "Dua tren cac nguon da index, thong tin lien quan nhat la: "
        + " ".join(answer_parts)
    )
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    print(generate_with_citation("Hinh phat tang tru ma tuy?")["answer"])
