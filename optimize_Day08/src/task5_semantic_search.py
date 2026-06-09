"""Task 5 - semantic search over the local vector index."""

from __future__ import annotations

from .local_rag_utils import (
    REAL_EMBEDDING_MODEL,
    cosine_similarity,
    lexical_overlap_score,
    load_chunks,
    load_vector_index,
    neural_embeddings,
    normalize_text,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top_k chunks ranked by real neural embedding similarity."""
    if top_k <= 0:
        return []

    query_embedding = neural_embeddings([normalize_text(query)], REAL_EMBEDDING_MODEL)[0]
    chunks = load_vector_index()
    if not chunks:
        chunks = load_chunks()

    results: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        doc_embedding = chunk.get("embedding")
        if doc_embedding is None:
            doc_embedding = neural_embeddings([content], REAL_EMBEDDING_MODEL)[0]
        dense_score = max(0.0, cosine_similarity(query_embedding, doc_embedding))
        overlap_score = lexical_overlap_score(query, content)
        score = 0.9 * dense_score + 0.1 * overlap_score
        if score <= 0:
            continue
        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": dict(chunk.get("metadata", {})),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
