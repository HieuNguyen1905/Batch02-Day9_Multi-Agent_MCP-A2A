"""Task 6 - lexical search using a small BM25 implementation."""

from __future__ import annotations

from .local_rag_utils import SimpleBM25, load_chunks, tokenize


CORPUS: list[dict] = []


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = load_chunks()
    return CORPUS


def build_bm25_index(corpus: list[dict]) -> SimpleBM25:
    tokenized_corpus = [tokenize(doc.get("content", "")) for doc in corpus]
    return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top_k chunks ranked by BM25 keyword score."""
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results: list[dict] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        doc = corpus[idx]
        results.append(
            {
                "content": doc.get("content", ""),
                "score": score,
                "metadata": dict(doc.get("metadata", {})),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
