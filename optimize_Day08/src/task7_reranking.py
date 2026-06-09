"""Task 7 - local reranking strategies."""

from __future__ import annotations

from .local_rag_utils import (
    cosine_similarity,
    hashing_embedding,
    lexical_overlap_score,
)


def _base_score(value: float) -> float:
    return value / (abs(value) + 1.0) if value else 0.0


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Lightweight cross-encoder substitute based on query/document overlap."""
    reranked: list[dict] = []
    for candidate in candidates:
        overlap = lexical_overlap_score(query, candidate.get("content", ""))
        original = _base_score(float(candidate.get("score", 0.0)))
        item = {**candidate, "metadata": dict(candidate.get("metadata", {}))}
        item["score"] = float(0.75 * overlap + 0.25 * original)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[: max(0, top_k)]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance for relevance plus diversity."""
    if top_k <= 0 or not candidates:
        return []

    embeddings = [
        candidate.get("embedding") or hashing_embedding(candidate.get("content", ""))
        for candidate in candidates
    ]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine_similarity(query_embedding, embeddings[idx])
            diversity_penalty = max(
                (cosine_similarity(embeddings[idx], embeddings[sel]) for sel in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = {**candidates[idx], "metadata": dict(candidates[idx].get("metadata", {}))}
        item["score"] = float(cosine_similarity(query_embedding, embeddings[idx]))
        results.append(item)
    return results


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion for merging multiple ranked result lists."""
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    results: list[dict] = []
    for content, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[: max(0, top_k)]:
        item = {**content_map[content], "metadata": dict(content_map[content].get("metadata", {}))}
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(hashing_embedding(query), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    print(rerank("hinh phat ma tuy", dummy, top_k=2))
