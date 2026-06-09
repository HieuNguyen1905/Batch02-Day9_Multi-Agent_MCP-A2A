"""Task 9 - supervisor + 3-worker retrieval pipeline.

Supervisor:
    - dispatches three retrieval workers concurrently
    - fuses dense + sparse results
    - reranks the hybrid candidates
    - falls back to PageIndex/vectorless results when hybrid confidence is low

Workers:
    1. semantic worker: neural/vector similarity
    2. lexical worker: BM25 keyword search
    3. pageindex worker: vectorless structural fallback
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


@dataclass(slots=True)
class WorkerResult:
    """Result returned by one retrieval worker."""

    name: str
    results: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def _run_worker(name: str, query: str, top_k: int) -> WorkerResult:
    """Run a retrieval worker without letting one failure kill the supervisor."""
    worker_map = {
        "semantic": semantic_search,
        "lexical": lexical_search,
        "pageindex": pageindex_search,
    }
    try:
        results = await asyncio.to_thread(worker_map[name], query, top_k)
    except Exception as exc:
        return WorkerResult(name=name, error=str(exc))

    tagged_results: list[dict] = []
    for item in results:
        tagged = {**item, "metadata": dict(item.get("metadata", {}))}
        tagged["worker"] = name
        tagged["source"] = name
        tagged_results.append(tagged)
    return WorkerResult(name=name, results=tagged_results)


async def retrieve_async(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Supervisor: run three workers, fuse results, rerank and fallback if needed."""
    if top_k <= 0:
        return []

    worker_top_k = top_k * 2
    semantic_task = _run_worker("semantic", query, worker_top_k)
    lexical_task = _run_worker("lexical", query, worker_top_k)
    pageindex_task = _run_worker("pageindex", query, top_k)

    semantic_result, lexical_result, pageindex_result = await asyncio.gather(
        semantic_task,
        lexical_task,
        pageindex_task,
    )

    dense_results = semantic_result.results
    sparse_results = lexical_result.results
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)

    for item in merged:
        item["source"] = "supervisor_hybrid"
        item["workers"] = ["semantic", "lexical"]

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "supervisor_hybrid"
            item["workers"] = ["semantic", "lexical"]
    else:
        final_results = merged[:top_k]

    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        return pageindex_result.results[:top_k]

    return final_results[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Sync wrapper for the supervisor pipeline.

    Existing tasks call `retrieve(...)` synchronously, so this wrapper preserves
    that API while the actual supervisor uses asyncio internally.
    """
    coro = retrieve_async(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


if __name__ == "__main__":
    for result in retrieve("hinh phat ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
