"""Task 8 - PageIndex-style vectorless fallback search.

This module exposes the same interface as a PageIndex integration but uses the
local Markdown corpus when PAGEINDEX_API_KEY is not configured.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .local_rag_utils import lexical_overlap_score, load_chunks


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> list[dict]:
    """Return local document metadata as the offline upload manifest."""
    manifest: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        manifest.append(
            {
                "filename": md_file.name,
                "type": md_file.parent.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
            }
        )
    return manifest


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless fallback based on structural/keyword matching."""
    if top_k <= 0:
        return []

    results: list[dict] = []
    for chunk in load_chunks():
        metadata = dict(chunk.get("metadata", {}))
        content = chunk.get("content", "")
        title_boost = lexical_overlap_score(query, metadata.get("source", "")) * 0.15
        body_score = lexical_overlap_score(query, content)
        score = float(min(1.0, body_score + title_boost))
        if score <= 0:
            continue
        results.append(
            {
                "content": content,
                "score": score,
                "metadata": metadata,
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in pageindex_search("ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
