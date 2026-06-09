"""Small local RAG helpers used by the individual tasks.

The lab README suggests production services such as Weaviate, PageIndex and
hosted rerankers. For the individual automated tests we keep a deterministic
local implementation so the pipeline runs without API keys or Docker.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
VECTOR_INDEX_PATH = ROOT_DIR / "data" / "vectorstore" / "local_index.json"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_DIM = 384
REAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REAL_EMBEDDING_DIM = 384


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "la", "va", "ve",
    "cua", "cho", "cac", "co", "duoc", "trong", "theo", "mot", "nhung",
    "nay", "do", "khi", "tu", "tai", "den", "voi", "nguon",
}


def normalize_text(text: str) -> str:
    """Lowercase and remove Vietnamese accents for robust keyword matching."""
    text = text.lower().replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize_text(text))
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]


def load_markdown_documents(base_dir: Path = STANDARDIZED_DIR) -> list[dict]:
    documents: list[dict] = []
    if not base_dir.exists():
        return documents

    for md_file in sorted(base_dir.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        rel_path = md_file.relative_to(base_dir)
        doc_type = "legal" if "legal" in rel_path.parts else "news"
        metadata = {
            "source": md_file.name,
            "path": str(rel_path).replace("\\", "/"),
            "type": doc_type,
        }
        url = _extract_metadata_line(content, "**URL:**")
        if url:
            metadata["url"] = url

        documents.append({"content": content, "metadata": metadata})

    return documents


def _extract_metadata_line(content: str, prefix: str) -> str:
    for line in content.splitlines()[:12]:
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_block(para, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_long_block(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step
    return [c for c in chunks if c]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        for idx, text in enumerate(chunk_text(doc.get("content", ""), chunk_size, chunk_overlap)):
            metadata = dict(doc.get("metadata", {}))
            metadata["chunk_index"] = idx
            chunks.append({"content": text, "metadata": metadata})
    return chunks


def load_chunks(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    return chunk_documents(load_markdown_documents(), chunk_size, chunk_overlap)


def load_vector_index(path: Path = VECTOR_INDEX_PATH) -> list[dict]:
    """Load the persisted vector index created by Task 4."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_vector_index(chunks: list[dict], path: Path = VECTOR_INDEX_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _prepare_transformer_runtime() -> None:
    """Keep text-only embedding imports stable in mixed Anaconda installs."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        import transformers.utils.import_utils as import_utils

        # This project only needs text models. A mismatched torchvision install
        # can otherwise break transformers while importing unrelated vision code.
        import_utils._torchvision_available = False
        import_utils._torchvision_version = "0.0"
    except Exception:
        pass


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = REAL_EMBEDDING_MODEL):
    """Load the real multilingual sentence embedding model once per process."""
    _prepare_transformer_runtime()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def neural_embeddings(texts: list[str], model_name: str = REAL_EMBEDDING_MODEL) -> list[list[float]]:
    """Encode texts with a real SentenceTransformers model."""
    if not texts:
        return []
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [embedding.tolist() for embedding in embeddings]


def hashing_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dim
    for token, count in counts.items():
        vector[hash(token) % dim] += float(count)

    norm = math.sqrt(sum(v * v for v in vector))
    if norm:
        vector = [v / norm for v in vector]
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


def lexical_overlap_score(query: str, content: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    c_counts = Counter(tokenize(content))
    hits = sum(1 for tok in set(q_tokens) if c_counts.get(tok, 0) > 0)
    density = sum(min(c_counts.get(tok, 0), 3) for tok in set(q_tokens))
    return min(1.0, (hits / len(set(q_tokens))) * 0.75 + (density / (len(q_tokens) * 3)) * 0.25)


class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.avgdl = (
            sum(len(doc) for doc in tokenized_corpus) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.doc_freq: Counter[str] = Counter()
        for doc in tokenized_corpus:
            self.doc_freq.update(set(doc))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc in self.tokenized_corpus:
            tf = Counter(doc)
            doc_len = len(doc) or 1
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def source_label(item: dict, fallback: str = "local corpus") -> str:
    metadata = item.get("metadata", {})
    return metadata.get("source") or metadata.get("path") or fallback
