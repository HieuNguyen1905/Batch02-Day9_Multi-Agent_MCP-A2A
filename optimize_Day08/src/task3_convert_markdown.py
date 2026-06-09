"""Task 3 - convert landing files into Markdown under data/standardized."""

from __future__ import annotations

import json
from pathlib import Path


try:
    from markitdown import MarkItDown
except Exception:  # MarkItDown is optional for the local test path.
    MarkItDown = None


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _fallback_legal_markdown(filepath: Path) -> str:
    stem = filepath.stem.replace("-", " ")
    return (
        f"# {stem}\n\n"
        f"**Van ban goc:** `{filepath.as_posix()}`\n"
        "**Loai tai lieu:** legal\n\n"
        "Tai lieu phap luat ve phong, chong ma tuy duoc luu o dang goc PDF/DOC. "
        "Neu moi truong khong cai MarkItDown, ban markdown nay dong vai tro "
        "metadata de pipeline van co the index, search va cite nguon tai lieu. "
        "Noi dung chi tiet can doi chieu voi file goc trong data/landing/legal/."
    )


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return []

    converter = MarkItDown() if MarkItDown else None
    saved: list[Path] = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        if converter:
            try:
                result = converter.convert(str(filepath))
                content = result.text_content
            except Exception:
                content = _fallback_legal_markdown(filepath)
        else:
            content = _fallback_legal_markdown(filepath)
        output_path.write_text(content, encoding="utf-8")
        saved.append(output_path)
    return saved


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not news_dir.exists():
        return []

    saved: list[Path] = []
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"
        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('source_name', 'News')}\n"
            f"**URL:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            "**Type:** news\n\n"
        )
        output_path.write_text(header + data.get("content_markdown", ""), encoding="utf-8")
        saved.append(output_path)
    return saved


def convert_all() -> list[Path]:
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    for path in convert_all():
        print(f"Saved {path}")
