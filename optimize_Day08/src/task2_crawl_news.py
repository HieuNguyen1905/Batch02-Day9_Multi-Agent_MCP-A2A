"""Task 2 - crawl news articles about Vietnamese artists and drug cases."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vov.vn/phap-luat/cong-an-tphcm-thong-tin-vu-ca-si-chi-dan-nguoi-mau-andrea-aybar-bi-bat-post1135460.vov",
    "https://nld.com.vn/phap-luat/bat-giam-dien-vien-huu-tin-vi-lien-quan-ma-tuy-20220617202258452.htm",
    "https://nld.com.vn/phap-luat/dien-vien-le-hang-bi-bat-vi-mua-ban-ma-tuy-20230423173501249.htm",
    "https://cand.com.vn/Vu-an-noi-tieng/He-qua-cua-loi-song-buong-tha-i467319/",
    "https://vnexpress.net/nha-thiet-ke-nguyen-cong-tri-bi-bat-vi-lien-quan-ma-tuy-4917929.html",
]


def setup_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _html_to_text(html: str) -> str:
    title = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    title_text = re.sub(r"\s+", " ", title.group(1)).strip() if title else "Unknown"
    body = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return f"# {title_text}\n\n{body[:4000]}"


async def crawl_article(url: str) -> dict:
    """Crawl one article and return metadata plus markdown-like content."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        content = result.markdown or _html_to_text(getattr(result, "html", ""))
        title = result.metadata.get("title", "Unknown") if result.metadata else "Unknown"
    except Exception:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        content = _html_to_text(response.text)
        title = content.splitlines()[0].lstrip("# ").strip() or "Unknown"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> list[Path]:
    """Crawl every URL and save one JSON file per article."""
    setup_directory()
    saved: list[Path] = []
    for i, url in enumerate(ARTICLE_URLS, 1):
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{i:02d}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(filepath)
    return saved


if __name__ == "__main__":
    for path in asyncio.run(crawl_all()):
        print(f"Saved {path}")
