"""Task 1 - collect legal documents about drugs and prohibited substances."""

from __future__ import annotations

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-73-2021-qh14-33659?format=pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
    },
    {
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-105-2021-nd-cp-34944-37821?format=pdf",
        "filename": "nghi-dinh-105-2021.pdf",
    },
    {
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-57-2022-nd-cp-37734-41623?format=pdf",
        "filename": "nghi-dinh-57-2022.pdf",
    },
]


def setup_directory() -> Path:
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_file(url: str, filename: str, timeout: int = 30) -> Path:
    """Download one legal document into the landing/legal folder."""
    setup_directory()
    output_path = DATA_DIR / filename
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def download_all() -> list[Path]:
    """Download the curated legal documents used by this lab."""
    downloaded: list[Path] = []
    for doc in LEGAL_DOCUMENTS:
        downloaded.append(download_file(doc["url"], doc["filename"]))
    return downloaded


if __name__ == "__main__":
    for path in download_all():
        print(f"Saved {path} ({path.stat().st_size} bytes)")
