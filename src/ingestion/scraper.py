"""Web scraper for official German government and immigration websites.

Fetches HTML content from curated source URLs, strips navigation and
boilerplate elements, and persists the extracted text as JSON files in
``data/raw/``.  Each JSON record contains the page URL, title, cleaned
body text, source category, language code, and scrape timestamp.

Usage::

    python -m src.ingestion.scraper
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")

USER_AGENT = (
    "GRIFF-Bot/1.0 "
    "(+https://github.com/griff-project; educational RAG project)"
)

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # exponential backoff multiplier

# Tags whose content is irrelevant to the main text.
STRIP_TAGS: list[str] = [
    "nav",
    "footer",
    "header",
    "script",
    "style",
    "noscript",
    "aside",
    "form",
    "iframe",
    "svg",
]

# ---------------------------------------------------------------------------
# Source catalogue
# ---------------------------------------------------------------------------

SOURCES: list[dict[str, str]] = [
    {
        "url": "https://www.make-it-in-germany.com/en/visa/kinds-of-visa",
        "category": "visa",
        "language": "en",
    },
    {
        "url": (
            "https://www.make-it-in-germany.com/en/visa/"
            "kinds-of-visa/work/eu-blue-card"
        ),
        "category": "blue_card",
        "language": "en",
    },
    {
        "url": (
            "https://www.make-it-in-germany.com/en/living-in-germany/"
            "housing/looking-for-a-house"
        ),
        "category": "housing",
        "language": "en",
    },
    {
        "url": (
            "https://www.make-it-in-germany.com/en/working-in-germany/"
            "social-benefits/health-insurance"
        ),
        "category": "health_insurance",
        "language": "en",
    },
    {
        "url": (
            "https://www.bamf.de/EN/Themen/MigrationAufenthalt/"
            "ZuwsijandererDrittstaaten/Migrathek/migrathek-node.html"
        ),
        "category": "migration",
        "language": "en",
    },
    {
        "url": "https://www.berlin.de/willkommen/en/arrival/registration/",
        "category": "anmeldung",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/moving-to-berlin",
        "category": "relocation",
        "language": "en",
    },
    {
        "url": (
            "https://allaboutberlin.com/guides/anmeldung-in-english-berlin"
        ),
        "category": "anmeldung",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/german-health-insurance",
        "category": "health_insurance",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/first-days-in-germany",
        "category": "relocation",
        "language": "en",
    },
    {
        "url": (
            "https://www.make-it-in-germany.com/en/visa/"
            "living-permanently-in-germany/settlement-permit"
        ),
        "category": "settlement",
        "language": "en",
    },
    {
        "url": (
            "https://www.make-it-in-germany.com/en/working-in-germany/"
            "tax/tax-system"
        ),
        "category": "tax",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/german-tax-system",
        "category": "tax",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/finanzamt-tax-number",
        "category": "tax",
        "language": "en",
    },
    {
        "url": "https://allaboutberlin.com/guides/krankenkasse",
        "category": "health_insurance",
        "language": "en",
    },
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _url_to_filename(url: str) -> str:
    """Derive a stable, filesystem-safe filename from a URL.

    Uses the first 12 hex characters of the URL's SHA-256 hash so that
    filenames stay short and unique even for very long URLs.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"{url_hash}.json"


def _fetch_page(url: str) -> requests.Response:
    """Fetch a single URL with retry logic and exponential backoff.

    Args:
        url: Absolute URL to fetch.

    Returns:
        The :class:`requests.Response` object on success.

    Raises:
        requests.RequestException: After *MAX_RETRIES* failed attempts.
    """
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            wait = BACKOFF_FACTOR ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %ds",
                attempt,
                MAX_RETRIES,
                url,
                exc,
                wait,
            )
            time.sleep(wait)

    # All retries exhausted.
    raise last_exc  # type: ignore[misc]


def _extract_content(html: str) -> tuple[str, str]:
    """Extract the page title and cleaned body text from raw HTML.

    Removes navigation, footer, script, style, and other non-content
    elements before collapsing whitespace.

    Args:
        html: Raw HTML string.

    Returns:
        A ``(title, body_text)`` tuple.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements.
    for tag_name in STRIP_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # Try to find the main content area; fall back to <body> or the
    # whole document.
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        main = soup

    # Collapse excessive whitespace while preserving paragraph breaks.
    lines = (line.strip() for line in main.get_text("\n").splitlines())
    body_text = "\n".join(line for line in lines if line)

    return title, body_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scrape_sources(
    sources: list[dict[str, str]] | None = None,
    output_dir: Path = RAW_DIR,
) -> list[dict[str, Any]]:
    """Scrape all *sources* and persist each page as a JSON file.

    Args:
        sources: List of source dicts with ``url``, ``category``, and
            ``language`` keys.  Defaults to :data:`SOURCES`.
        output_dir: Directory where JSON files are written.

    Returns:
        A list of result dictionaries (one per successfully scraped page).
    """
    if sources is None:
        sources = SOURCES

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failed: int = 0

    for source in tqdm(sources, desc="Scraping pages", unit="page"):
        url = source["url"]
        try:
            response = _fetch_page(url)
            title, content = _extract_content(response.text)

            record: dict[str, Any] = {
                "url": url,
                "title": title,
                "content": content,
                "category": source["category"],
                "language": source["language"],
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }

            filepath = output_dir / _url_to_filename(url)
            filepath.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(record)
            logger.info("✓ Scraped %s (%d chars)", url, len(content))

        except Exception:
            failed += 1
            logger.exception("✗ Failed to scrape %s", url)

    logger.info(
        "Scraping complete — %d succeeded, %d failed out of %d sources",
        len(results),
        failed,
        len(sources),
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrape_sources()
