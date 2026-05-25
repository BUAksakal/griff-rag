"""Text chunker for scraped GRIFF documents.

Reads JSON files produced by :mod:`src.ingestion.scraper`, splits each
document's content into overlapping character-level chunks, and writes
all chunks to a single ``data/chunks/chunks.json`` file.

Each chunk retains the source document's metadata (URL, category,
language) and is annotated with its positional index within the document.

Usage::

    python -m src.ingestion.chunker
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
CHUNKS_DIR = Path("data/chunks")
CHUNKS_FILE = CHUNKS_DIR / "chunks.json"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks of *chunk_size* characters.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.  The last chunk may be shorter than
        *chunk_size*.

    Raises:
        ValueError: If *overlap* is not smaller than *chunk_size*.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step

    return chunks


def build_chunks(
    raw_dir: Path = RAW_DIR,
    output_file: Path = CHUNKS_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Read scraped JSON documents, chunk them, and write the result.

    Args:
        raw_dir: Directory containing the per-page JSON files from the
            scraper.
        output_file: Path for the combined chunks JSON output.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        The full list of chunk dictionaries that was written to disk.
    """
    raw_dir = Path(raw_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s — nothing to chunk.", raw_dir)
        return []

    all_chunks: list[dict[str, Any]] = []
    total_docs = 0

    for filepath in json_files:
        try:
            doc = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("Skipping unreadable file: %s", filepath)
            continue

        content: str = doc.get("content", "")
        if not content:
            logger.warning("Empty content in %s — skipping.", filepath.name)
            continue

        text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        total_chunks_for_doc = len(text_chunks)

        for idx, chunk_content in enumerate(text_chunks):
            chunk_record: dict[str, Any] = {
                "text": chunk_content,
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "language": doc.get("language", ""),
                "chunk_index": idx,
                "total_chunks": total_chunks_for_doc,
            }
            all_chunks.append(chunk_record)

        total_docs += 1
        logger.info(
            "Chunked %s → %d chunks", filepath.name, total_chunks_for_doc
        )

    # Persist all chunks.
    output_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Chunking complete — %d documents processed, %d chunks created.",
        total_docs,
        len(all_chunks),
    )

    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = build_chunks()

    # Print summary statistics.
    print("\n" + "=" * 50)
    print("CHUNKING SUMMARY")
    print("=" * 50)

    # Count unique source documents from the chunks themselves.
    unique_urls = {c["url"] for c in chunks}
    print(f"  Documents processed : {len(unique_urls)}")
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  Chunk size          : {DEFAULT_CHUNK_SIZE} chars")
    print(f"  Overlap             : {DEFAULT_OVERLAP} chars")
    print(f"  Output file         : {CHUNKS_FILE}")
    print("=" * 50)
