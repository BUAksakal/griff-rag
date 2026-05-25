"""Index builder for GRIFF.

Reads pre-processed chunks from ``data/chunks/chunks.json``, computes
dense embeddings via :mod:`src.retrieval.embedder`, and persists two
complementary indexes:

* **ChromaDB** — dense vector store (``griff_docs`` collection).
* **BM25Okapi** — keyword-based sparse index (pickled to disk).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.retrieval.embedder import embed_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

load_dotenv()

CHUNKS_PATH: str = os.path.join("data", "chunks", "chunks.json")
"""Path to the chunked documents produced by the ingestion pipeline."""

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
"""Directory where the ChromaDB persistent storage is kept."""

BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
"""File path for the pickled BM25Okapi index."""

BM25_TEXTS_PATH: str = BM25_INDEX_PATH.replace(".pkl", "_texts.pkl")
"""Companion file that stores the raw chunk texts in the same order as the
BM25 index, needed during retrieval to map BM25 scores back to documents."""

COLLECTION_NAME: str = "griff_docs"
"""Name of the ChromaDB collection."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_chunks(path: str) -> list[dict]:
    """Read and return the chunks JSON file.

    Each chunk dict is expected to have at least:

    * ``text`` — the chunk body
    * ``url``  — source URL
    * ``category`` — topical category

    Args:
        path: Filesystem path to ``chunks.json``.

    Returns:
        A list of chunk dictionaries.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
    """
    chunks_file = Path(path)
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_file.resolve()}'. "
            "Run the ingestion pipeline first: python -m src.ingestion.chunker"
        )

    with open(chunks_file, encoding="utf-8") as fh:
        chunks: list[dict] = json.load(fh)

    logger.info("Loaded %d chunks from '%s'.", len(chunks), path)
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB index
# ---------------------------------------------------------------------------


def _build_chroma_index(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Populate a ChromaDB persistent collection with embedded chunks.

    If a collection with the same name already exists it is deleted and
    recreated to guarantee a clean state.

    Args:
        chunks: List of chunk dicts (must contain ``text``, ``url``,
            ``category``).
        embeddings: Pre-computed embedding vectors aligned with *chunks*.
    """
    persist_dir = Path(CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection to ensure idempotency.
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        logger.warning(
            "Collection '%s' already exists — deleting and recreating.",
            COLLECTION_NAME,
        )
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Adding %d chunks to ChromaDB collection '%s' …", len(chunks), COLLECTION_NAME)

    # ChromaDB has a per-call size limit; we batch in groups of 5 000.
    batch_size = 5_000
    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="ChromaDB upsert",
        unit="batch",
    ):
        end = min(start + batch_size, len(chunks))
        batch_ids = [f"chunk_{i}" for i in range(start, end)]
        batch_docs = [c["text"] for c in chunks[start:end]]
        batch_embeds = embeddings[start:end]
        batch_meta = [
            {
                "url": c.get("url", ""),
                "category": c.get("category", ""),
            }
            for c in chunks[start:end]
        ]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_meta,
        )

    logger.info(
        "ChromaDB collection '%s' now contains %d documents.",
        COLLECTION_NAME,
        collection.count(),
    )


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def _build_bm25_index(chunks: list[dict]) -> None:
    """Build a BM25Okapi index and persist it to disk.

    Two files are written:

    1. The BM25 index itself (``BM25_INDEX_PATH``).
    2. A companion list of raw chunk texts (``BM25_TEXTS_PATH``) so the
       retriever can map scores back to documents.

    Args:
        chunks: List of chunk dicts containing at least a ``text`` field.
    """
    index_path = Path(BM25_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    tokenized = [text.lower().split() for text in tqdm(texts, desc="Tokenising for BM25")]

    logger.info("Fitting BM25Okapi on %d documents …", len(tokenized))
    bm25 = BM25Okapi(tokenized)

    with open(index_path, "wb") as fh:
        pickle.dump(bm25, fh)
    logger.info("BM25 index saved to '%s'.", index_path)

    texts_path = Path(BM25_TEXTS_PATH)
    with open(texts_path, "wb") as fh:
        pickle.dump(texts, fh)
    logger.info("BM25 chunk texts saved to '%s'.", texts_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_indexes() -> None:
    """Run the full indexing pipeline.

    1. Load chunks from disk.
    2. Compute dense embeddings with ``bge-m3``.
    3. Populate the ChromaDB collection.
    4. Build and persist the BM25 index.
    """
    chunks = _load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]

    logger.info("Computing embeddings for %d chunks …", len(texts))
    embeddings = embed_documents(texts)

    _build_chroma_index(chunks, embeddings)
    _build_bm25_index(chunks)

    logger.info("All indexes built successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    build_indexes()
