"""Embedding module for GRIFF.

Uses BAAI/bge-m3 via sentence-transformers to produce multilingual dense
embeddings.  The model is loaded lazily on first use and reused as a
singleton throughout the application lifetime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME: str = "BAAI/bge-m3"
"""HuggingFace model identifier for the multilingual embedding model."""

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_embedder_instance: SentenceTransformer | None = None


def _detect_device() -> str:
    """Return the best available torch device string.

    Returns ``"cuda"`` when a CUDA-capable GPU is present, otherwise
    ``"cpu"``.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Detected compute device: %s", device)
    return device


def get_embedder() -> SentenceTransformer:
    """Return the shared :class:`SentenceTransformer` instance.

    The model is loaded on the first call (lazy initialisation).  All
    subsequent calls return the same object.

    Returns:
        SentenceTransformer: The loaded ``BAAI/bge-m3`` model.
    """
    global _embedder_instance  # noqa: PLW0603

    if _embedder_instance is None:
        device = _detect_device()
        logger.info("Loading embedding model '%s' on device '%s' …", MODEL_NAME, device)
        _embedder_instance = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("Embedding model loaded successfully.")

    return _embedder_instance


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def embed_query(text: str) -> list[float]:
    """Embed a single query string and return its dense vector.

    Args:
        text: The query text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError: If *text* is empty or ``None``.
    """
    if not text:
        raise ValueError("Query text must be a non-empty string.")

    model = get_embedder()
    logger.debug("Embedding query (%d chars).", len(text))
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of document texts in batches.

    Args:
        texts: Document strings to embed.
        batch_size: Number of texts per encoding batch.  Defaults to 32.

    Returns:
        A list of embedding vectors (each a list of floats), in the same
        order as the input texts.

    Raises:
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("The texts list must not be empty.")

    model = get_embedder()
    logger.info(
        "Embedding %d documents (batch_size=%d) …",
        len(texts),
        batch_size,
    )

    all_embeddings: list[list[float]] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        vectors = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(vectors.tolist())

    logger.info("Finished embedding %d documents.", len(all_embeddings))
    return all_embeddings
