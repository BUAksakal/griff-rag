"""Hybrid retrieval pipeline for GRIFF.

Provides three search strategies:

1. **Naive dense** — ChromaDB cosine similarity only.
2. **BM25** — keyword-based sparse retrieval only.
3. **Hybrid** — dense + BM25 fused via Reciprocal Rank Fusion (RRF)
   and reranked with ``BAAI/bge-reranker-base``.

All heavy models (embedding model, BM25 index, reranker) are loaded lazily
on first use and cached as module-level singletons.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker

from src.retrieval.embedder import embed_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

load_dotenv()

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
BM25_TEXTS_PATH: str = BM25_INDEX_PATH.replace(".pkl", "_texts.pkl")

COLLECTION_NAME: str = "griff_docs"
RERANKER_MODEL: str = "BAAI/bge-reranker-base"
RRF_K: int = 60
"""Constant *k* in the RRF formula: ``score = Σ 1 / (k + rank)``."""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

ResultDict = dict[str, str | float]
"""Type alias for a single search result.

Keys: ``text``, ``url``, ``category``, ``score``.
"""


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------


class Retriever:
    """Hybrid dense + sparse retriever with optional reranking.

    All resources are loaded lazily the first time a search method is
    called.

    Attributes:
        _chroma_collection: ChromaDB collection handle.
        _bm25: BM25Okapi index.
        _bm25_texts: Raw chunk texts aligned with the BM25 index.
        _reranker: Cross-encoder reranker model.
    """

    def __init__(self) -> None:
        self._chroma_collection: chromadb.Collection | None = None
        self._bm25 = None  # rank_bm25.BM25Okapi
        self._bm25_texts: list[str] | None = None
        self._bm25_metadata: list[dict] | None = None
        self._reranker: FlagReranker | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_chroma_collection(self) -> chromadb.Collection:
        """Return the ChromaDB collection, loading it on first access."""
        if self._chroma_collection is None:
            logger.info("Connecting to ChromaDB at '%s' …", CHROMA_PERSIST_DIR)
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._chroma_collection = client.get_collection(name=COLLECTION_NAME)
            logger.info(
                "ChromaDB collection '%s' loaded (%d documents).",
                COLLECTION_NAME,
                self._chroma_collection.count(),
            )
        return self._chroma_collection

    def _get_bm25(self):  # noqa: ANN202  — BM25Okapi
        """Return the BM25 index, loading it on first access."""
        if self._bm25 is None:
            index_path = Path(BM25_INDEX_PATH)
            texts_path = Path(BM25_TEXTS_PATH)

            if not index_path.exists():
                raise FileNotFoundError(
                    f"BM25 index not found at '{index_path.resolve()}'. "
                    "Run: python -m src.retrieval.indexer"
                )
            if not texts_path.exists():
                raise FileNotFoundError(
                    f"BM25 texts file not found at '{texts_path.resolve()}'. "
                    "Run: python -m src.retrieval.indexer"
                )

            logger.info("Loading BM25 index from '%s' …", index_path)
            with open(index_path, "rb") as fh:
                self._bm25 = pickle.load(fh)  # noqa: S301

            with open(texts_path, "rb") as fh:
                self._bm25_texts = pickle.load(fh)  # noqa: S301

            logger.info("BM25 index loaded (%d documents).", len(self._bm25_texts))

            # Pre-fetch metadata from ChromaDB so BM25 results carry url/category.
            collection = self._get_chroma_collection()
            all_docs = collection.get(
                ids=[f"chunk_{i}" for i in range(len(self._bm25_texts))],
                include=["metadatas"],
            )
            self._bm25_metadata = all_docs["metadatas"]

        return self._bm25

    def _get_reranker(self) -> FlagReranker:
        """Return the reranker model, loading it on first access."""
        if self._reranker is None:
            logger.info("Loading reranker model '%s' …", RERANKER_MODEL)
            self._reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
            logger.info("Reranker model loaded.")
        return self._reranker

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def naive_dense_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[ResultDict]:
        """Dense-only retrieval via ChromaDB cosine similarity.

        Args:
            query: User query string.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of result dicts with keys ``text``, ``url``,
            ``category``, ``score``.
        """
        t0 = time.perf_counter()

        query_vec = embed_query(query)
        collection = self._get_chroma_collection()

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output: list[ResultDict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "text": doc,
                    "url": meta.get("url", ""),
                    "category": meta.get("category", ""),
                    "score": round(1 - dist, 4),  # cosine distance → similarity
                }
            )

        elapsed = time.perf_counter() - t0
        logger.info("naive_dense_search returned %d results in %.3fs.", len(output), elapsed)
        return output

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[ResultDict]:
        """Keyword-based BM25 retrieval.

        Args:
            query: User query string.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of result dicts.
        """
        t0 = time.perf_counter()

        bm25 = self._get_bm25()
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get the indices of the top-k scores.
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        output: list[ResultDict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metadata[idx] if self._bm25_metadata else {}
            output.append(
                {
                    "text": self._bm25_texts[idx],
                    "url": meta.get("url", ""),
                    "category": meta.get("category", ""),
                    "score": round(float(scores[idx]), 4),
                }
            )

        elapsed = time.perf_counter() - t0
        logger.info("bm25_search returned %d results in %.3fs.", len(output), elapsed)
        return output

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[ResultDict]:
        """Hybrid retrieval: dense + BM25 fused with RRF, then reranked.

        Steps:

        1. Retrieve candidates from both dense and BM25 channels.
        2. Merge with Reciprocal Rank Fusion (``k=60``).
        3. Rerank the top candidates using ``bge-reranker-base``.
        4. Return the final top-*k* results.

        Args:
            query: User query string.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of result dicts.
        """
        t0 = time.perf_counter()

        # Fetch a generous candidate pool from each channel.
        candidate_k = max(top_k * 3, 30)
        dense_results = self.naive_dense_search(query, top_k=candidate_k)
        bm25_results = self.bm25_search(query, top_k=candidate_k)

        # ----- Reciprocal Rank Fusion -----
        rrf_scores: dict[str, float] = {}
        text_to_result: dict[str, ResultDict] = {}

        for rank, res in enumerate(dense_results, start=1):
            key = res["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            text_to_result[key] = res

        for rank, res in enumerate(bm25_results, start=1):
            key = res["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            text_to_result[key] = res

        # Sort by fused score descending.
        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

        # Take a wider window for reranking.
        rerank_pool_size = min(len(fused), max(top_k * 2, 20))
        candidates = fused[:rerank_pool_size]

        # ----- Reranking -----
        reranker = self._get_reranker()
        pairs = [[query, text] for text, _ in candidates]
        rerank_scores = reranker.compute_score(pairs)

        # compute_score may return a single float when len(pairs) == 1.
        if isinstance(rerank_scores, (float, int)):
            rerank_scores = [rerank_scores]

        scored: list[tuple[str, float]] = list(
            zip([text for text, _ in candidates], rerank_scores)
        )
        scored.sort(key=lambda x: x[1], reverse=True)

        output: list[ResultDict] = []
        for text, score in scored[:top_k]:
            res = text_to_result[text]
            output.append(
                {
                    "text": res["text"],
                    "url": res["url"],
                    "category": res["category"],
                    "score": round(float(score), 4),
                }
            )

        elapsed = time.perf_counter() - t0
        logger.info("hybrid_search returned %d results in %.3fs.", len(output), elapsed)
        return output


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_retriever_instance: Retriever | None = None


def get_retriever() -> Retriever:
    """Return the shared :class:`Retriever` singleton.

    The instance is created lazily on the first call.

    Returns:
        Retriever: The application-wide retriever.
    """
    global _retriever_instance  # noqa: PLW0603

    if _retriever_instance is None:
        logger.info("Initialising Retriever singleton …")
        _retriever_instance = Retriever()

    return _retriever_instance
