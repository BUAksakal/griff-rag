"""
GRIFF Evaluation Pipeline — Compare retrieval methods.

Runs 20 gold-standard test questions (Turkish, English, German) through
three retrieval strategies and measures context precision, faithfulness,
answer relevancy, keyword coverage, and latency.

Usage:
    python -m evaluate.evaluator

The output is a formatted comparison table suitable for README / reports.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabulate import tabulate

from src.retrieval.retriever import bm25_search, hybrid_search, naive_dense_search
from src.generation.generator import generate_answer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate.evaluator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
TEST_QUESTIONS_PATH = _THIS_DIR / "test_questions.json"
TOP_K = 5
API_SLEEP_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class QuestionResult:
    """Metrics for a single question evaluated with one retrieval method."""

    question_id: str
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    keyword_score: float = 0.0
    latency: float = 0.0


@dataclass
class MethodResult:
    """Aggregated metrics for one retrieval method across all questions."""

    method_name: str
    question_results: list[QuestionResult] = field(default_factory=list)

    # -- aggregated averages --------------------------------------------------
    @property
    def avg_context_precision(self) -> float:
        """Return the mean context precision across all evaluated questions."""
        if not self.question_results:
            return 0.0
        return sum(q.context_precision for q in self.question_results) / len(
            self.question_results
        )

    @property
    def avg_faithfulness(self) -> float:
        """Return the mean faithfulness score across all evaluated questions."""
        if not self.question_results:
            return 0.0
        return sum(q.faithfulness for q in self.question_results) / len(
            self.question_results
        )

    @property
    def avg_answer_relevancy(self) -> float:
        """Return the mean answer relevancy across all evaluated questions."""
        if not self.question_results:
            return 0.0
        return sum(q.answer_relevancy for q in self.question_results) / len(
            self.question_results
        )

    @property
    def avg_keyword_score(self) -> float:
        """Return the mean keyword score across all evaluated questions."""
        if not self.question_results:
            return 0.0
        return sum(q.keyword_score for q in self.question_results) / len(
            self.question_results
        )

    @property
    def avg_latency(self) -> float:
        """Return the mean latency (seconds) across all evaluated questions."""
        if not self.question_results:
            return 0.0
        return sum(q.latency for q in self.question_results) / len(
            self.question_results
        )


# ---------------------------------------------------------------------------
# Test-question loader
# ---------------------------------------------------------------------------
def load_test_questions(path: Path | str = TEST_QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load gold-standard test questions from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file containing the test questions.

    Returns
    -------
    list[dict[str, Any]]
        A list of question dictionaries, each containing ``id``,
        ``question``, ``language``, ``category``, ``expected_keywords``,
        and ``difficulty``.

    Raises
    ------
    FileNotFoundError
        If the test-questions file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {path}. "
            "Please create it first (see evaluate/test_questions.json)."
        )
    with open(path, encoding="utf-8") as fh:
        questions: list[dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d test questions from %s", len(questions), path.name)
    return questions


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_context_precision(
    chunks: list[dict[str, Any]],
    expected_category: str,
) -> float:
    """Compute the fraction of retrieved chunks whose category matches.

    A retrieved chunk is considered *relevant* when its ``category`` metadata
    field matches the ``expected_category`` of the question.

    Parameters
    ----------
    chunks:
        List of retrieved chunk dictionaries.  Each chunk is expected to
        carry a ``metadata`` dict with a ``category`` key.
    expected_category:
        The gold-standard category for the current question.

    Returns
    -------
    float
        Precision score in [0.0, 1.0].
    """
    if not chunks:
        return 0.0
    relevant = sum(
        1
        for chunk in chunks
        if chunk.get("metadata", {}).get("category", "").lower()
        == expected_category.lower()
    )
    return relevant / len(chunks)


def compute_faithfulness(answer: str, chunks: list[dict[str, Any]]) -> float:
    """Heuristic faithfulness: overlap between answer tokens and chunk tokens.

    Tokenises both the generated answer and the concatenated chunk texts,
    then measures what fraction of answer tokens appear in the chunk corpus.
    This is a lightweight proxy — a production system would use an
    NLI-based faithfulness classifier.

    Parameters
    ----------
    answer:
        The generated answer text.
    chunks:
        The retrieved chunk dictionaries (must contain a ``text`` key).

    Returns
    -------
    float
        Overlap score in [0.0, 1.0].
    """
    if not answer or not chunks:
        return 0.0

    def _tokenize(text: str) -> set[str]:
        """Lowercase and split into word-level tokens."""
        return set(re.findall(r"\w+", text.lower()))

    answer_tokens = _tokenize(answer)
    chunk_text = " ".join(chunk.get("text", "") for chunk in chunks)
    chunk_tokens = _tokenize(chunk_text)

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens & chunk_tokens
    return len(overlap) / len(answer_tokens)


def compute_answer_relevancy(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """Check whether the answer covers the expected keywords.

    A keyword is considered *present* if it appears as a case-insensitive
    substring of the answer.

    Parameters
    ----------
    answer:
        The generated answer text.
    expected_keywords:
        Gold-standard keywords that a good answer should mention.

    Returns
    -------
    float
        Fraction of expected keywords found in the answer, in [0.0, 1.0].
    """
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def compute_keyword_score(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """Fraction of expected keywords found in the answer.

    Identical to :func:`compute_answer_relevancy` but kept as a separate
    metric slot so the evaluation table can distinguish *keyword coverage*
    from a future, more sophisticated *answer relevancy* model.

    Parameters
    ----------
    answer:
        The generated answer text.
    expected_keywords:
        Gold-standard keywords that a good answer should mention.

    Returns
    -------
    float
        Fraction in [0.0, 1.0].
    """
    return compute_answer_relevancy(answer, expected_keywords)


# ---------------------------------------------------------------------------
# Single-question evaluator
# ---------------------------------------------------------------------------
def evaluate_question(
    question: dict[str, Any],
    search_fn: Any,
    *,
    top_k: int = TOP_K,
) -> QuestionResult:
    """Run a single question through retrieval → generation → scoring.

    Parameters
    ----------
    question:
        A test-question dictionary from the gold-standard set.
    search_fn:
        A callable ``(query: str, top_k: int) -> list[dict]`` that
        retrieves document chunks.
    top_k:
        Number of chunks to retrieve.

    Returns
    -------
    QuestionResult
        Computed metrics for this question.
    """
    qid: str = question["id"]
    query: str = question["question"]
    category: str = question["category"]
    expected_keywords: list[str] = question["expected_keywords"]

    start = time.perf_counter()

    # 1) Retrieve chunks ---------------------------------------------------
    try:
        chunks: list[dict[str, Any]] = search_fn(query, top_k=top_k)
    except Exception:
        logger.exception("Retrieval failed for %s", qid)
        chunks = []

    # 2) Generate answer ----------------------------------------------------
    answer = ""
    try:
        answer = generate_answer(query, chunks)
        # Rate-limit pause between API calls
        time.sleep(API_SLEEP_SECONDS)
    except Exception:
        logger.exception("Generation failed for %s", qid)

    elapsed = time.perf_counter() - start

    # 3) Compute metrics ----------------------------------------------------
    ctx_prec = compute_context_precision(chunks, category)
    faith = compute_faithfulness(answer, chunks)
    relevancy = compute_answer_relevancy(answer, expected_keywords)
    kw_score = compute_keyword_score(answer, expected_keywords)

    result = QuestionResult(
        question_id=qid,
        context_precision=ctx_prec,
        faithfulness=faith,
        answer_relevancy=relevancy,
        keyword_score=kw_score,
        latency=elapsed,
    )

    logger.info(
        "  %s | ctx=%.2f | faith=%.2f | rel=%.2f | kw=%.2f | %.2fs",
        qid,
        ctx_prec,
        faith,
        relevancy,
        kw_score,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------
METHODS: list[tuple[str, Any]] = [
    ("Naive Dense (baseline)", naive_dense_search),
    ("BM25 Only", bm25_search),
    ("Hybrid + Rerank", hybrid_search),
]


def run_evaluation(
    questions: list[dict[str, Any]] | None = None,
) -> list[MethodResult]:
    """Run the full evaluation across all methods and questions.

    Parameters
    ----------
    questions:
        Optional list of question dicts.  If *None*, they are loaded from
        :data:`TEST_QUESTIONS_PATH`.

    Returns
    -------
    list[MethodResult]
        One :class:`MethodResult` per retrieval method, with per-question
        scores attached.
    """
    if questions is None:
        questions = load_test_questions()

    results: list[MethodResult] = []

    for method_name, search_fn in METHODS:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", method_name)
        logger.info("=" * 60)

        method_result = MethodResult(method_name=method_name)

        for question in questions:
            try:
                qr = evaluate_question(question, search_fn)
                method_result.question_results.append(qr)
            except Exception:
                logger.exception(
                    "Unexpected error evaluating %s with %s",
                    question.get("id", "?"),
                    method_name,
                )

        results.append(method_result)

    return results


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
def print_results(results: list[MethodResult]) -> None:
    """Print a formatted comparison table to stdout.

    Parameters
    ----------
    results:
        List of :class:`MethodResult` objects, one per retrieval method.
    """
    headers = ["Method", "C.Prec", "Faith", "A.Rel", "Key", "Lat(s)"]
    rows = [
        [
            r.method_name,
            f"{r.avg_context_precision:.3f}",
            f"{r.avg_faithfulness:.3f}",
            f"{r.avg_answer_relevancy:.3f}",
            f"{r.avg_keyword_score:.3f}",
            f"{r.avg_latency:.2f}",
        ]
        for r in results
    ]

    table = tabulate(
        rows,
        headers=headers,
        tablefmt="simple",
        colalign=("left", "right", "right", "right", "right", "right"),
    )

    width = max(len(line) for line in table.splitlines())
    separator = "=" * width
    print()
    print(separator)
    print(table)
    print(separator)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: load questions, evaluate, and print results."""
    logger.info("Starting GRIFF evaluation pipeline")
    logger.info("Test questions file: %s", TEST_QUESTIONS_PATH)

    try:
        results = run_evaluation()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    except Exception:
        logger.exception("Evaluation pipeline failed")
        raise SystemExit(1)

    print_results(results)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
