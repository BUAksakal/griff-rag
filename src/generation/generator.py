"""Answer generator for the GRIFF RAG pipeline.

Uses the Groq API (LLaMA-3.3-70B) by default or falls back to a local
HuggingFace causal-LM when ``USE_LOCAL_LLM=true`` is set in the environment.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_MODEL_NAME: str = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

LANGUAGE_MAP: dict[str, str] = {
    "tr": "Turkish",
    "en": "English",
    "de": "German",
    "ar": "Arabic",
    "uk": "Ukrainian",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pl": "Polish",
    "auto": "the same language as the question",
}

SYSTEM_PROMPT: str = (
    "You are GRIFF (German Regulatory & Immigration Facts For Foreigners), an expert assistant "
    "specializing in German bureaucracy, immigration, visa, tax, health insurance, and official procedures.\n\n"
    "Rules:\n"
    "1. Answer ONLY based on the provided context chunks. Do not use prior knowledge.\n"
    "2. If the context does not contain enough information, say so clearly.\n"
    "3. Always cite your sources by mentioning the URL where the information comes from.\n"
    "4. Answer in the SAME LANGUAGE as the question "
    "(Turkish, English, German, Arabic, Ukrainian, Russian, Spanish, French, Italian, Polish).\n"
    "5. Be concise but thorough. Use bullet points for multi-step processes.\n"
    "6. If there are deadlines or time limits, highlight them prominently."
)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_groq_client: Any | None = None
_local_model: Any | None = None
_local_tokenizer: Any | None = None


def _get_groq_client() -> Any:
    """Return a cached Groq client instance."""
    global _groq_client  # noqa: PLW0603
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        from groq import Groq  # type: ignore[import-untyped]

        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised (model=%s).", GROQ_MODEL)
    return _groq_client


def _get_local_model() -> tuple[Any, Any]:
    """Return a cached (model, tokenizer) tuple for the local LLM."""
    global _local_model, _local_tokenizer  # noqa: PLW0603
    if _local_model is None or _local_tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        logger.info("Loading local model '%s' — this may take a while …", LOCAL_MODEL_NAME)
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
        )
        logger.info("Local model '%s' loaded successfully.", LOCAL_MODEL_NAME)
    return _local_model, _local_tokenizer


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_context_block(context_chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Format context chunks into a numbered text block.

    Returns:
        A tuple of ``(formatted_text, source_urls)``.
    """
    lines: list[str] = []
    sources: list[str] = []

    for idx, chunk in enumerate(context_chunks, start=1):
        text = chunk.get("text", chunk.get("content", ""))
        url = chunk.get("url", chunk.get("source", ""))
        lines.append(f"[Chunk {idx}] (Source: {url})\n{text}")
        if url and url not in sources:
            sources.append(url)

    return "\n\n".join(lines), sources


def _build_user_prompt(
    query: str,
    context_chunks: list[dict[str, Any]],
    response_language: str,
) -> tuple[str, list[str]]:
    """Assemble the full user-side prompt.

    Returns:
        A tuple of ``(prompt_text, source_urls)``.
    """
    context_block, sources = _build_context_block(context_chunks)

    parts: list[str] = [
        "### Context Chunks ###",
        context_block,
        "",
        "### User Question ###",
        query,
    ]

    if response_language != "auto":
        lang_name = LANGUAGE_MAP.get(response_language, response_language)
        parts.append(f"\n### Language Instruction ###\nRespond in {lang_name}.")

    return "\n".join(parts), sources


# ---------------------------------------------------------------------------
# Generation back-ends
# ---------------------------------------------------------------------------


def _generate_with_groq(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Call the Groq API and return ``(answer_text, model_name)``."""
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    answer: str = response.choices[0].message.content
    return answer, GROQ_MODEL


def _generate_with_local(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Run inference on the local HuggingFace model and return ``(answer_text, model_name)``."""
    model, tokenizer = _get_local_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_text: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.3,
        do_sample=True,
    )

    # Decode only the newly generated tokens.
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer: str = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer, LOCAL_MODEL_NAME


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    context_chunks: list[dict[str, Any]],
    response_language: str = "auto",
) -> dict[str, Any]:
    """Generate an answer for *query* using retrieved *context_chunks*.

    Args:
        query: The user's natural-language question.
        context_chunks: A list of dicts, each containing at least ``text``
            (or ``content``) and ``url`` (or ``source``) keys.
        response_language: ISO 639-1 code (e.g. ``"tr"``, ``"en"``) or
            ``"auto"`` to let the model match the question's language.

    Returns:
        A dict with keys ``answer``, ``sources``, and ``model_used``.
    """
    if not context_chunks:
        logger.warning("generate_answer called with an empty context_chunks list.")
        return {
            "answer": "No context chunks were provided — unable to generate an answer.",
            "sources": [],
            "model_used": GROQ_MODEL if not USE_LOCAL_LLM else LOCAL_MODEL_NAME,
        }

    user_prompt, sources = _build_user_prompt(query, context_chunks, response_language)

    try:
        if USE_LOCAL_LLM:
            logger.info("Generating answer with local model '%s'.", LOCAL_MODEL_NAME)
            answer, model_used = _generate_with_local(SYSTEM_PROMPT, user_prompt)
        else:
            logger.info("Generating answer with Groq model '%s'.", GROQ_MODEL)
            answer, model_used = _generate_with_groq(SYSTEM_PROMPT, user_prompt)

        logger.info("Answer generated successfully (model=%s, length=%d chars).", model_used, len(answer))
        return {
            "answer": answer,
            "sources": sources,
            "model_used": model_used,
        }

    except Exception:
        logger.exception("Failed to generate answer.")
        return {
            "answer": "An error occurred while generating the answer. Please try again later.",
            "sources": sources,
            "model_used": GROQ_MODEL if not USE_LOCAL_LLM else LOCAL_MODEL_NAME,
        }
