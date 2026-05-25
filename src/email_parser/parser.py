"""German official email / letter parser for GRIFF.

Sends raw German correspondence to the Groq API (LLaMA-3.3-70B) and returns
a structured JSON summary with sender, subject, urgency, deadline, required
actions, and a plain-language summary.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
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

SYSTEM_PROMPT_TEMPLATE: str = """\
You are GRIFF (German Regulatory & Immigration Facts For Foreigners), \
an expert at reading German official correspondence.

Your task is to parse the provided German email or letter and extract \
structured information. Return your answer as **valid JSON only** — \
no markdown fences, no commentary, just the raw JSON object.

The JSON object must have exactly these keys:
- "sender": string — the institution or organisation that sent the letter.
- "subject": string — a brief description of what the letter is about.
- "urgency": string — one of "low", "medium", "high", or "critical".
- "deadline": string or null — any deadline mentioned (ISO-8601 date if possible, otherwise verbatim text). Use null if none.
- "required_actions": list of strings — concrete actions the recipient must take.
- "summary": string — a plain-language summary of the letter.
- "original_language": string — the ISO 639-1 code of the input text (e.g. "de").

Respond entirely in {response_language}.\
"""

# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_groq_client: Any | None = None


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
        logger.info("Groq client initialised for email parser (model=%s).", GROQ_MODEL)
    return _groq_client


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from *text*.

    Tries direct ``json.loads`` first, then looks for a fenced code block.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    # 1. Try the raw text directly.
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 2. Try to find a ```json ... ``` block.
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # 3. Try to find the first { … } pair.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from the model response.")


def _build_fallback_response(raw_text: str) -> dict[str, Any]:
    """Return a best-effort fallback when JSON parsing fails entirely."""
    return {
        "sender": "Unknown",
        "subject": "Unable to parse",
        "urgency": "medium",
        "deadline": None,
        "required_actions": ["Please review the original text manually."],
        "summary": raw_text[:500],
        "original_language": "de",
        "parse_error": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_email(
    email_text: str,
    response_language: str = "en",
) -> dict[str, Any]:
    """Parse a German official email or letter into structured data.

    Args:
        email_text: The raw text of the German email or letter.
        response_language: ISO 639-1 language code for the output
            (default ``"en"``).  Supported codes are listed in
            ``LANGUAGE_MAP``.

    Returns:
        A dict with keys ``sender``, ``subject``, ``urgency``, ``deadline``,
        ``required_actions``, ``summary``, ``original_language``, and
        ``model_used``.  If JSON parsing fails, a ``parse_error: True``
        flag is added and the raw model output is placed in ``summary``.
    """
    if not email_text or not email_text.strip():
        logger.warning("parse_email called with empty text.")
        return {
            "sender": "Unknown",
            "subject": "Empty input",
            "urgency": "low",
            "deadline": None,
            "required_actions": [],
            "summary": "No email text was provided.",
            "original_language": "unknown",
            "model_used": GROQ_MODEL,
        }

    lang_name = LANGUAGE_MAP.get(response_language, response_language)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(response_language=lang_name)

    try:
        client = _get_groq_client()
        logger.info("Sending email text to Groq for parsing (%d chars).", len(email_text))

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": email_text},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        raw_answer: str = response.choices[0].message.content
        logger.debug("Raw model response:\n%s", raw_answer)

        try:
            result = _extract_json(raw_answer)
            logger.info("Email parsed successfully.")
        except ValueError:
            logger.warning("JSON extraction failed — returning fallback response.")
            result = _build_fallback_response(raw_answer)

        result["model_used"] = GROQ_MODEL
        return result

    except Exception:
        logger.exception("Failed to parse email via Groq API.")
        fallback = _build_fallback_response(email_text)
        fallback["model_used"] = GROQ_MODEL
        return fallback
