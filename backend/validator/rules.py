"""
Rule-Based Validation & Normalization Engine.

Validates raw NER output against legal-domain rules:
  DATE   → must parse to a real date; normalize to YYYY-MM-DD
  AMOUNT → must contain currency indicator; normalize to numeric + currency code
  PARTY  → must be ≥2 words, no purely numeric tokens
  TERMINATION_CLAUSE → must contain ≥1 termination keyword, ≥20 chars
"""
import re
import logging
from typing import Dict, List
from dateutil import parser as dateparser
from dateutil.parser import ParserError

logger = logging.getLogger(__name__)

# ── DATE validation ───────────────────────────────────────────────────────────

def _validate_date(entity: Dict) -> Dict | None:
    text = entity["text"].strip()
    try:
        dt = dateparser.parse(text, fuzzy=True)
        if dt is None:
            return None
        # Reject obviously wrong years
        if dt.year < 1900 or dt.year > 2100:
            return None
        entity["normalized"] = dt.strftime("%Y-%m-%d")
        entity["valid"] = True
        return entity
    except (ParserError, OverflowError, ValueError):
        return None


# ── AMOUNT validation ─────────────────────────────────────────────────────────

_CURRENCY_MAP = {
    "$": "USD", "USD": "USD", "EUR": "EUR", "GBP": "GBP",
    "CAD": "CAD", "AUD": "AUD",
}

_AMOUNT_CLEAN_RE = re.compile(r"[^\d.]")

def _validate_amount(entity: Dict) -> Dict | None:
    text = entity["text"].strip()
    currency = "USD"

    # Detect currency
    for symbol, code in _CURRENCY_MAP.items():
        if symbol in text:
            currency = code
            break
    else:
        # No currency indicator found — reject
        return None

    # Extract numeric value
    numeric_str = _AMOUNT_CLEAN_RE.sub("", text)
    try:
        value = float(numeric_str) if numeric_str else None
    except ValueError:
        value = None

    if value is None or value <= 0:
        return None

    entity["currency"] = currency
    entity["normalized_value"] = round(value, 2)
    entity["normalized"] = f"{currency} {value:,.2f}"
    entity["valid"] = True
    return entity


# ── PARTY validation ──────────────────────────────────────────────────────────

_NOISE_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "agreement", "contract", "herein", "hereunder",
}

def _validate_party(entity: Dict) -> Dict | None:
    text = entity["text"].strip()
    words = text.split()

    # Must have ≥2 tokens
    if len(words) < 2:
        return None

    # Must not be all digits / punctuation
    if all(not ch.isalpha() for ch in text):
        return None

    # Filter common noise
    if words[0].lower() in _NOISE_WORDS:
        return None

    # Must have at least one capitalized word
    if not any(w[0].isupper() for w in words if w.isalpha()):
        return None

    entity["valid"] = True
    return entity


# ── TERMINATION validation ────────────────────────────────────────────────────

_TERM_KW = re.compile(
    r"\b(terminat|cancell?ation|rescission|expir|cessation|discontinu|"
    r"notice\s+of\s+termination|shall\s+cease|wind[\s-]up)\w*\b",
    re.IGNORECASE,
)

def _validate_termination(entity: Dict) -> Dict | None:
    text = entity["text"].strip()
    if len(text) < 20:
        return None
    if not _TERM_KW.search(text):
        return None
    entity["valid"] = True
    return entity


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_normalize(raw: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Validate and normalize each entity group.
    Invalid entities are silently dropped.
    """
    validated: Dict[str, List[Dict]] = {
        "dates": [],
        "amounts": [],
        "parties": [],
        "termination_clauses": [],
    }

    for e in raw.get("dates", []):
        result = _validate_date(dict(e))
        if result:
            validated["dates"].append(result)

    for e in raw.get("amounts", []):
        result = _validate_amount(dict(e))
        if result:
            validated["amounts"].append(result)

    for e in raw.get("parties", []):
        result = _validate_party(dict(e))
        if result:
            validated["parties"].append(result)

    for e in raw.get("termination_clauses", []):
        result = _validate_termination(dict(e))
        if result:
            validated["termination_clauses"].append(result)

    logger.info(
        "Validation: %d dates, %d amounts, %d parties, %d termination clauses",
        len(validated["dates"]), len(validated["amounts"]),
        len(validated["parties"]), len(validated["termination_clauses"]),
    )
    return validated
