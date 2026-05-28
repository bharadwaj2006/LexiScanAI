"""
=============================================================
LEXISCAN AUTO — STEP 3B: RULE-BASED VALIDATION
=============================================================
PURPOSE:
    Raw NER output is often imperfect.
    Even a well-trained model sometimes extracts:
        - Malformed dates: "March 202" (truncated)
        - Amount without currency: "2500000" (no $ or USD)
        - Half-sentences as termination clauses

    This module applies strict validation rules on top of
    the model output to:
        1. CHECK that each entity matches expected format
        2. FLAG entities that fail validation
        3. NORMALISE entities into clean, standard formats
        4. FILTER out entities with low confidence

    Think of this as the "lawyer trust" layer — it's what
    makes the system reliable enough for a legal firm.

OUTPUT FORMAT STANDARDS:
    DATE              → YYYY-MM-DD  (e.g., "2024-03-15")
    DOLLAR_AMOUNT     → $X,XXX.XX   (e.g., "$2,500,000.00")
    PARTY             → Title Case, length > 3 chars
    TERMINATION_CLAUSE→ Must contain at least one termination keyword
=============================================================
"""

import re
import logging
from datetime import datetime
from dateutil import parser as dateutil_parser   # pip install python-dateutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# DATE patterns — covers common formats found in legal contracts
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                         # 2024-03-15
    r"\b\d{2}/\d{2}/\d{4}\b",                         # 15/03/2024
    r"\b\d{2}-\d{2}-\d{4}\b",                         # 15-03-2024
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",  # March 15, 2024
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",  # 15th March 2024
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",  # March 2024
]

# DOLLAR AMOUNT patterns — covers various currency formats in US legal docs
AMOUNT_PATTERNS = [
    r"\$[\d,]+(?:\.\d{2})?",      # $2,500,000 or $2,500,000.00
    r"USD\s*[\d,]+(?:\.\d{2})?",  # USD 2,500,000
    r"[\d,]+\s*dollars?",          # 2,500,000 dollars
    r"\bUS\$[\d,]+(?:\.\d{2})?",  # US$2,500,000
]

# Keywords that must appear in a valid TERMINATION_CLAUSE
TERMINATION_KEYWORDS = [
    "terminat", "cancel", "withdraw", "notice", "expir",
    "dissolv", "rescind", "void", "end", "cease"
]

# Party name — too short = likely extraction noise
PARTY_MIN_LENGTH = 4


# ─────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────

def validate_date(text: str) -> dict:
    """
    Check if extracted text is a valid date.
    Attempt to normalise it to YYYY-MM-DD standard format.

    Returns:
        {"valid": True/False, "normalised": "YYYY-MM-DD" or None, "reason": "..."}
    """
    text = text.strip()

    # Check against known patterns first
    matched = any(re.search(p, text, re.IGNORECASE) for p in DATE_PATTERNS)
    if not matched:
        return {"valid": False, "normalised": None, "reason": "No recognised date pattern"}

    # Try to parse into a real date (catches "February 30" type errors)
    try:
        parsed = dateutil_parser.parse(text, dayfirst=False)

        # Sanity check — dates in legal contracts are typically 2000–2050
        if not (2000 <= parsed.year <= 2050):
            return {"valid": False, "normalised": None, "reason": f"Year {parsed.year} out of expected range"}

        normalised = parsed.strftime("%Y-%m-%d")
        return {"valid": True, "normalised": normalised, "reason": "OK"}

    except (ValueError, OverflowError) as e:
        return {"valid": False, "normalised": None, "reason": f"Parse error: {e}"}


def validate_dollar_amount(text: str) -> dict:
    """
    Check if extracted text is a valid dollar amount.
    Normalises to $X,XXX.XX format.

    Returns:
        {"valid": True/False, "normalised": "$X,XXX.XX" or None, "reason": "..."}
    """
    text = text.strip()

    # Must match at least one currency pattern
    matched = any(re.search(p, text, re.IGNORECASE) for p in AMOUNT_PATTERNS)
    if not matched:
        return {"valid": False, "normalised": None, "reason": "No currency symbol or keyword found"}

    # Extract just the numeric part
    numeric_str = re.sub(r"[^\d.]", "", text)
    if not numeric_str:
        return {"valid": False, "normalised": None, "reason": "No numeric value found"}

    try:
        value = float(numeric_str)

        # Sanity check — amounts in legal contracts
        if value <= 0:
            return {"valid": False, "normalised": None, "reason": "Amount must be positive"}
        if value > 1_000_000_000_000:  # > $1 trillion — likely extraction noise
            return {"valid": False, "normalised": None, "reason": "Amount suspiciously large"}

        # Format to $X,XXX.XX
        normalised = "${:,.2f}".format(value)
        return {"valid": True, "normalised": normalised, "reason": "OK"}

    except ValueError as e:
        return {"valid": False, "normalised": None, "reason": f"Numeric parse error: {e}"}


def validate_party(text: str) -> dict:
    """
    Check if extracted text is a plausible party/entity name.
    Normalises to Title Case.

    Returns:
        {"valid": True/False, "normalised": "..." or None, "reason": "..."}
    """
    text = text.strip()

    if len(text) < PARTY_MIN_LENGTH:
        return {"valid": False, "normalised": None, "reason": f"Too short (< {PARTY_MIN_LENGTH} chars)"}

    # Must have at least one alphabetic character
    if not re.search(r"[a-zA-Z]", text):
        return {"valid": False, "normalised": None, "reason": "No alphabetic characters"}

    # Must not be only stopwords
    stopwords = {"the", "a", "an", "this", "that", "and", "or", "of", "in", "to", "for"}
    words = [w.lower() for w in text.split()]
    if all(w in stopwords for w in words):
        return {"valid": False, "normalised": None, "reason": "Only stopwords — likely extraction noise"}

    # Normalise to Title Case
    normalised = " ".join(w.capitalize() for w in text.split())
    return {"valid": True, "normalised": normalised, "reason": "OK"}


def validate_termination_clause(text: str) -> dict:
    """
    Check if extracted text is a plausible termination clause.
    Must contain at least one termination-related keyword.

    Returns:
        {"valid": True/False, "normalised": "..." or None, "reason": "..."}
    """
    text = text.strip()

    if len(text) < 15:
        return {"valid": False, "normalised": None, "reason": "Too short to be a clause"}

    text_lower = text.lower()
    found_keywords = [kw for kw in TERMINATION_KEYWORDS if kw in text_lower]

    if not found_keywords:
        return {
            "valid": False,
            "normalised": None,
            "reason": f"No termination keywords found. Expected one of: {TERMINATION_KEYWORDS}"
        }

    # Return as-is (termination clauses are kept verbatim)
    return {"valid": True, "normalised": text, "reason": f"Keywords found: {found_keywords}"}


# ─────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ─────────────────────────────────────────────

VALIDATORS = {
    "DATE":               validate_date,
    "DOLLAR_AMOUNT":      validate_dollar_amount,
    "PARTY":              validate_party,
    "TERMINATION_CLAUSE": validate_termination_clause,
}


def validate_entities(raw_entities: list) -> dict:
    """
    Validate and normalise a list of extracted entities.

    Args:
        raw_entities: list of dicts from NER model, each:
            {"text": "...", "label": "DATE", "start": 0, "end": 10}

    Returns a structured dict:
        {
          "dates":               ["2024-03-15", "2026-03-14"],
          "parties":             ["Meridian Capital LLC", ...],
          "amounts":             ["$2,500,000.00", ...],
          "termination_clauses": ["Either party may terminate..."],
          "rejected":            [{"text": "...", "label": "...", "reason": "..."}, ...],
          "validation_summary":  {"total": N, "passed": N, "rejected": N}
        }
    """
    result = {
        "dates":               [],
        "parties":             [],
        "amounts":             [],
        "termination_clauses": [],
        "rejected":            [],
    }

    passed = 0
    for entity in raw_entities:
        text  = entity.get("text", "")
        label = entity.get("label", "")

        validator = VALIDATORS.get(label)
        if validator is None:
            logger.warning(f"Unknown label '{label}' — skipping.")
            continue

        validation = validator(text)

        if validation["valid"]:
            # Add normalised value to the correct bucket
            normalised = validation["normalised"]
            if label == "DATE":
                if normalised not in result["dates"]:
                    result["dates"].append(normalised)
            elif label == "DOLLAR_AMOUNT":
                if normalised not in result["amounts"]:
                    result["amounts"].append(normalised)
            elif label == "PARTY":
                if normalised not in result["parties"]:
                    result["parties"].append(normalised)
            elif label == "TERMINATION_CLAUSE":
                if normalised not in result["termination_clauses"]:
                    result["termination_clauses"].append(normalised)
            passed += 1
        else:
            result["rejected"].append({
                "text":   text,
                "label":  label,
                "reason": validation["reason"]
            })
            logger.debug(f"  REJECTED [{label}] '{text}' — {validation['reason']}")

    total = len(raw_entities)
    result["validation_summary"] = {
        "total":    total,
        "passed":   passed,
        "rejected": total - passed,
    }

    return result


# ─────────────────────────────────────────────
# TEST — run directly to see validation in action
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate raw NER output (mix of good and bad entities)
    raw_entities = [
        {"text": "March 15, 2024",          "label": "DATE"},
        {"text": "2026-12-31",              "label": "DATE"},
        {"text": "March 202",               "label": "DATE"},     # ← SHOULD FAIL
        {"text": "the",                     "label": "DATE"},     # ← SHOULD FAIL

        {"text": "$2,500,000.00",           "label": "DOLLAR_AMOUNT"},
        {"text": "USD 150,000",             "label": "DOLLAR_AMOUNT"},
        {"text": "2500000",                 "label": "DOLLAR_AMOUNT"},  # ← SHOULD FAIL (no $)
        {"text": "-500",                    "label": "DOLLAR_AMOUNT"},  # ← SHOULD FAIL (negative)

        {"text": "Meridian Capital LLC",    "label": "PARTY"},
        {"text": "Vantage Holdings Inc.",   "label": "PARTY"},
        {"text": "the",                     "label": "PARTY"},    # ← SHOULD FAIL
        {"text": "AB",                      "label": "PARTY"},    # ← SHOULD FAIL (too short)

        {"text": "Either party may terminate this agreement with 90 days written notice",
                                            "label": "TERMINATION_CLAUSE"},
        {"text": "This is just a sentence.", "label": "TERMINATION_CLAUSE"},  # ← SHOULD FAIL
    ]

    print("=" * 60)
    print("LEXISCAN AUTO — VALIDATION TEST")
    print("=" * 60)

    validated = validate_entities(raw_entities)

    print(f"\n✓ DATES           : {validated['dates']}")
    print(f"✓ AMOUNTS         : {validated['amounts']}")
    print(f"✓ PARTIES         : {validated['parties']}")
    print(f"✓ TERMINATION     : {validated['termination_clauses']}")
    print(f"\n✗ REJECTED ({len(validated['rejected'])}):")
    for r in validated["rejected"]:
        print(f"   [{r['label']}] '{r['text']}' → {r['reason']}")

    s = validated["validation_summary"]
    print(f"\nSUMMARY: {s['total']} total | {s['passed']} passed | {s['rejected']} rejected")
