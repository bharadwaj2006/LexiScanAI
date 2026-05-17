"""
NER Model — SpaCy en_core_web_sm + EntityRuler with legal patterns + regex fallback.

Strategy (zero-training-required for demo):
  1. SpaCy EntityRuler injects high-precision legal patterns BEFORE the base NER.
  2. The base NER ORG/PERSON labels are remapped to PARTY.
  3. Regex fallback catches any remaining DATE / AMOUNT patterns.
  4. Termination clauses are extracted at the sentence level by keyword search.
"""
import re
import logging
from typing import Dict, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Singleton model ───────────────────────────────────────────────────────────

_nlp = None


def get_nlp_model():
    global _nlp
    if _nlp is None:
        _nlp = _build_pipeline()
    return _nlp


def _build_pipeline():
    import spacy
    logger.info("Loading SpaCy model …")
    nlp = spacy.load("en_core_web_sm")

    # Insert EntityRuler BEFORE the NER component so our patterns take priority
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

    patterns = []

    # ── DATE patterns ─────────────────────────────────────────────────────────
    month_words = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    ]
    # "January 15, 2024"
    patterns.append({
        "label": "LEGAL_DATE",
        "pattern": [
            {"LOWER": {"IN": month_words}},
            {"TEXT": {"REGEX": r"^\d{1,2}(st|nd|rd|th)?$"}},
            {"TEXT": ",", "OP": "?"},
            {"TEXT": {"REGEX": r"^\d{4}$"}},
        ],
    })
    # "15th day of January, 2024"
    patterns.append({
        "label": "LEGAL_DATE",
        "pattern": [
            {"TEXT": {"REGEX": r"^\d{1,2}(st|nd|rd|th)?$"}},
            {"LOWER": "day", "OP": "?"},
            {"LOWER": "of", "OP": "?"},
            {"LOWER": {"IN": month_words}},
            {"TEXT": ",", "OP": "?"},
            {"TEXT": {"REGEX": r"^\d{4}$"}},
        ],
    })

    # ── AMOUNT patterns ───────────────────────────────────────────────────────
    # "$5,000,000" or "$625,000.00"
    patterns.append({
        "label": "LEGAL_AMOUNT",
        "pattern": [{"TEXT": {"REGEX": r"^\$[\d,]+(\.\d{1,2})?$"}}],
    })
    # "USD 150,000" / "EUR 50,000"
    patterns.append({
        "label": "LEGAL_AMOUNT",
        "pattern": [
            {"TEXT": {"REGEX": r"^(USD|EUR|GBP|CAD|AUD)$"}},
            {"TEXT": {"REGEX": r"^[\d,]+(\.\d{1,2})?$"}},
        ],
    })
    # "5,000,000 United States Dollars"
    patterns.append({
        "label": "LEGAL_AMOUNT",
        "pattern": [
            {"TEXT": {"REGEX": r"^[\d,]+(\.\d{1,2})?$"}},
            {"LOWER": {"IN": ["united", "us", "u.s."]}},
            {"LOWER": {"IN": ["states", "dollars", "dollar"]}, "OP": "?"},
            {"LOWER": {"IN": ["dollars", "dollar"]}, "OP": "?"},
        ],
    })

    # ── PARTY patterns (org suffixes) ─────────────────────────────────────────
    org_suffixes = [
        "llc", "corp", "corporation", "inc", "incorporated", "ltd",
        "limited", "llp", "lp", "plc", "co", "company", "group",
        "holdings", "associates", "partners", "partnership",
    ]
    patterns.append({
        "label": "LEGAL_PARTY",
        "pattern": [
            {"IS_TITLE": True},
            {"IS_TITLE": True, "OP": "?"},
            {"IS_TITLE": True, "OP": "?"},
            {"LOWER": {"IN": org_suffixes}},
        ],
    })

    ruler.add_patterns(patterns)
    logger.info("EntityRuler loaded with %d patterns.", len(patterns))
    return nlp


# ── Regex fallbacks ───────────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r"""
    \b(
        (?:January|February|March|April|May|June|July|August|September|
           October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
        [\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{4}
      |
        \d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}
      |
        \d{4}-\d{2}-\d{2}
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_AMOUNT_RE = re.compile(
    r"""
    \b(
        \$[\d,]+(?:\.\d{1,2})?
      |
        (?:USD|EUR|GBP|CAD|AUD)\s*[\d,]+(?:\.\d{1,2})?
      |
        [\d,]+(?:\.\d{1,2})?\s+(?:United\s+States\s+)?[Dd]ollars?
    )\b
    """,
    re.VERBOSE,
)

_TERMINATION_KEYWORDS = re.compile(
    r"\b(terminat|cancell?ation|rescission|expir|cessation|discontinu|"
    r"notice\s+of\s+termination|shall\s+cease|wind[\s-]up)\w*\b",
    re.IGNORECASE,
)


def _extract_termination_clauses(text: str) -> List[Dict]:
    """Return sentences / short paragraphs that contain termination language."""
    results = []
    seen = set()
    # Split by newline-bounded paragraphs first, then sentences
    segments = re.split(r"\n{2,}", text)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if _TERMINATION_KEYWORDS.search(seg):
            # Further split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", seg)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                if _TERMINATION_KEYWORDS.search(sent) and sent not in seen:
                    seen.add(sent)
                    results.append({
                        "text": sent,
                        "label": "TERMINATION_CLAUSE",
                        "confidence": 0.88,
                    })
    return results[:10]  # cap at 10


# ── Public extraction API ─────────────────────────────────────────────────────

def extract_legal_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract LEGAL_DATE, LEGAL_AMOUNT, LEGAL_PARTY, TERMINATION_CLAUSE
    from the given text and return a raw (pre-validation) dict.
    """
    nlp = get_nlp_model()
    doc = nlp(text)

    dates: List[Dict] = []
    amounts: List[Dict] = []
    parties: List[Dict] = []
    seen_texts = {"dates": set(), "amounts": set(), "parties": set()}

    # ── SpaCy entities ────────────────────────────────────────────────────────
    for ent in doc.ents:
        t = ent.text.strip()

        if ent.label_ in ("LEGAL_DATE", "DATE") and t not in seen_texts["dates"]:
            seen_texts["dates"].add(t)
            dates.append({"text": t, "label": "DATE", "confidence": 0.91})

        elif ent.label_ == "LEGAL_AMOUNT" and t not in seen_texts["amounts"]:
            seen_texts["amounts"].add(t)
            amounts.append({"text": t, "label": "AMOUNT", "confidence": 0.94})

        elif ent.label_ in ("MONEY", "CARDINAL") and "$" in t and t not in seen_texts["amounts"]:
            seen_texts["amounts"].add(t)
            amounts.append({"text": t, "label": "AMOUNT", "confidence": 0.82})

        elif ent.label_ in ("LEGAL_PARTY", "ORG", "PERSON") and t not in seen_texts["parties"]:
            if len(t.split()) >= 2:
                seen_texts["parties"].add(t)
                parties.append({"text": t, "label": "PARTY", "confidence": 0.87})

    # ── Regex fallbacks ───────────────────────────────────────────────────────
    for m in _DATE_RE.finditer(text):
        t = m.group().strip()
        if t not in seen_texts["dates"]:
            seen_texts["dates"].add(t)
            dates.append({"text": t, "label": "DATE", "confidence": 0.85})

    for m in _AMOUNT_RE.finditer(text):
        t = m.group().strip()
        if t not in seen_texts["amounts"]:
            seen_texts["amounts"].add(t)
            amounts.append({"text": t, "label": "AMOUNT", "confidence": 0.86})

    termination = _extract_termination_clauses(text)

    return {
        "dates": dates,
        "amounts": amounts,
        "parties": parties,
        "termination_clauses": termination,
    }
