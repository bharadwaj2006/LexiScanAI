import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LexiScan Auto API",
    description="Legal Contract Entity Extractor — OCR + NLP Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading NER model …")
    from ner.model import get_nlp_model
    get_nlp_model()
    logger.info("NER model ready ✓")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "LexiScan Auto API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "LexiScan Auto", "version": "1.0.0"}


@app.post("/api/extract")
async def extract(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    start = time.time()
    ocr_used = False
    page_count = 1
    raw_text = ""

    try:
        if file and file.filename:
            content = await file.read()
            fname = file.filename.lower()

            if fname.endswith(".pdf"):
                from ocr.pipeline import is_scanned_pdf, extract_text_from_pdf
                scanned = is_scanned_pdf(content)
                raw_text, page_count = extract_text_from_pdf(content, use_ocr=scanned)
                ocr_used = scanned
                logger.info(f"PDF processed — pages:{page_count}, ocr:{ocr_used}")
            else:
                raw_text = content.decode("utf-8", errors="ignore")

        elif text:
            raw_text = text
        else:
            raise HTTPException(400, "Provide either a file or text field.")

        if not raw_text.strip():
            raise HTTPException(422, "No extractable text found in the input.")

        from ner.model import extract_legal_entities
        from validator.rules import validate_and_normalize

        raw_entities = extract_legal_entities(raw_text)
        validated = validate_and_normalize(raw_entities)
        total = sum(len(v) for v in validated.values())
        elapsed = int((time.time() - start) * 1000)

        return JSONResponse({
            "status": "success",
            "processing_time_ms": elapsed,
            "ocr_used": ocr_used,
            "page_count": page_count,
            "total_entities_found": total,
            "entities": validated,
            "raw_text_preview": raw_text[:800] + ("…" if len(raw_text) > 800 else ""),
            "character_count": len(raw_text),
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Extraction failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Processing error: {exc}")


@app.get("/api/demo")
async def demo():
    """Run extraction on a built-in sample contract (no upload needed)."""
    sample = """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2024 (the "Effective Date")
by and between Acme Corporation, a Delaware corporation ("Service Provider"), with its principal
place of business at 123 Main Street, New York, NY 10001, and GlobalTech Industries LLC
("Client"), a California limited liability company located at 456 Innovation Drive,
San Francisco, CA 94105.

ARTICLE 1 — COMPENSATION
1.1 The Client shall pay a total fee of $2,500,000 (Two Million Five Hundred Thousand United States
Dollars) for the Services, payable in quarterly instalments of $625,000 each.
1.2 An additional retainer of USD 150,000 per annum applies from March 1, 2024 through February 28, 2026.

ARTICLE 5 — TERM AND TERMINATION
5.1 This Agreement commences on the Effective Date and expires on January 14, 2026, unless earlier
terminated under this Article.
5.2 Termination for Convenience. Either party may terminate this Agreement for any reason upon
ninety (90) days prior written notice to the other party.
5.3 Termination for Cause. Either party may terminate this Agreement immediately upon written notice
if the other party materially breaches this Agreement and fails to cure such breach within thirty
(30) days of written notice thereof.
5.4 Upon termination, all licences granted hereunder shall immediately cease and Service Provider
shall return all Client materials within fifteen (15) business days.

Executed by the parties as of the date first written above.

ACME CORPORATION                    GLOBALTECH INDUSTRIES LLC
By: John Smith, CEO                 By: Sarah Johnson, President
Date: January 15, 2024              Date: January 15, 2024
"""
    from ner.model import extract_legal_entities
    from validator.rules import validate_and_normalize

    t0 = time.time()
    raw = extract_legal_entities(sample)
    validated = validate_and_normalize(raw)
    elapsed = int((time.time() - t0) * 1000)
    total = sum(len(v) for v in validated.values())

    return JSONResponse({
        "status": "success",
        "processing_time_ms": elapsed,
        "ocr_used": False,
        "page_count": 1,
        "total_entities_found": total,
        "entities": validated,
        "raw_text_preview": sample[:800],
        "character_count": len(sample),
    })
