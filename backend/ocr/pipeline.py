"""
OCR Pipeline — Tesseract + pdf2image for scanned PDFs,
PyMuPDF for native digital PDFs.
"""
import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Native PDF text extraction ────────────────────────────────────────────────

def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """Return True when the PDF contains almost no selectable text (likely scanned)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        char_count = sum(len(p.get_text().strip()) for p in doc)
        doc.close()
        return char_count < 150
    except Exception as exc:
        logger.warning("is_scanned_pdf check failed (%s) — assuming digital.", exc)
        return False


def extract_text_native(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extract text from a native digital PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    n = len(doc)
    doc.close()
    return "\n\n".join(pages), n


# ── OCR text extraction ───────────────────────────────────────────────────────

def _preprocess(image):
    """Grayscale + 2× upscale for better Tesseract accuracy."""
    from PIL import Image
    img = image.convert("L")
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)
    return img


def extract_text_ocr(pdf_bytes: bytes) -> Tuple[str, int]:
    """Convert scanned PDF pages to images → Tesseract OCR → text."""
    from pdf2image import convert_from_bytes
    import pytesseract

    images = convert_from_bytes(pdf_bytes, dpi=300)
    texts = []
    for img in images:
        processed = _preprocess(img)
        text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        texts.append(text)
    return "\n\n".join(texts), len(images)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes, use_ocr: bool = False) -> Tuple[str, int]:
    """Main entry-point: route to OCR or native extraction based on flag."""
    if use_ocr:
        logger.info("Using Tesseract OCR pipeline …")
        return extract_text_ocr(pdf_bytes)
    logger.info("Using native PyMuPDF extraction …")
    return extract_text_native(pdf_bytes)
