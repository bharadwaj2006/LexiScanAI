"""
=============================================================
LEXISCAN AUTO — STEP 1: OCR PIPELINE
=============================================================
PURPOSE:
    Accept a PDF (native digital OR scanned image-based).
    Extract clean, raw text ready for NLP.

LIBRARIES NEEDED (install once):
    pip install pymupdf pytesseract pillow opencv-python-headless
    sudo apt-get install tesseract-ocr   # Linux/Mac
    # Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki

HOW IT WORKS:
    1. Try to extract text digitally (native PDF) — fast & accurate.
    2. If no text found, treat as scanned — rasterize pages to images.
    3. Pre-process each image (grayscale, deskew, denoise, threshold).
    4. Run Tesseract OCR on the cleaned image.
    5. Return one clean string of text per PDF.
=============================================================
"""

import fitz                          # PyMuPDF — opens PDFs
import pytesseract                   # Tesseract OCR wrapper
import cv2                           # OpenCV — image processing
import numpy as np
from PIL import Image
import io
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPER: Pre-process image before OCR
# ─────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Clean up a scanned page image so Tesseract reads it accurately.
    Steps:
      - Convert to grayscale (colour confuses OCR)
      - Deskew (straighten tilted scans)
      - Denoise (remove speckles, stamps)
      - Threshold (make text pure black, background pure white)
    """
    # 1. Convert PIL image → NumPy array for OpenCV
    img = np.array(pil_image)

    # 2. Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 3. Denoise — remove salt-and-pepper noise from scanner
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Adaptive threshold — handles uneven lighting across the page
    #    (common in scanned contracts with stamps or highlighting)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,  # neighbourhood size
        C=2            # constant subtracted from mean
    )

    # 5. Deskew — detect skew angle and rotate to straighten text
    coords = np.column_stack(np.where(thresh < 128))   # find dark pixels
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.3:   # only rotate if skew is noticeable
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    # Convert back to PIL for Tesseract
    return Image.fromarray(thresh)


# ─────────────────────────────────────────────
# CORE: Extract text from a single PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Main function. Accepts path to any PDF.
    Returns a dict with:
        - 'text'       : full cleaned text string
        - 'method'     : 'digital' or 'ocr'
        - 'page_count' : number of pages processed
        - 'filename'   : original filename

    Usage:
        result = extract_text_from_pdf("contracts/agreement_001.pdf")
        print(result['text'])
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    logger.info(f"Processing: {filename}")

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    all_text = []

    # ── ATTEMPT 1: Native digital text extraction ──
    # Native PDFs have embedded text — extract it directly (no OCR needed)
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text("text")   # extract embedded text
        all_text.append(text.strip())

    combined_native = "\n\n".join(all_text).strip()

    # If we got meaningful text, return it (fast path)
    if len(combined_native) > 100:
        logger.info(f"  ✓ Digital extraction — {page_count} pages, {len(combined_native)} chars")
        doc.close()
        return {
            "text": combined_native,
            "method": "digital",
            "page_count": page_count,
            "filename": filename
        }

    # ── ATTEMPT 2: Scanned PDF — use OCR ──
    # No embedded text found; the PDF is image-only (scanned document)
    logger.info(f"  → No digital text found. Switching to OCR mode...")
    all_text = []

    # Tesseract config:
    #   --oem 3  = LSTM neural net engine (most accurate)
    #   --psm 6  = Assume a uniform block of text (good for contract pages)
    tesseract_config = "--oem 3 --psm 6"

    for page_num in range(page_count):
        page = doc[page_num]

        # Render page to a high-resolution image (300 DPI = good OCR quality)
        mat = fitz.Matrix(300 / 72, 300 / 72)   # 300 DPI scaling
        pix = page.get_pixmap(matrix=mat)

        # Convert PyMuPDF pixmap → PIL Image
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Pre-process the image
        clean_img = preprocess_image(pil_img)

        # Run Tesseract OCR
        page_text = pytesseract.image_to_string(clean_img, config=tesseract_config)
        all_text.append(page_text.strip())
        logger.info(f"    Page {page_num + 1}/{page_count} OCR'd — {len(page_text)} chars")

    doc.close()
    combined_ocr = "\n\n".join(all_text).strip()

    logger.info(f"  ✓ OCR complete — {page_count} pages, {len(combined_ocr)} chars")
    return {
        "text": combined_ocr,
        "method": "ocr",
        "page_count": page_count,
        "filename": filename
    }


# ─────────────────────────────────────────────
# UTILITY: Process a folder of PDFs
# ─────────────────────────────────────────────
def batch_extract(pdf_folder: str, output_folder: str) -> list:
    """
    Process all PDFs in a folder.
    Saves each extracted text as a .txt file.
    Returns list of result dicts.
    """
    os.makedirs(output_folder, exist_ok=True)
    results = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            result = extract_text_from_pdf(pdf_path)

            # Save extracted text
            out_file = os.path.join(output_folder, pdf_file.replace(".pdf", ".txt"))
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            results.append(result)
        except Exception as e:
            logger.error(f"  ✗ Failed on {pdf_file}: {e}")

    logger.info(f"Batch complete. {len(results)}/{len(pdf_files)} PDFs processed.")
    return results


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your actual PDF path to test
    TEST_PDF = "data/raw_pdfs/sample_contract.pdf"

    if os.path.exists(TEST_PDF):
        result = extract_text_from_pdf(TEST_PDF)
        print("\n" + "=" * 60)
        print(f"File    : {result['filename']}")
        print(f"Method  : {result['method']}")
        print(f"Pages   : {result['page_count']}")
        print(f"Chars   : {len(result['text'])}")
        print("=" * 60)
        print("PREVIEW (first 500 chars):")
        print(result["text"][:500])
    else:
        print(f"Test PDF not found at: {TEST_PDF}")
        print("Place any PDF at that path and re-run.")
