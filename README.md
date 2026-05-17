# LexiScan Auto 🔍

**Legal Contract Entity Extractor** — AI-powered NER pipeline for financial law firms.

> Automatically extracts **Dates**, **Parties**, **Dollar Amounts**, and **Termination Clauses** from any PDF contract.

---

## Tech Stack

| Component | Technology |
|---|---|
| OCR | Tesseract OCR + pdf2image + PyMuPDF |
| NER Model | SpaCy `en_core_web_sm` + custom EntityRuler |
| Validation | Regex rule engine + python-dateutil |
| API | FastAPI + Uvicorn |
| Container | Docker (multi-stage) |
| Deployment | Railway (cloud) |

---

## Run Locally

### Option A — Docker (recommended)

```bash
# Clone & build
git clone <your-repo>
cd LexiAI
docker-compose up --build
```

Visit **http://localhost:8000**

### Option B — Native Python

> Requires Tesseract installed: https://github.com/tesseract-ocr/tesseract

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload --port 8000
```

---

## API Reference

### `GET /health`
Returns service status.

### `POST /api/extract`
Extract entities from a contract.

**Body** (`multipart/form-data`):
| Field | Type | Description |
|---|---|---|
| `file` | File | PDF or TXT file (optional if `text` provided) |
| `text` | string | Raw contract text (optional if `file` provided) |

**Response** (`application/json`):
```json
{
  "status": "success",
  "processing_time_ms": 342,
  "ocr_used": false,
  "page_count": 3,
  "total_entities_found": 12,
  "entities": {
    "dates":  [ { "text": "January 15, 2024", "normalized": "2024-01-15", "confidence": 0.91 } ],
    "parties": [ { "text": "Acme Corporation", "confidence": 0.87 } ],
    "amounts": [ { "text": "$2,500,000", "normalized": "USD 2,500,000.00", "currency": "USD", "confidence": 0.94 } ],
    "termination_clauses": [ { "text": "Either party may terminate...", "confidence": 0.88 } ]
  },
  "raw_text_preview": "This Agreement is entered into..."
}
```

### `GET /api/demo`
Run extraction on a built-in sample contract (no upload needed).

---

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Select your repo — Railway auto-detects `railway.toml` and `Dockerfile`
4. Click **Deploy** — wait ~3 minutes for the Docker build
5. Your live URL appears in the Railway dashboard

---

## Train Custom NER Model

```bash
cd backend
python ner/train.py
# Outputs model to backend/models/legal_ner/
# F1 score printed at end of training
```

---

## Project Structure

```
LexiAI/
├── backend/
│   ├── main.py            # FastAPI app
│   ├── ocr/pipeline.py    # Tesseract + PyMuPDF
│   ├── ner/model.py       # SpaCy NER + EntityRuler
│   ├── ner/train.py       # Fine-tuning script
│   ├── validator/rules.py # Rule-based validation
│   └── data/training_data.py
├── frontend/
│   ├── index.html         # Dashboard UI
│   ├── style.css          # Glassmorphism dark theme
│   └── app.js             # API integration
├── Dockerfile
├── docker-compose.yml
└── railway.toml
```

---

*Built by Zaalima Development Pvt. Ltd. — Confidential*
