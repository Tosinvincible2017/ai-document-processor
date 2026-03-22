# AI Document Processor

A production-ready document processing system built with FastAPI, Tesseract OCR, YOLO object detection, and Claude Vision API. Handles PDFs and images with smart handwriting detection and mixed-document routing.

## Features

- **Multi-engine OCR** — Tesseract for printed text, Claude Vision API for handwriting
- **YOLO region detection** — Identifies handwritten vs printed regions in mixed documents
- **Smart routing** — Automatically routes each region to the best OCR engine
- **Document classification** — Classifies documents as invoices, receipts, reports, etc.
- **Structured extraction** — Uses Claude API to extract structured JSON from raw text
- **PWA mobile app** — Installable on Android/iOS with offline support
- **Document preview** — In-browser preview for uploaded PDFs and images
- **Email notifications** — Sends processing results via email
- **SQLite database** — Persistent storage with search and pagination

## Setup

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `ANTHROPIC_API_KEY` — Your Anthropic API key
- `TESSERACT_CMD` — Path to Tesseract executable (Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

Optional:
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` — For email notifications

### Run

```bash
python main.py
```

The server starts at `https://localhost:8000`. Open in your browser to use the web UI.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/upload` | Upload & process a document |
| `POST` | `/api/upload/async` | Process in background, returns job_id |
| `GET` | `/api/results/{job_id}` | Get processing result |
| `GET` | `/api/results` | List all results |
| `GET` | `/api/preview/{job_id}` | Preview document pages |
| `GET` | `/api/health` | Health check |

## Architecture

```
Upload → Validate → Preprocess → YOLO Region Detection
  ├─ Handwritten regions → Claude Vision API
  └─ Printed regions → Tesseract OCR
→ Merge text → Claude classification → Structured JSON → Store
```

## Mobile Installation

The app is a Progressive Web App (PWA):
- **Android**: Open in Chrome → Menu → "Install app"
- **iOS**: Open in Safari → Share → "Add to Home Screen"
