from pydantic import BaseModel
from typing import Optional


class InvoiceLineItem(BaseModel):
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class InvoiceData(BaseModel):
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    line_items: list[InvoiceLineItem] = []


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str


class OcrMetadata(BaseModel):
    """OCR quality metadata returned with processing results."""
    confidence: Optional[float] = None
    profile_used: Optional[str] = None
    psm_mode: Optional[str] = None
    quality_score: Optional[float] = None
    word_count: Optional[int] = None
    passes_run: Optional[int] = None
    merge_applied: Optional[bool] = None
    all_scores: Optional[dict] = None
    low_confidence_words: Optional[list] = None
    extraction_method: Optional[str] = None  # "tesseract", "claude_vision", or "mixed (vision+tesseract)"
    handwriting_detected: Optional[bool] = None
    yolo_detection: Optional[dict] = None  # YOLO region detection details


class ProcessingResult(BaseModel):
    job_id: str
    filename: str
    status: str
    doc_type: str
    raw_text: str
    structured_data: dict
    pages_processed: int
    processing_time_seconds: float
    created_at: str
    ocr_metadata: Optional[OcrMetadata] = None
    email_report: Optional[dict] = None
    preview: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
