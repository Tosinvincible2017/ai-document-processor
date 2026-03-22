import logging
import csv
import io
import os
from fastapi import APIRouter, UploadFile, Query, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import anthropic

from models import UploadResponse, ProcessingResult, ErrorResponse
from pipeline import DocumentPipeline
from storage import load_result, list_results as list_all_results, delete_result as remove_result
from database import (
    get_document, list_documents, delete_document,
    get_stats, export_all, insert_document,
)
from email_service import send_report_email, is_email_configured
from preview_service import get_preview_path, get_thumbnail_path, get_preview_page_count, delete_previews
from config import ANTHROPIC_API_KEY, TESSERACT_CMD

logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────
# Upload endpoints
# ──────────────────────────────────────────────

@router.post("/upload", response_model=ProcessingResult, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_document(
    file: UploadFile,
    doc_type: str = Query(None, description="Force document type: 'invoice' or 'general'"),
    email_to: str = Query(None, description="Email address to send the report to"),
):
    """Upload a PDF or image for processing. Optionally email the report."""
    file_bytes = await file.read()
    filename = file.filename or "unknown"

    pipeline = DocumentPipeline(progress_callback=lambda msg: logger.info(msg))

    try:
        result = pipeline.process(filename, file_bytes, doc_type_hint=doc_type, file_size=len(file_bytes))

        # Send email report if requested
        if email_to:
            email_result = send_report_email(result, email_to)
            result["email_report"] = email_result
            if email_result["sent"]:
                logger.info(f"Report emailed to {email_to}")
            else:
                logger.warning(f"Email failed: {email_result['message']}")

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")


@router.post("/upload/async", response_model=UploadResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    doc_type: str = Query(None, description="Force document type: 'invoice' or 'general'"),
    email_to: str = Query(None, description="Email address to send the report to"),
):
    """Upload a document for background processing. Optionally email when done."""
    file_bytes = await file.read()
    filename = file.filename or "unknown"

    from preprocessor import validate_file
    ok, error = validate_file(filename, file_bytes)
    if not ok:
        raise HTTPException(status_code=400, detail=error)

    import uuid
    job_id = uuid.uuid4().hex[:12]

    def _process_background():
        pipeline = DocumentPipeline(progress_callback=lambda msg: logger.info(msg))
        try:
            result = pipeline.process(filename, file_bytes, doc_type_hint=doc_type, file_size=len(file_bytes))

            # Send email report if requested
            if email_to:
                email_result = send_report_email(result, email_to)
                if email_result["sent"]:
                    logger.info(f"Background report emailed to {email_to}")
                else:
                    logger.warning(f"Background email failed: {email_result['message']}")

        except Exception as e:
            logger.error(f"Background processing failed for {filename}: {e}")
            from storage import save_result
            from datetime import datetime, timezone
            error_result = {
                "job_id": job_id,
                "filename": filename,
                "status": "failed",
                "doc_type": "unknown",
                "raw_text": "",
                "structured_data": {"error": str(e)},
                "pages_processed": 0,
                "processing_time_seconds": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            save_result(job_id, error_result)
            try:
                insert_document(error_result, file_size_bytes=len(file_bytes))
            except Exception:
                pass

    background_tasks.add_task(_process_background)

    return UploadResponse(
        job_id=job_id,
        filename=filename,
        status="processing",
        message="Document accepted for processing. Poll /api/results/{job_id} for status."
               + (f" Report will be emailed to {email_to} when done." if email_to else ""),
    )


# ──────────────────────────────────────────────
# Send report email for an existing result
# ──────────────────────────────────────────────

@router.post("/results/{job_id}/email")
async def email_result(
    job_id: str,
    email_to: str = Query(..., description="Email address to send the report to"),
):
    """Send (or resend) a report email for an already-processed document."""
    # Find the result
    doc = get_document(job_id)
    if not doc:
        result = load_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Result not found: {job_id}")
        doc = result

    if not is_email_configured():
        raise HTTPException(
            status_code=503,
            detail="Email not configured. Set SMTP_USER and SMTP_PASSWORD in .env"
        )

    email_result = send_report_email(doc, email_to)

    if not email_result["sent"]:
        raise HTTPException(status_code=500, detail=email_result["message"])

    return {
        "sent": True,
        "job_id": job_id,
        "email_to": email_to,
        "message": email_result["message"],
    }


# ──────────────────────────────────────────────
# Results — powered by SQLite
# ──────────────────────────────────────────────

@router.get("/results/{job_id}", response_model=ProcessingResult, responses={404: {"model": ErrorResponse}})
async def get_result(job_id: str):
    """Retrieve a processing result by job ID."""
    doc = get_document(job_id)
    if doc:
        return doc
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result not found: {job_id}")
    return result


@router.get("/results", response_model=list[ProcessingResult])
async def list_results(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all processing results, newest first."""
    docs, _ = list_documents(limit=limit, offset=offset)
    if docs:
        return docs
    return list_all_results(limit=limit, offset=offset)


@router.delete("/results/{job_id}")
async def delete_result(job_id: str):
    """Delete a processing result from both DB and JSON file."""
    db_deleted = delete_document(job_id)
    file_deleted = remove_result(job_id)
    delete_previews(job_id)
    if not db_deleted and not file_deleted:
        raise HTTPException(status_code=404, detail=f"Result not found: {job_id}")
    return {"deleted": True, "job_id": job_id}


# ──────────────────────────────────────────────
# Database-powered search & filter
# ──────────────────────────────────────────────

@router.get("/db/search")
async def search_documents(
    q: str = Query(None, description="Full-text search across filename, vendor, text, invoice number"),
    doc_type: str = Query(None, description="Filter by doc_type: 'invoice' or 'general'"),
    status: str = Query(None, description="Filter by status: 'completed' or 'failed'"),
    vendor: str = Query(None, description="Filter by vendor name (partial match)"),
    date_from: str = Query(None, description="Filter created_at >= date (ISO format)"),
    date_to: str = Query(None, description="Filter created_at <= date (ISO format)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="asc or desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Search and filter documents with pagination."""
    docs, total = list_documents(
        limit=limit, offset=offset,
        doc_type=doc_type, status=status, vendor=vendor,
        search=q, date_from=date_from, date_to=date_to,
        sort_by=sort_by, sort_order=sort_order,
    )
    return {
        "results": docs,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
    }


@router.get("/db/stats")
async def database_stats():
    """Get aggregate statistics from the database."""
    return get_stats()


@router.get("/db/export/json")
async def export_json(
    doc_type: str = Query(None, description="Filter by doc_type"),
):
    """Export all documents as a JSON download."""
    import json
    docs = export_all(doc_type=doc_type)
    content = json.dumps(docs, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=doc_processor_export.json"},
    )


@router.get("/db/export/csv")
async def export_csv(
    doc_type: str = Query(None, description="Filter by doc_type"),
):
    """Export all documents as a CSV download."""
    docs = export_all(doc_type=doc_type)

    output = io.StringIO()
    if not docs:
        output.write("No data\n")
    else:
        fields = ["job_id", "filename", "doc_type", "status", "pages_processed",
                   "processing_time_seconds", "vendor", "invoice_number",
                   "invoice_date", "due_date", "total_amount", "currency", "created_at"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for doc in docs:
            writer.writerow(doc)

    content = output.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=doc_processor_export.csv"},
    )


# ──────────────────────────────────────────────
# Document preview
# ──────────────────────────────────────────────

@router.get("/preview/{job_id}")
async def get_preview(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number (for multi-page PDFs)"),
):
    """Get document preview image (full size)."""
    from fastapi.responses import FileResponse
    path = get_preview_path(job_id, page)
    if not path:
        raise HTTPException(status_code=404, detail=f"Preview not found for {job_id} page {page}")
    return FileResponse(path, media_type="image/png")


@router.get("/preview/{job_id}/thumb")
async def get_thumbnail(job_id: str):
    """Get document thumbnail (small, for sidebar/cards)."""
    from fastapi.responses import FileResponse
    path = get_thumbnail_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"Thumbnail not found for {job_id}")
    return FileResponse(path, media_type="image/png")


@router.get("/preview/{job_id}/info")
async def get_preview_info(job_id: str):
    """Get preview metadata (page count, availability)."""
    count = get_preview_page_count(job_id)
    return {
        "job_id": job_id,
        "preview_available": count > 0,
        "pages": count,
        "urls": [f"/api/preview/{job_id}?page={i+1}" for i in range(count)],
        "thumbnail_url": f"/api/preview/{job_id}/thumb" if count > 0 else None,
    }


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Check service health and dependencies."""
    import shutil
    from database import DB_PATH
    tesseract_ok = bool(shutil.which("tesseract") or TESSERACT_CMD)
    api_key_set = bool(ANTHROPIC_API_KEY)
    db_exists = os.path.exists(DB_PATH)

    return {
        "status": "ok" if (tesseract_ok and api_key_set) else "degraded",
        "tesseract_available": tesseract_ok,
        "api_key_set": api_key_set,
        "database": "connected" if db_exists else "missing",
        "email_configured": is_email_configured(),
    }
