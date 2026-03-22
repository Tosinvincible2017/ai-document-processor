"""
Document processing pipeline — V2.

Orchestrates: validate -> save -> OCR -> classify -> parse -> store

V2 improvements:
- OCR metadata (confidence, profile, scores) passed through entire pipeline
- OCR confidence included in API responses
- Better error context for debugging
"""

import os
import uuid
import time
import logging
from datetime import datetime, timezone

from preprocessor import validate_file, detect_file_type, save_upload
from ocr_engine import extract_text
from parser import classify_document, parse_invoice, parse_general
from storage import save_result
from database import insert_document
from preview_service import generate_preview

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Orchestrates the full document processing pipeline."""

    def __init__(self, progress_callback=None):
        self.progress = progress_callback or (lambda msg: print(f"[Pipeline] {msg}"))

    def process(self, filename: str, file_bytes: bytes,
                doc_type_hint: str = None, file_size: int = 0) -> dict:
        """
        Full pipeline: validate -> save -> OCR -> classify -> parse -> store.

        Returns ProcessingResult as dict including OCR metadata.
        """
        job_id = uuid.uuid4().hex[:12]
        start_time = time.time()
        file_path = None

        try:
            # Step 1: Validate
            self.progress(f"[{job_id}] Validating {filename}")
            ok, error = validate_file(filename, file_bytes)
            if not ok:
                raise ValueError(error)

            # Step 2: Save upload
            self.progress(f"[{job_id}] Saving upload")
            file_path = save_upload(filename, file_bytes)

            # Step 3: Detect file type
            file_type = detect_file_type(filename)
            self.progress(f"[{job_id}] File type: {file_type}")

            # Step 3.5: Generate preview (before OCR, while file exists)
            self.progress(f"[{job_id}] Generating preview...")
            try:
                preview_info = generate_preview(file_path, job_id, file_type)
                if preview_info.get("preview_available"):
                    self.progress(f"[{job_id}] Preview generated ({preview_info.get('total_pages', 1)} page(s))")
            except Exception as prev_err:
                logger.warning(f"Preview generation failed for {job_id}: {prev_err}")
                preview_info = {"preview_available": False}

            # Step 4: OCR / text extraction (V2: returns ocr_metadata)
            self.progress(f"[{job_id}] Extracting text (multi-pass OCR)...")
            raw_text, page_count, ocr_metadata = extract_text(file_path, file_type)
            self.progress(f"[{job_id}] Extracted {len(raw_text)} chars from {page_count} page(s)")

            if ocr_metadata:
                conf = ocr_metadata.get("confidence", ocr_metadata.get("avg_confidence", "N/A"))
                profile = ocr_metadata.get("profile", "N/A")
                self.progress(f"[{job_id}] OCR confidence: {conf}%, profile: {profile}")

            # Step 5: Classify document
            if doc_type_hint and doc_type_hint in ("invoice", "general"):
                doc_type = doc_type_hint
                self.progress(f"[{job_id}] Using forced doc type: {doc_type}")
            else:
                self.progress(f"[{job_id}] Classifying document...")
                doc_type = classify_document(raw_text)
                self.progress(f"[{job_id}] Classified as: {doc_type}")

            # Step 6: Parse / structure
            self.progress(f"[{job_id}] Parsing with Claude API...")
            if doc_type == "invoice":
                structured_data = parse_invoice(raw_text)
            else:
                structured_data = parse_general(raw_text)

            # Step 7: Build result
            elapsed = round(time.time() - start_time, 2)
            result = {
                "job_id": job_id,
                "filename": filename,
                "status": "completed",
                "doc_type": doc_type,
                "raw_text": raw_text,
                "structured_data": structured_data,
                "pages_processed": page_count,
                "processing_time_seconds": elapsed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Include preview info
            if preview_info.get("preview_available"):
                result["preview"] = {
                    "available": True,
                    "pages": preview_info.get("total_pages", 1),
                    "url": f"/api/preview/{job_id}",
                    "thumbnail_url": f"/api/preview/{job_id}/thumb",
                }

            # V2: Include OCR metadata in result
            if ocr_metadata:
                extraction_method = ocr_metadata.get("extraction_method", "tesseract")
                handwriting_detected = ocr_metadata.get("handwriting_detected", False)

                if handwriting_detected:
                    self.progress(f"[{job_id}] Handwriting detected — used {extraction_method}")

                result["ocr_metadata"] = {
                    "confidence": ocr_metadata.get("confidence",
                                                    ocr_metadata.get("avg_confidence")),
                    "profile_used": ocr_metadata.get("profile", "native"),
                    "psm_mode": ocr_metadata.get("psm"),
                    "quality_score": ocr_metadata.get("score"),
                    "word_count": ocr_metadata.get("word_count"),
                    "passes_run": ocr_metadata.get("passes_run"),
                    "merge_applied": ocr_metadata.get("merge_applied", False),
                    "all_scores": ocr_metadata.get("all_scores"),
                    "extraction_method": extraction_method,
                    "handwriting_detected": handwriting_detected,
                }

                # Include YOLO detection details if available
                yolo_info = ocr_metadata.get("yolo_detection")
                if yolo_info:
                    result["ocr_metadata"]["yolo_detection"] = yolo_info
                    self.progress(f"[{job_id}] YOLO regions: "
                                  f"{yolo_info.get('handwritten_regions', 0)} handwritten, "
                                  f"{yolo_info.get('printed_regions', 0)} printed")
                # Include low-confidence words for transparency
                low_conf = ocr_metadata.get("low_confidence_words", [])
                if low_conf:
                    result["ocr_metadata"]["low_confidence_words"] = [
                        {"word": w, "confidence": c} for w, c in low_conf[:10]
                    ]

            # Step 8: Save result (JSON file + SQLite database)
            save_result(job_id, result)
            try:
                insert_document(result, file_size_bytes=file_size or len(file_bytes))
                self.progress(f"[{job_id}] Saved to database")
            except Exception as db_err:
                logger.warning(f"DB insert failed for {job_id}: {db_err}")
            self.progress(f"[{job_id}] Done in {elapsed}s")

            return result

        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Pipeline error for {filename}: {e}")
            raise RuntimeError(f"Processing failed: {e}")
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except OSError:
                    pass
