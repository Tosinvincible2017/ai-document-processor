"""
SQLite database layer for persisting processed documents.

Tables:
    documents  – one row per processed document (metadata + raw text + structured JSON)
    line_items – invoice line items (FK to documents)
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(BASE_DIR, "doc_processor.db")

# ──────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────

@contextmanager
def get_db():
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT    UNIQUE NOT NULL,
                filename        TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'completed',
                doc_type        TEXT    NOT NULL DEFAULT 'general',
                raw_text        TEXT    NOT NULL DEFAULT '',
                structured_data TEXT    NOT NULL DEFAULT '{}',
                pages_processed INTEGER NOT NULL DEFAULT 0,
                processing_time REAL    NOT NULL DEFAULT 0.0,
                file_size_bytes INTEGER DEFAULT 0,
                created_at      TEXT    NOT NULL,

                -- Searchable invoice fields (denormalized for fast queries)
                vendor          TEXT,
                invoice_number  TEXT,
                invoice_date    TEXT,
                due_date        TEXT,
                total_amount    REAL,
                currency        TEXT
            );

            CREATE TABLE IF NOT EXISTS line_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                description TEXT    NOT NULL DEFAULT '',
                quantity    REAL,
                unit_price  REAL,
                amount      REAL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_documents_job_id     ON documents(job_id);
            CREATE INDEX IF NOT EXISTS idx_documents_doc_type   ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_documents_vendor     ON documents(vendor);
            CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
            CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_line_items_doc_id    ON line_items(document_id);
        """)
    logger.info(f"Database initialized: {DB_PATH}")


# ──────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────

def insert_document(result: dict, file_size_bytes: int = 0) -> int:
    """
    Insert a processed document into the database.
    Returns the row id.
    """
    sd = result.get("structured_data", {})

    # Extract denormalized invoice fields
    vendor = sd.get("vendor")
    invoice_number = sd.get("invoice_number")
    invoice_date = sd.get("date")
    due_date = sd.get("due_date")
    total_amount = sd.get("total")
    currency = sd.get("currency")

    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO documents
                (job_id, filename, status, doc_type, raw_text, structured_data,
                 pages_processed, processing_time, file_size_bytes, created_at,
                 vendor, invoice_number, invoice_date, due_date, total_amount, currency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["job_id"],
            result["filename"],
            result.get("status", "completed"),
            result.get("doc_type", "general"),
            result.get("raw_text", ""),
            json.dumps(sd, ensure_ascii=False),
            result.get("pages_processed", 0),
            result.get("processing_time_seconds", 0.0),
            file_size_bytes,
            result.get("created_at", datetime.now(timezone.utc).isoformat()),
            vendor,
            invoice_number,
            invoice_date,
            due_date,
            total_amount,
            currency,
        ))

        doc_id = cur.lastrowid

        # Insert line items if invoice
        line_items = sd.get("line_items", [])
        if line_items:
            conn.executemany("""
                INSERT INTO line_items (document_id, description, quantity, unit_price, amount)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (doc_id, li.get("description", ""), li.get("quantity"), li.get("unit_price"), li.get("amount"))
                for li in line_items
            ])

    logger.info(f"Inserted document {result['job_id']} (id={doc_id}, {len(line_items)} line items)")
    return doc_id


def get_document(job_id: str) -> dict | None:
    """Fetch a single document by job_id."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM documents WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        doc = _row_to_dict(row)

        # Attach line items
        items = conn.execute(
            "SELECT * FROM line_items WHERE document_id = ? ORDER BY id", (row["id"],)
        ).fetchall()
        doc["line_items"] = [dict(i) for i in items]

        return doc


def list_documents(
    limit: int = 50,
    offset: int = 0,
    doc_type: str = None,
    status: str = None,
    vendor: str = None,
    search: str = None,
    date_from: str = None,
    date_to: str = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> tuple[list[dict], int]:
    """
    List documents with filtering, search, and pagination.
    Returns (results, total_count).
    """
    allowed_sort = {"created_at", "filename", "doc_type", "total_amount", "vendor", "processing_time"}
    if sort_by not in allowed_sort:
        sort_by = "created_at"
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    where_clauses = []
    params = []

    if doc_type:
        where_clauses.append("doc_type = ?")
        params.append(doc_type)
    if status:
        where_clauses.append("status = ?")
        params.append(status)
    if vendor:
        where_clauses.append("vendor LIKE ?")
        params.append(f"%{vendor}%")
    if search:
        where_clauses.append("(filename LIKE ? OR vendor LIKE ? OR raw_text LIKE ? OR invoice_number LIKE ?)")
        params.extend([f"%{search}%"] * 4)
    if date_from:
        where_clauses.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("created_at <= ?")
        params.append(date_to)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    with get_db() as conn:
        # Total count
        count = conn.execute(f"SELECT COUNT(*) FROM documents WHERE {where_sql}", params).fetchone()[0]

        # Results
        rows = conn.execute(
            f"SELECT * FROM documents WHERE {where_sql} ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

        results = [_row_to_dict(r) for r in rows]

    return results, count


def delete_document(job_id: str) -> bool:
    """Delete a document and its line items. Returns True if found."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM documents WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0


def get_stats() -> dict:
    """Get aggregate stats from the database."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        invoices = conn.execute("SELECT COUNT(*) FROM documents WHERE doc_type = 'invoice'").fetchone()[0]
        general = conn.execute("SELECT COUNT(*) FROM documents WHERE doc_type = 'general'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'failed'").fetchone()[0]

        total_amount = conn.execute(
            "SELECT COALESCE(SUM(total_amount), 0) FROM documents WHERE doc_type = 'invoice' AND total_amount IS NOT NULL"
        ).fetchone()[0]

        avg_time = conn.execute(
            "SELECT COALESCE(AVG(processing_time), 0) FROM documents WHERE status = 'completed'"
        ).fetchone()[0]

        recent = conn.execute(
            "SELECT vendor, total_amount, currency, created_at FROM documents WHERE doc_type = 'invoice' AND vendor IS NOT NULL ORDER BY created_at DESC LIMIT 5"
        ).fetchall()

        vendors = conn.execute(
            "SELECT vendor, COUNT(*) as cnt, SUM(total_amount) as total FROM documents WHERE vendor IS NOT NULL GROUP BY vendor ORDER BY cnt DESC LIMIT 10"
        ).fetchall()

    return {
        "total_documents": total,
        "invoices": invoices,
        "general": general,
        "failed": failed,
        "total_invoice_amount": round(total_amount, 2),
        "avg_processing_time": round(avg_time, 2),
        "recent_invoices": [dict(r) for r in recent],
        "top_vendors": [dict(r) for r in vendors],
    }


def export_all(doc_type: str = None) -> list[dict]:
    """Export all documents (for CSV/JSON export)."""
    with get_db() as conn:
        if doc_type:
            rows = conn.execute(
                "SELECT * FROM documents WHERE doc_type = ? ORDER BY created_at DESC", (doc_type,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a clean dict matching the API response format."""
    d = dict(row)
    # Parse structured_data back to dict
    try:
        d["structured_data"] = json.loads(d.get("structured_data", "{}"))
    except (json.JSONDecodeError, TypeError):
        d["structured_data"] = {}

    # Rename for API compatibility
    d["processing_time_seconds"] = d.pop("processing_time", 0.0)

    # Remove internal fields from API responses
    d.pop("id", None)

    return d
