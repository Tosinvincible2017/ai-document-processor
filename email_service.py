"""
Email report service for Document Processor.

Sends beautifully formatted HTML email reports with:
- Document summary (for general docs)
- Invoice breakdown (for invoices)
- OCR quality info
- JSON attachment of full structured data
"""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formataddr
from datetime import datetime

from config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM_NAME, SMTP_USE_TLS,
)

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check if SMTP credentials are set."""
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def send_report_email(result: dict, recipient_email: str) -> dict:
    """
    Send a document processing report via email.

    Args:
        result: The full processing result dict
        recipient_email: Email address to send to

    Returns:
        {"sent": True/False, "message": "..."}
    """
    if not is_email_configured():
        return {
            "sent": False,
            "message": "Email not configured. Set SMTP_USER and SMTP_PASSWORD in .env"
        }

    if not recipient_email or "@" not in recipient_email:
        return {"sent": False, "message": f"Invalid email address: {recipient_email}"}

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = formataddr((SMTP_FROM_NAME, SMTP_USER))
        msg["To"] = recipient_email
        msg["Subject"] = _build_subject(result)

        # Plain text fallback
        plain = _build_plain_text(result)
        msg.attach(MIMEText(plain, "plain", "utf-8"))

        # Rich HTML email
        html = _build_html_email(result)
        msg.attach(MIMEText(html, "html", "utf-8"))

        # Attach full JSON result
        json_data = json.dumps(result, indent=2, ensure_ascii=False)
        attachment = MIMEApplication(json_data.encode("utf-8"), _subtype="json")
        safe_name = result.get("filename", "document").replace(" ", "_")
        attachment.add_header(
            "Content-Disposition", "attachment",
            filename=f"{safe_name}_report.json"
        )
        msg.attach(attachment)

        # Send
        logger.info(f"Sending report email to {recipient_email} via {SMTP_HOST}:{SMTP_PORT}")

        if SMTP_USE_TLS:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info(f"Report email sent successfully to {recipient_email}")
        return {"sent": True, "message": f"Report sent to {recipient_email}"}

    except smtplib.SMTPAuthenticationError:
        error = "SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD in .env"
        logger.error(error)
        return {"sent": False, "message": error}
    except smtplib.SMTPException as e:
        error = f"SMTP error: {e}"
        logger.error(error)
        return {"sent": False, "message": error}
    except Exception as e:
        error = f"Failed to send email: {e}"
        logger.error(error)
        return {"sent": False, "message": error}


# ──────────────────────────────────────────────
# Email content builders
# ──────────────────────────────────────────────

def _build_subject(result: dict) -> str:
    """Build email subject line."""
    filename = result.get("filename", "Document")
    doc_type = result.get("doc_type", "general").title()
    sd = result.get("structured_data", {})

    if result.get("doc_type") == "invoice":
        vendor = sd.get("vendor", "Unknown")
        total = sd.get("total")
        currency = sd.get("currency", "")
        if total:
            return f"Document Report: Invoice from {vendor} - {currency} {total:,.2f}"
        return f"Document Report: Invoice from {vendor}"
    else:
        title = sd.get("title", filename)
        return f"Document Report: {title}"


def _build_plain_text(result: dict) -> str:
    """Build plain-text email body."""
    lines = []
    lines.append("DOCUMENT PROCESSING REPORT")
    lines.append("=" * 40)
    lines.append(f"File: {result.get('filename', 'Unknown')}")
    lines.append(f"Type: {result.get('doc_type', 'N/A')}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Pages: {result.get('pages_processed', 0)}")
    lines.append(f"Processing Time: {result.get('processing_time_seconds', 0)}s")
    lines.append("")

    sd = result.get("structured_data", {})

    if result.get("doc_type") == "invoice":
        lines.append("INVOICE DETAILS")
        lines.append("-" * 30)
        lines.append(f"Vendor: {sd.get('vendor', 'N/A')}")
        lines.append(f"Invoice #: {sd.get('invoice_number', 'N/A')}")
        lines.append(f"Date: {sd.get('date', 'N/A')}")
        lines.append(f"Total: {sd.get('currency', '')} {sd.get('total', 'N/A')}")
        items = sd.get("line_items", [])
        if items:
            lines.append(f"\nLine Items ({len(items)}):")
            for li in items:
                lines.append(f"  - {li.get('description', '?')}: {li.get('amount', '?')}")
    else:
        lines.append("DOCUMENT SUMMARY")
        lines.append("-" * 30)
        lines.append(f"Title: {sd.get('title', 'N/A')}")
        lines.append(f"Type: {sd.get('document_type', 'N/A')}")
        summary = sd.get("summary", "No summary available.")
        lines.append(f"\n{summary}")

        key_fields = sd.get("key_fields", {})
        if key_fields:
            lines.append("\nKey Fields:")
            for k, v in key_fields.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for sk, sv in v.items():
                        lines.append(f"    {sk}: {sv}")
                else:
                    lines.append(f"  {k}: {v}")

        entities = sd.get("entities", [])
        if entities:
            lines.append(f"\nEntities: {', '.join(str(e) for e in entities)}")

    lines.append("\n---")
    lines.append("Full JSON data is attached to this email.")
    lines.append("Generated by Document Processor")

    return "\n".join(lines)


def _build_html_email(result: dict) -> str:
    """Build rich HTML email with styling."""
    sd = result.get("structured_data", {})
    doc_type = result.get("doc_type", "general")
    ocr_meta = result.get("ocr_metadata", {})

    # Confidence badge
    conf = None
    if ocr_meta:
        conf = ocr_meta.get("confidence")

    conf_html = ""
    if conf is not None:
        conf_color = "#22c55e" if conf >= 80 else "#f59e0b" if conf >= 50 else "#ef4444"
        conf_html = f"""
        <div style="display:inline-block;padding:4px 12px;border-radius:12px;
                    background:{conf_color}15;color:{conf_color};font-size:13px;
                    border:1px solid {conf_color}40;margin-left:8px;">
            OCR Confidence: {round(conf)}%
        </div>"""

    # Type badge
    type_color = "#6c63ff" if doc_type == "invoice" else "#3b82f6"
    type_label = doc_type.title()

    # Build body sections
    if doc_type == "invoice":
        body_html = _build_invoice_html(sd)
    else:
        body_html = _build_general_html(sd)

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f4f4f7;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:640px;margin:0 auto;padding:24px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#1a1d27 0%,#2e3348 100%);
                border-radius:12px 12px 0 0;padding:32px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;font-weight:600;">
            Document Processing Report
        </h1>
        <p style="color:#8b8fa8;margin:8px 0 0;font-size:14px;">
            {result.get('filename', 'Unknown Document')}
        </p>
    </div>

    <!-- Meta bar -->
    <div style="background:#242837;padding:16px 32px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="display:inline-block;padding:4px 12px;border-radius:12px;
                     background:{type_color}20;color:{type_color};font-size:13px;
                     border:1px solid {type_color}40;">
            {type_label}
        </span>
        <span style="display:inline-block;padding:4px 12px;border-radius:12px;
                     background:#22c55e15;color:#22c55e;font-size:13px;
                     border:1px solid #22c55e40;">
            Completed
        </span>
        {conf_html}
        <span style="color:#8b8fa8;font-size:12px;margin-left:auto;">
            {result.get('pages_processed', 0)} page(s) &middot;
            {result.get('processing_time_seconds', 0)}s &middot;
            {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </span>
    </div>

    <!-- Body -->
    <div style="background:#fff;padding:32px;border-radius:0 0 12px 12px;
                border:1px solid #e5e7eb;border-top:none;">
        {body_html}

        <!-- Footer -->
        <div style="margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;
                    text-align:center;color:#9ca3af;font-size:12px;">
            <p>Full JSON data is attached to this email.</p>
            <p style="margin-top:4px;">Generated by <strong>Document Processor</strong></p>
        </div>
    </div>
</div>
</body>
</html>"""

    return html


def _build_invoice_html(sd: dict) -> str:
    """Build HTML section for invoice results."""
    currency = sd.get("currency", "")

    def fmt_money(val):
        if val is None:
            return "N/A"
        try:
            return f"{currency} {float(val):,.2f}"
        except (ValueError, TypeError):
            return str(val)

    html = """
    <h2 style="color:#1f2937;margin:0 0 20px;font-size:18px;">Invoice Details</h2>

    <table style="width:100%;border-collapse:collapse;margin-bottom:24px;">
    """

    fields = [
        ("Vendor", sd.get("vendor")),
        ("Invoice #", sd.get("invoice_number")),
        ("Date", sd.get("date")),
        ("Due Date", sd.get("due_date")),
        ("Currency", currency),
    ]

    for label, val in fields:
        display = val if val else '<span style="color:#d1d5db">N/A</span>'
        html += f"""
        <tr>
            <td style="padding:8px 12px;color:#6b7280;font-size:14px;width:120px;
                       border-bottom:1px solid #f3f4f6;">{label}</td>
            <td style="padding:8px 12px;color:#1f2937;font-size:14px;font-weight:500;
                       border-bottom:1px solid #f3f4f6;">{display}</td>
        </tr>"""

    html += "</table>"

    # Totals
    html += """
    <div style="background:#f8fafc;border-radius:8px;padding:16px;margin-bottom:24px;">
        <table style="width:100%;border-collapse:collapse;">"""

    for label, key in [("Subtotal", "subtotal"), ("Tax", "tax")]:
        html += f"""
        <tr>
            <td style="padding:6px 0;color:#6b7280;font-size:14px;">{label}</td>
            <td style="padding:6px 0;color:#1f2937;font-size:14px;text-align:right;">
                {fmt_money(sd.get(key))}
            </td>
        </tr>"""

    html += f"""
        <tr>
            <td style="padding:10px 0 6px;color:#1f2937;font-size:16px;font-weight:700;
                       border-top:2px solid #e5e7eb;">Total</td>
            <td style="padding:10px 0 6px;color:#6c63ff;font-size:18px;font-weight:700;
                       text-align:right;border-top:2px solid #e5e7eb;">
                {fmt_money(sd.get("total"))}
            </td>
        </tr>
    </table></div>"""

    # Line items
    items = sd.get("line_items", [])
    if items:
        html += f"""
        <h3 style="color:#1f2937;margin:0 0 12px;font-size:15px;">
            Line Items ({len(items)})
        </h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
                <tr style="background:#f8fafc;">
                    <th style="padding:8px;text-align:left;color:#6b7280;border-bottom:2px solid #e5e7eb;">Description</th>
                    <th style="padding:8px;text-align:center;color:#6b7280;border-bottom:2px solid #e5e7eb;">Qty</th>
                    <th style="padding:8px;text-align:right;color:#6b7280;border-bottom:2px solid #e5e7eb;">Unit Price</th>
                    <th style="padding:8px;text-align:right;color:#6b7280;border-bottom:2px solid #e5e7eb;">Amount</th>
                </tr>
            </thead><tbody>"""

        for li in items:
            html += f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #f3f4f6;">{li.get('description', '')}</td>
                <td style="padding:8px;text-align:center;border-bottom:1px solid #f3f4f6;">{li.get('quantity', '-')}</td>
                <td style="padding:8px;text-align:right;border-bottom:1px solid #f3f4f6;">{fmt_money(li.get('unit_price'))}</td>
                <td style="padding:8px;text-align:right;border-bottom:1px solid #f3f4f6;font-weight:500;">{fmt_money(li.get('amount'))}</td>
            </tr>"""

        html += "</tbody></table>"

    return html


def _build_general_html(sd: dict) -> str:
    """Build HTML section for general document results with rich summary."""
    title = sd.get("title", "Untitled Document")
    doc_type_label = sd.get("document_type", "General Document")
    date = sd.get("date", "N/A")
    author = sd.get("author")
    summary = sd.get("summary", "No summary available.")

    html = f"""
    <h2 style="color:#1f2937;margin:0 0 4px;font-size:20px;">{_esc(title)}</h2>
    <p style="color:#6b7280;margin:0 0 20px;font-size:14px;">
        {_esc(doc_type_label)}
        {f' &middot; {_esc(author)}' if author else ''}
        {f' &middot; {_esc(date)}' if date and date != 'N/A' else ''}
    </p>

    <!-- Summary -->
    <div style="background:linear-gradient(135deg,#eff6ff 0%,#f0fdf4 100%);
                border-radius:8px;padding:20px;margin-bottom:24px;
                border-left:4px solid #3b82f6;">
        <h3 style="color:#1e40af;margin:0 0 8px;font-size:14px;text-transform:uppercase;
                   letter-spacing:0.5px;">Summary</h3>
        <p style="color:#374151;margin:0;font-size:14px;line-height:1.7;">
            {_esc(summary)}
        </p>
    </div>"""

    # Key fields
    key_fields = sd.get("key_fields", {})
    if key_fields:
        html += """
        <h3 style="color:#1f2937;margin:0 0 12px;font-size:15px;">Key Information</h3>
        <table style="width:100%;border-collapse:collapse;margin-bottom:24px;">"""

        for k, v in key_fields.items():
            label = k.replace("_", " ").title()
            if isinstance(v, dict):
                # Nested fields
                html += f"""
                <tr><td colspan="2" style="padding:10px 12px 4px;color:#1f2937;
                         font-weight:600;font-size:14px;border-bottom:1px solid #f3f4f6;">
                    {_esc(label)}</td></tr>"""
                for sk, sv in v.items():
                    sub_label = sk.replace("_", " ").title()
                    html += f"""
                    <tr>
                        <td style="padding:6px 12px 6px 28px;color:#6b7280;font-size:13px;
                                   width:40%;border-bottom:1px solid #f3f4f6;">{_esc(sub_label)}</td>
                        <td style="padding:6px 12px;color:#1f2937;font-size:13px;
                                   border-bottom:1px solid #f3f4f6;">{_esc(str(sv))}</td>
                    </tr>"""
            else:
                html += f"""
                <tr>
                    <td style="padding:8px 12px;color:#6b7280;font-size:14px;width:40%;
                               border-bottom:1px solid #f3f4f6;">{_esc(label)}</td>
                    <td style="padding:8px 12px;color:#1f2937;font-size:14px;
                               border-bottom:1px solid #f3f4f6;">{_esc(str(v))}</td>
                </tr>"""

        html += "</table>"

    # Entities
    entities = sd.get("entities", [])
    if entities:
        html += """<h3 style="color:#1f2937;margin:0 0 8px;font-size:15px;">Entities Found</h3>
        <div style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:6px;">"""
        for e in entities:
            html += f"""<span style="display:inline-block;padding:4px 10px;border-radius:6px;
                                    background:#eff6ff;color:#2563eb;font-size:13px;
                                    border:1px solid #bfdbfe;">{_esc(str(e))}</span>"""
        html += "</div>"

    # Dates
    dates = sd.get("dates", [])
    if dates:
        html += """<h3 style="color:#1f2937;margin:0 0 8px;font-size:15px;">Dates Found</h3>
        <div style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:6px;">"""
        for d in dates:
            html += f"""<span style="display:inline-block;padding:4px 10px;border-radius:6px;
                                    background:#fef3c7;color:#92400e;font-size:13px;
                                    border:1px solid #fde68a;">{_esc(str(d))}</span>"""
        html += "</div>"

    # Amounts
    amounts = sd.get("amounts", [])
    if amounts:
        html += """<h3 style="color:#1f2937;margin:0 0 8px;font-size:15px;">Amounts Found</h3>
        <div style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:6px;">"""
        for a in amounts:
            html += f"""<span style="display:inline-block;padding:4px 10px;border-radius:6px;
                                    background:#f0fdf4;color:#166534;font-size:13px;
                                    border:1px solid #bbf7d0;">{_esc(str(a))}</span>"""
        html += "</div>"

    return html


def _esc(s: str) -> str:
    """HTML-escape a string."""
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))
