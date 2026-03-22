"""
NLP parsing layer using Claude API.

Key improvements for noisy/scanned documents:
- Prompts explicitly tell Claude the text may contain OCR errors
- Claude is asked to infer/correct obvious OCR mistakes
- Confidence metadata is passed to guide Claude's behavior
- Retry logic with relaxed prompts on parse failure
"""

import json
import logging
import anthropic
from config import ANTHROPIC_API_KEY, MODEL

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _chat(prompt: str, system: str, max_tokens: int = 4096) -> str:
    """Call Claude API and return raw text with code fences stripped."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text

    # Strip code fences if Claude wrapped the JSON
    if text.strip().startswith("```"):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def _parse_json_response(text: str) -> dict:
    """Parse JSON from Claude response with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object in the response
    # Sometimes Claude adds text before/after the JSON
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try fixing common JSON issues
    fixed = text.strip()
    # Fix trailing commas
    fixed = fixed.replace(',}', '}').replace(',]', ']')
    # Fix single quotes
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    logger.warning(f"JSON parse failed after all strategies. Raw: {text[:500]}")
    return {"parse_error": True, "raw": text}


def classify_document(raw_text: str) -> str:
    """Classify document type using first 800 chars. Returns 'invoice' or 'general'."""
    system = (
        "You are a document classifier. The text below was extracted via OCR and may contain "
        "errors, misspellings, or garbled characters. It could be printed or handwritten text. "
        "Look past the noise to determine the document type. "
        "Return ONLY one word: invoice, receipt, letter, notes, contract, or general."
    )
    prompt = f"Classify this document (note: text may have OCR errors, possibly handwritten):\n\n{raw_text[:800]}"

    try:
        result = _chat(prompt, system, max_tokens=50)
        doc_type = result.strip().lower().split()[0] if result.strip() else "general"
        if doc_type in ("invoice", "receipt"):
            return "invoice"
        return "general"
    except Exception as e:
        logger.warning(f"Classification failed: {e}, defaulting to 'general'")
        return "general"


# ──────────────────────────────────────────────
# System prompts — OCR-error-aware
# ──────────────────────────────────────────────

INVOICE_SYSTEM = """You are an expert document data extraction specialist.
You are processing text extracted via OCR from a scanned, photographed, or handwritten document.

CRITICAL INSTRUCTIONS:
1. The text WILL contain OCR errors — misspelled words, garbled characters, wrong numbers.
   Use context to infer the correct values. For handwritten invoices, numbers and amounts
   may be particularly error-prone — cross-verify totals carefully.
2. Common OCR mistakes to watch for:
   - O/0 confusion (letter O vs zero): "1O,OOO" → "10,000"
   - l/1/I confusion: "lnvoice" → "Invoice", "1tem" → "Item"
   - S/5, B/8, Z/2 confusion in numbers
   - Merged or split words: "to tal" → "total", "amountdue" → "amount due"
   - Missing decimals: "10000" might be "100.00" based on context
   - Currency symbols garbled: use context to determine currency
3. For amounts, cross-verify: line items should sum to subtotal, subtotal + tax = total
4. If a field is truly unreadable, use null — don't guess randomly
5. Return ONLY valid JSON. Start with { and end with }. No markdown, no explanation."""

INVOICE_PROMPT = """Extract all data from this invoice/receipt into JSON with this exact structure:
{{
    "vendor": "Company/person who issued the invoice (correct OCR errors in the name)",
    "invoice_number": "Invoice number or ID",
    "date": "Invoice date (YYYY-MM-DD format)",
    "due_date": "Due date if present (YYYY-MM-DD format)",
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "USD/NGN/EUR/GBP etc (infer from context if symbol is garbled)",
    "line_items": [
        {{
            "description": "Item description (fix obvious OCR typos)",
            "quantity": 1,
            "unit_price": 0.00,
            "amount": 0.00
        }}
    ]
}}

Rules:
- Use null for fields genuinely not found (not just garbled)
- For amounts, use numeric values only (no currency symbols)
- Correct obvious OCR errors using context
- Cross-check: line item amounts should be consistent with totals

OCR-extracted text (may contain errors):
{text}"""


GENERAL_SYSTEM = """You are an expert document data extraction specialist.
You are processing text extracted via OCR from a scanned, photographed, or handwritten document.

CRITICAL INSTRUCTIONS:
1. The text WILL contain OCR errors — misspelled words, garbled characters, wrong numbers.
   Use context and domain knowledge to infer correct values.
2. Common OCR mistakes: O/0, l/1/I confusion, merged/split words, garbled symbols.
3. For handwritten documents: words marked with [?] are uncertain readings. Use context
   to determine the most likely correct word. Handwritten text may have irregular spacing,
   abbreviations, and personal shorthand — interpret these naturally.
4. Correct obvious errors in your output — provide clean, accurate extracted data.
5. If a field is truly unreadable, use null rather than guessing.
6. Return ONLY valid JSON. Start with { and end with }. No markdown, no explanation."""

GENERAL_PROMPT = """Extract structured data from this document into JSON with this structure:
{{
    "title": "Document title or subject (correct any OCR errors)",
    "document_type": "What kind of document this is (e.g. lab report, letter, contract, academic paper, certificate, memo, etc.)",
    "date": "Primary date found (YYYY-MM-DD format)",
    "author": "Author, sender, or issuing organization if present",
    "key_fields": {{}},
    "entities": ["Named entities: people, companies, organizations, places (with OCR errors corrected)"],
    "dates": ["All dates found (YYYY-MM-DD format where possible)"],
    "amounts": ["All monetary or numeric amounts found (corrected)"],
    "summary": "A detailed 4-6 sentence summary of the document. Cover: (1) what the document is, (2) who created it and for what purpose, (3) the key findings, data, or conclusions, (4) any notable details or action items. Write in clear professional prose suitable for an email report. If parts were unreadable, mention that."
}}

Rules:
- Put domain-specific fields in key_fields as key-value pairs
- For technical/scientific documents, extract all measured values, parameters, and results into key_fields
- Correct obvious OCR errors using context
- The summary MUST be detailed enough to understand the document without reading the original
- Use null for fields genuinely not present

OCR-extracted text (may contain errors):
{text}"""


# ──────────────────────────────────────────────
# Parsing functions with retry
# ──────────────────────────────────────────────

def parse_invoice(raw_text: str) -> dict:
    """Extract structured invoice data using Claude with OCR error awareness."""
    logger.info("Parsing document as invoice")
    prompt = INVOICE_PROMPT.format(text=raw_text)

    # First attempt
    try:
        response = _chat(prompt, INVOICE_SYSTEM)
        result = _parse_json_response(response)

        if not result.get("parse_error"):
            # Validate and fix common issues
            result = _validate_invoice_data(result)
            return result
    except Exception as e:
        logger.warning(f"Invoice parsing attempt 1 failed: {e}")

    # Retry with simplified prompt
    logger.info("Retrying invoice parse with simplified prompt")
    try:
        retry_prompt = f"""The OCR text below is from an invoice. Extract what you can into JSON.
Focus on: vendor, total amount, date, and any line items.
Return valid JSON only.

Text:
{raw_text[:3000]}"""

        response = _chat(retry_prompt, INVOICE_SYSTEM)
        result = _parse_json_response(response)
        if not result.get("parse_error"):
            result = _validate_invoice_data(result)
            return result
    except Exception as e:
        logger.error(f"Invoice parsing retry failed: {e}")

    return {"parse_error": True, "error": "Failed to parse invoice after retries", "raw": raw_text[:1000]}


def parse_general(raw_text: str) -> dict:
    """Extract structured data from a general document with OCR error awareness."""
    logger.info("Parsing document as general")
    prompt = GENERAL_PROMPT.format(text=raw_text)

    # First attempt
    try:
        response = _chat(prompt, GENERAL_SYSTEM)
        result = _parse_json_response(response)
        if not result.get("parse_error"):
            return result
    except Exception as e:
        logger.warning(f"General parsing attempt 1 failed: {e}")

    # Retry with simplified prompt
    logger.info("Retrying general parse with simplified prompt")
    try:
        retry_prompt = f"""Extract key information from this OCR text into JSON.
Include: title, document_type, date, summary, and any key_fields.
Return valid JSON only.

Text:
{raw_text[:3000]}"""

        response = _chat(retry_prompt, GENERAL_SYSTEM)
        return _parse_json_response(response)
    except Exception as e:
        logger.error(f"General parsing retry failed: {e}")

    return {"parse_error": True, "error": "Failed to parse document after retries", "raw": raw_text[:1000]}


# ──────────────────────────────────────────────
# Invoice data validation
# ──────────────────────────────────────────────

def _validate_invoice_data(data: dict) -> dict:
    """Validate and fix invoice data consistency."""
    try:
        subtotal = data.get("subtotal")
        tax = data.get("tax")
        total = data.get("total")
        line_items = data.get("line_items", [])

        # Fix: If total is missing but subtotal and tax exist
        if total is None and subtotal is not None and tax is not None:
            data["total"] = round(subtotal + tax, 2)
            logger.info(f"Computed missing total: {data['total']}")

        # Fix: If subtotal is missing but line items exist
        if subtotal is None and line_items:
            item_total = sum(li.get("amount", 0) or 0 for li in line_items)
            if item_total > 0:
                data["subtotal"] = round(item_total, 2)
                logger.info(f"Computed missing subtotal from line items: {data['subtotal']}")

        # Fix: If total exists but subtotal doesn't and no tax
        if total is not None and subtotal is None and (tax is None or tax == 0):
            data["subtotal"] = total
            data["tax"] = 0

        # Warn if totals don't match
        if all(v is not None for v in [data.get("subtotal"), data.get("tax"), data.get("total")]):
            expected = round(data["subtotal"] + data["tax"], 2)
            actual = data["total"]
            if abs(expected - actual) > 0.02:
                logger.warning(f"Total mismatch: subtotal({data['subtotal']}) + "
                               f"tax({data['tax']}) = {expected}, but total = {actual}")

    except (TypeError, ValueError) as e:
        logger.warning(f"Invoice validation error: {e}")

    return data
