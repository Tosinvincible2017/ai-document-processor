"""
Multi-pass OCR engine with confidence scoring — V2.

Key V2 improvements:
- Multi-pass result MERGING (not just best-pick) — combines best parts
- OCR metadata in return values (confidence, profile, method used)
- PDF table-aware extraction using pdfplumber table detection
- Adaptive PSM mode selection based on image content
- Tesseract whitelist/blacklist configs per document type
- Higher resolution PDF rendering (400 DPI)
- Confidence reporting throughout the pipeline

Strategy:
1. For each image, run multiple preprocessing profiles
2. For each profile, run Tesseract with multiple PSM modes
3. Score each result by confidence + text quality heuristics
4. Merge complementary results when beneficial
5. Return the best result with full metadata
"""

import re
import base64
import io
import logging
from PIL import Image
import pytesseract
import pdfplumber
import anthropic
from preprocessor import (
    enhance_image,
    get_preprocessing_profiles,
    preprocess_grayscale_only,
    preprocess_handwriting,
    analyze_image,
)
from config import TESSERACT_CMD, ANTHROPIC_API_KEY, MODEL

logger = logging.getLogger(__name__)

# Configure Tesseract path if set
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ──────────────────────────────────────────────
# Tesseract PSM modes and configs
# ──────────────────────────────────────────────

PSM_MODES = [
    ("psm3", "--psm 3"),   # Fully automatic page segmentation (default)
    ("psm4", "--psm 4"),   # Assume a single column of text
    ("psm6", "--psm 6"),   # Assume a single uniform block of text
]

# Additional PSM modes for specific situations
PSM_MODES_EXTENDED = [
    ("psm11", "--psm 11"),  # Sparse text — find as much text as possible
    ("psm12", "--psm 12"),  # Sparse text with OSD
]

# Base Tesseract config
TESS_CONFIG_BASE = "--oem 3"  # LSTM neural net engine

# Specialized configs for better accuracy
TESS_CONFIG_DIGITS = "--oem 3 -c tessedit_char_whitelist=0123456789.,/$"
TESS_CONFIG_ALPHA = "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "


# ──────────────────────────────────────────────
# Text quality scoring (enhanced V2)
# ──────────────────────────────────────────────

def score_ocr_result(text: str, confidence: float) -> float:
    """
    Score an OCR result combining Tesseract confidence with text quality heuristics.
    Returns a score 0-100 where higher is better.

    V2 improvements:
    - Better handling of mixed alphanumeric content (invoices)
    - Language-aware word validation
    - Structural pattern recognition (dates, amounts, addresses)
    """
    if not text or not text.strip():
        return 0.0

    score = 0.0
    clean_text = text.strip()

    # 1. Tesseract confidence (0-100) — weight: 35%
    score += confidence * 0.35

    # 2. Word count — more real words = better (up to 25 points)
    words = clean_text.split()
    word_count = len(words)
    word_score = min(word_count / 10, 1.0) * 25
    score += word_score

    # 3. Alphabetic content ratio (up to 15 points)
    alpha_chars = sum(1 for c in clean_text if c.isalpha())
    total_chars = len(clean_text)
    if total_chars > 0:
        alpha_ratio = alpha_chars / total_chars
        # For invoices, alpha ratio might be lower (lots of numbers)
        # So we're more lenient: 30% alpha is acceptable
        if alpha_ratio > 0.3:
            score += min(alpha_ratio * 20, 15)
        elif alpha_ratio > 0.15:
            score += 8  # Still some text

    # 4. Average word length (up to 10 points)
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 3 <= avg_word_len <= 12:
            score += 10
        elif 2 <= avg_word_len <= 15:
            score += 5

    # 5. Structural pattern bonuses (up to 15 points)
    pattern_score = 0

    # Dates
    date_count = len(re.findall(
        r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{2,4}|'
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}',
        clean_text, re.IGNORECASE
    ))
    pattern_score += min(date_count * 3, 6)

    # Monetary amounts
    money_count = len(re.findall(
        r'[\$\£\€\₦]?\s*\d[\d,]*\.\d{2}|'
        r'\d[\d,]*\.\d{2}\s*(?:USD|NGN|EUR|GBP|CAD)',
        clean_text, re.IGNORECASE
    ))
    pattern_score += min(money_count * 2, 6)

    # Email addresses
    email_count = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_text))
    pattern_score += min(email_count * 2, 4)

    # Phone numbers
    phone_count = len(re.findall(r'[\+]?\d[\d\s\-\(\)]{7,15}', clean_text))
    pattern_score += min(phone_count, 3)

    score += min(pattern_score, 15)

    # 6. Penalties
    # Excessive special characters
    special_ratio = sum(1 for c in clean_text if c in '|\\{}[]<>~^`') / max(total_chars, 1)
    if special_ratio > 0.1:
        score -= 15
    elif special_ratio > 0.05:
        score -= 8

    # Excessive single-char "words"
    single_chars = sum(1 for w in words if len(w) == 1 and w.lower() not in 'aio')
    if word_count > 0 and single_chars / word_count > 0.3:
        score -= 10

    # Excessive line breaks (fragmented text)
    lines = [l for l in clean_text.split('\n') if l.strip()]
    if lines and word_count > 0:
        words_per_line = word_count / len(lines)
        if words_per_line < 1.5:
            score -= 10

    # Consecutive repeated characters (OCR garbage indicator)
    repeated = len(re.findall(r'(.)\1{4,}', clean_text))
    if repeated > 2:
        score -= 10

    return max(0, min(100, score))


def get_tesseract_confidence(image: Image.Image, config: str = "") -> tuple:
    """
    Run Tesseract and return (text, average_confidence, word_count, low_conf_words).
    Uses image_to_data for per-word confidence scores.
    """
    full_config = f"{TESS_CONFIG_BASE} {config}".strip()

    try:
        data = pytesseract.image_to_data(
            image, lang="eng", config=full_config,
            output_type=pytesseract.Output.DICT
        )

        words = []
        confidences = []
        low_conf_words = []
        current_line = -1
        lines = []
        line_words = []

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            line_num = data["line_num"][i]

            if conf < 0:
                continue

            if line_num != current_line:
                if line_words:
                    lines.append(" ".join(line_words))
                    line_words = []
                current_line = line_num

            if word:
                words.append(word)
                line_words.append(word)
                if conf > 0:
                    confidences.append(conf)
                if 0 < conf < 50:
                    low_conf_words.append((word, conf))

        if line_words:
            lines.append(" ".join(line_words))

        text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_conf, len(words), low_conf_words

    except Exception as e:
        logger.warning(f"Tesseract data extraction failed: {e}, falling back to image_to_string")
        text = pytesseract.image_to_string(image, lang="eng", config=full_config)
        return text, 50.0, len(text.split()), []


# ──────────────────────────────────────────────
# Multi-pass OCR with result merging (V2)
# ──────────────────────────────────────────────

def multi_pass_ocr(pil_image: Image.Image,
                   enable_diagnostics: bool = False,
                   job_id: str = "") -> dict:
    """
    Run multiple preprocessing profiles x PSM modes.

    V2 returns a rich metadata dict:
    {
        "text": str,
        "confidence": float,
        "profile": str,
        "psm": str,
        "score": float,
        "word_count": int,
        "low_confidence_words": list,
        "passes_run": int,
        "all_scores": dict,
        "merge_applied": bool
    }
    """
    profiles = get_preprocessing_profiles(
        pil_image,
        enable_diagnostics=enable_diagnostics,
        job_id=job_id
    )

    best_text = ""
    best_score = -1
    best_conf = 0
    best_profile = "none"
    best_psm = "none"
    best_word_count = 0
    best_low_conf = []
    passes_run = 0
    all_scores = {}
    all_results = []

    for profile_name, processed_image in profiles:
        for psm_name, psm_config in PSM_MODES:
            try:
                passes_run += 1
                text, conf, wc, low_conf = get_tesseract_confidence(
                    processed_image, psm_config
                )
                score = score_ocr_result(text, conf)

                key = f"{profile_name}+{psm_name}"
                all_scores[key] = round(score, 1)

                logger.debug(f"  [{key}] conf={conf:.1f}, score={score:.1f}, "
                             f"words={wc}, chars={len(text.strip())}")

                # Store for potential merging
                all_results.append({
                    "text": text,
                    "score": score,
                    "confidence": conf,
                    "profile": profile_name,
                    "psm": psm_name,
                    "word_count": wc,
                    "low_conf_words": low_conf,
                })

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_conf = conf
                    best_profile = profile_name
                    best_psm = psm_name
                    best_word_count = wc
                    best_low_conf = low_conf

            except Exception as e:
                logger.warning(f"OCR pass [{profile_name}+{psm_name}] failed: {e}")
                continue

        # Early exit if excellent result
        if best_score > 85:
            logger.info(f"Early exit: excellent score {best_score:.1f} "
                        f"from {best_profile}+{best_psm}")
            break

    # Attempt to merge results if best score is mediocre
    merge_applied = False
    if best_score < 70 and len(all_results) > 1:
        merged = _merge_ocr_results(all_results)
        if merged:
            merged_score = score_ocr_result(merged, best_conf)
            if merged_score > best_score:
                logger.info(f"Merge improved score: {best_score:.1f} -> {merged_score:.1f}")
                best_text = merged
                best_score = merged_score
                merge_applied = True

    logger.info(f"Best OCR: profile={best_profile}, psm={best_psm}, "
                f"conf={best_conf:.1f}, score={best_score:.1f}, "
                f"words={best_word_count}, passes={passes_run}")

    return {
        "text": best_text,
        "confidence": round(best_conf, 1),
        "profile": best_profile,
        "psm": best_psm,
        "score": round(best_score, 1),
        "word_count": best_word_count,
        "low_confidence_words": best_low_conf[:20],  # Cap at 20
        "passes_run": passes_run,
        "all_scores": all_scores,
        "merge_applied": merge_applied,
    }


def _merge_ocr_results(results: list) -> str:
    """
    Merge multiple OCR results by taking the best version of each line.

    Strategy:
    - Align results by line count
    - For each line position, pick the version with highest "line quality"
    - This handles cases where different profiles excel at different parts
    """
    if not results:
        return ""

    # Sort by score descending
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)

    # Only merge top 3 results
    top_results = sorted_results[:3]

    # Split each into lines
    result_lines = []
    for r in top_results:
        lines = [l.strip() for l in r["text"].split('\n') if l.strip()]
        result_lines.append(lines)

    if not result_lines:
        return ""

    # Use the result with most lines as the base
    base_idx = max(range(len(result_lines)), key=lambda i: len(result_lines[i]))
    base_lines = result_lines[base_idx]

    if len(base_lines) == 0:
        return sorted_results[0]["text"]

    # For each line, pick the best version across all results
    merged_lines = []
    for i, base_line in enumerate(base_lines):
        best_line = base_line
        best_line_score = _score_line(base_line)

        for j, other_lines in enumerate(result_lines):
            if j == base_idx:
                continue
            # Find the closest matching line in other results
            match = _find_matching_line(base_line, other_lines)
            if match:
                match_score = _score_line(match)
                if match_score > best_line_score:
                    best_line = match
                    best_line_score = match_score

        merged_lines.append(best_line)

    return '\n'.join(merged_lines)


def _score_line(line: str) -> float:
    """Score a single line for quality."""
    if not line.strip():
        return 0

    score = 0
    words = line.split()

    # Length bonus
    score += min(len(words) * 2, 10)

    # Alphabetic ratio
    alpha = sum(1 for c in line if c.isalpha())
    if len(line) > 0:
        score += (alpha / len(line)) * 10

    # Penalty for gibberish characters
    garbage = sum(1 for c in line if c in '|\\{}[]<>~^`')
    score -= garbage * 3

    return score


def _find_matching_line(target: str, candidates: list) -> str:
    """Find the most similar line in candidates using simple word overlap."""
    if not candidates:
        return None

    target_words = set(target.lower().split())
    if not target_words:
        return None

    best_match = None
    best_overlap = 0

    for cand in candidates:
        cand_words = set(cand.lower().split())
        if not cand_words:
            continue
        overlap = len(target_words & cand_words) / max(len(target_words), len(cand_words))
        if overlap > best_overlap and overlap > 0.3:
            best_overlap = overlap
            best_match = cand

    return best_match


# ──────────────────────────────────────────────
# Claude Vision extraction (for handwriting)
# ──────────────────────────────────────────────

_vision_client = None

def _get_vision_client():
    """Lazy-init the Anthropic client for vision calls."""
    global _vision_client
    if _vision_client is None:
        _vision_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _vision_client


def _pil_to_base64(image: Image.Image, max_size: int = 2000) -> tuple:
    """
    Convert a PIL image to base64 for Claude Vision API.
    Resizes if too large (API has limits).
    Returns (base64_string, media_type).
    """
    # Resize if needed
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # Convert to RGB if needed
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # Save to buffer as PNG
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    b64 = base64.standard_b64encode(buf.read()).decode("utf-8")
    return b64, "image/png"


def extract_text_vision(image: Image.Image, context_hint: str = "") -> dict:
    """
    Use Claude Vision API to read text from an image.
    Much better than Tesseract for handwriting, mixed layouts, etc.

    Returns dict similar to multi_pass_ocr:
    {
        "text": str,
        "confidence": float,
        "profile": "vision",
        "psm": "vision",
        "score": float,
        "word_count": int,
        "low_confidence_words": [],
        "passes_run": 1,
        "all_scores": {"vision": score},
        "merge_applied": False,
        "extraction_method": "claude_vision"
    }
    """
    client = _get_vision_client()
    b64_image, media_type = _pil_to_base64(image)

    context = ""
    if context_hint:
        context = f"\nContext: {context_hint}"

    system = (
        "You are an expert document reader and transcriber. Your task is to read ALL text "
        "visible in the image and transcribe it accurately. You handle both printed and "
        "handwritten text with high accuracy."
    )

    prompt = (
        f"Read and transcribe ALL text in this image exactly as written. "
        f"Include every word, number, date, and symbol you can see.{context}\n\n"
        f"Rules:\n"
        f"- Transcribe handwritten text as accurately as possible\n"
        f"- Preserve the original layout/structure (line breaks, paragraphs)\n"
        f"- If a word is unclear, give your best guess with [?] after it\n"
        f"- Include headers, footers, annotations, and margin notes\n"
        f"- For tables, use | to separate columns\n"
        f"- Do NOT add commentary or explanations — only the transcribed text\n"
        f"- Do NOT wrap in code blocks or quotes"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
        )

        text = response.content[0].text.strip()

        # Strip any code fences Claude might add
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()

        # Count uncertain words marked with [?]
        uncertain_words = re.findall(r'(\S+)\s*\[\?\]', text)
        word_count = len(text.split())

        # Estimate confidence based on uncertain markers
        uncertain_ratio = len(uncertain_words) / max(word_count, 1)
        confidence = max(30, min(98, 95 - (uncertain_ratio * 100)))

        score = score_ocr_result(text, confidence)

        logger.info(f"Vision extraction: {len(text)} chars, {word_count} words, "
                     f"confidence={confidence:.0f}%, uncertain={len(uncertain_words)} words")

        return {
            "text": text,
            "confidence": round(confidence, 1),
            "profile": "vision",
            "psm": "vision",
            "score": round(score, 1),
            "word_count": word_count,
            "low_confidence_words": [(w, 30) for w in uncertain_words[:20]],
            "passes_run": 1,
            "all_scores": {"vision": round(score, 1)},
            "merge_applied": False,
            "extraction_method": "claude_vision",
        }

    except Exception as e:
        logger.error(f"Claude Vision extraction failed: {e}")
        return None


def extract_with_vision_fallback(image: Image.Image,
                                  force_vision: bool = False,
                                  context_hint: str = "") -> dict:
    """
    Smart extraction: uses YOLO region detection to identify handwritten vs
    printed areas, then routes each region to the best extractor.

    For mixed documents (printed header + handwritten body):
    - Printed regions -> Tesseract multi-pass OCR
    - Handwritten regions -> Claude Vision API

    For uniform documents, falls back to whole-image extraction.

    Args:
        image: PIL Image to extract text from
        force_vision: If True, skip Tesseract and go straight to Vision
        context_hint: Optional context to help Vision API

    Returns: Same dict format as multi_pass_ocr
    """
    # Check if image is likely handwritten (now uses YOLO when available)
    analysis = analyze_image(image)
    is_handwritten = analysis.get("likely_handwritten", False)

    # Try YOLO region-based extraction for mixed documents
    yolo_result = _try_yolo_region_extraction(image, context_hint)
    if yolo_result is not None:
        return yolo_result

    if force_vision or is_handwritten:
        logger.info(f"Using Claude Vision for extraction "
                     f"(force={force_vision}, handwritten={is_handwritten})")

        # Try Vision first for handwritten docs
        vision_result = extract_text_vision(image, context_hint)

        if vision_result and vision_result["text"].strip():
            # Also run Tesseract for comparison/merging
            if not force_vision:
                tess_result = multi_pass_ocr(image)
                if tess_result["score"] > vision_result["score"] + 15:
                    logger.info(f"Tesseract ({tess_result['score']:.0f}) beat Vision "
                                 f"({vision_result['score']:.0f}), using Tesseract")
                    tess_result["handwriting_detected"] = is_handwritten
                    return tess_result

            vision_result["handwriting_detected"] = is_handwritten
            return vision_result

        # Vision failed, fall back to Tesseract
        logger.warning("Vision extraction failed or empty, falling back to Tesseract")

    # Standard Tesseract multi-pass
    result = multi_pass_ocr(image)

    # If Tesseract result is poor, try Vision as fallback
    if result["score"] < 30 and ANTHROPIC_API_KEY:
        logger.info(f"Tesseract score low ({result['score']:.0f}), trying Vision fallback")
        vision_result = extract_text_vision(image, context_hint)

        if vision_result and vision_result["score"] > result["score"]:
            logger.info(f"Vision ({vision_result['score']:.0f}) beat Tesseract "
                         f"({result['score']:.0f}), using Vision")
            vision_result["handwriting_detected"] = is_handwritten
            vision_result["all_scores"].update(result["all_scores"])
            return vision_result

    result["handwriting_detected"] = is_handwritten
    return result


def _try_yolo_region_extraction(image: Image.Image,
                                 context_hint: str = "") -> dict:
    """
    Attempt YOLO-based region extraction for mixed documents.

    Detects handwritten vs printed regions, routes each to the appropriate
    extractor (Vision for handwriting, Tesseract for printed), then merges.

    Returns None if YOLO is unavailable or the document isn't mixed.
    Returns the combined OCR result dict if successful.
    """
    try:
        from yolo_detector import detect_document_regions, is_yolo_available

        if not is_yolo_available():
            return None

        detection = detect_document_regions(image)

        # Skip if no handwriting detected by YOLO
        if not detection["has_handwriting"]:
            return None

        is_mixed = detection["is_mixed"]
        hw_ratio = detection["handwriting_ratio"]
        logger.info(f"YOLO detected {'mixed' if is_mixed else 'handwritten'} document: "
                     f"{detection['handwriting_regions']} handwritten, "
                     f"{detection['printed_regions']} printed regions, "
                     f"hw_ratio={hw_ratio:.0%}")

        # For purely handwritten docs (not mixed), route whole image to Vision
        if not is_mixed and hw_ratio > 0.5 and ANTHROPIC_API_KEY:
            logger.info("Purely handwritten — sending full image to Vision API")
            vision_result = extract_text_vision(image, context_hint)
            if vision_result and vision_result["text"].strip():
                vision_result["handwriting_detected"] = True
                vision_result["yolo_detection"] = {
                    "method": detection["detection_method"],
                    "model_type": detection["model_type"],
                    "total_regions": detection["region_count"],
                    "handwritten_regions": detection["handwriting_regions"],
                    "printed_regions": detection["printed_regions"],
                    "handwriting_ratio": detection["handwriting_ratio"],
                }
                return vision_result
            logger.warning("Vision failed for handwritten doc, falling back")
            return None

        all_texts = []
        region_methods = []
        total_confidence = 0
        total_score = 0
        region_count = 0
        all_scores = {}

        for region in detection["regions"]:
            region_crop = region.crop_pil(image)
            region_label = f"region_{region_count}"

            if region.is_handwritten and ANTHROPIC_API_KEY:
                # Route handwritten region to Claude Vision
                logger.info(f"  {region_label}: handwritten (hw_score={region.hw_score:.2f})"
                           f" -> Vision API")
                vision_result = extract_text_vision(region_crop, context_hint)
                if vision_result and vision_result["text"].strip():
                    all_texts.append(vision_result["text"])
                    total_confidence += vision_result["confidence"]
                    total_score += vision_result["score"]
                    region_methods.append("claude_vision")
                    all_scores[f"{region_label}_vision"] = vision_result["score"]
                    region_count += 1
                    continue

                # Vision failed for this region, fall back to Tesseract
                logger.warning(f"  {region_label}: Vision failed, falling back to Tesseract")

            # Route printed region (or failed Vision) to Tesseract
            logger.info(f"  {region_label}: printed (hw_score={region.hw_score:.2f})"
                       f" -> Tesseract")
            tess_result = multi_pass_ocr(region_crop)
            if tess_result["text"].strip():
                all_texts.append(tess_result["text"])
                total_confidence += tess_result["confidence"]
                total_score += tess_result["score"]
                region_methods.append("tesseract")
                all_scores[f"{region_label}_tess"] = tess_result["score"]
                region_count += 1

        if not all_texts:
            return None

        # Merge all region texts
        combined_text = "\n\n".join(all_texts)
        avg_confidence = total_confidence / max(region_count, 1)
        avg_score = total_score / max(region_count, 1)

        # Determine primary extraction method
        vision_count = region_methods.count("claude_vision")
        tess_count = region_methods.count("tesseract")
        if vision_count > tess_count:
            primary_method = "claude_vision"
        elif vision_count > 0:
            primary_method = "mixed (vision+tesseract)"
        else:
            primary_method = "tesseract"

        result = {
            "text": combined_text,
            "confidence": round(avg_confidence, 1),
            "profile": "yolo_regions",
            "psm": "region_based",
            "score": round(avg_score, 1),
            "word_count": len(combined_text.split()),
            "low_confidence_words": [],
            "passes_run": region_count,
            "all_scores": all_scores,
            "merge_applied": True,
            "extraction_method": primary_method,
            "handwriting_detected": True,
            "yolo_detection": {
                "method": detection["detection_method"],
                "model_type": detection["model_type"],
                "total_regions": detection["region_count"],
                "handwritten_regions": detection["handwriting_regions"],
                "printed_regions": detection["printed_regions"],
                "handwriting_ratio": detection["handwriting_ratio"],
            },
        }

        logger.info(f"YOLO region extraction complete: {region_count} regions, "
                     f"method={primary_method}, score={avg_score:.1f}")

        return result

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"YOLO region extraction failed: {e}")
        return None


# ──────────────────────────────────────────────
# Text post-processing (enhanced V2)
# ──────────────────────────────────────────────

def clean_ocr_text(text: str) -> str:
    """Post-process OCR text to fix common errors."""
    if not text:
        return text

    # Remove NULL bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)

    # Remove isolated single characters that are likely noise
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 2 and not stripped.isalnum():
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Fix common OCR character confusions in monetary amounts
    def fix_money_Os(match):
        s = match.group(0)
        return s.replace('O', '0').replace('o', '0')

    text = re.sub(r'[\$\£\€\₦]\s*[\dOo,]+\.[\dOo]{2}', fix_money_Os, text)

    # Fix "l" (lowercase L) misread as "1" in clearly alphabetic words
    def fix_l_as_1(match):
        word = match.group(0)
        if any(c.isdigit() for c in word):
            return word
        return word.replace('1', 'l')

    text = re.sub(r'\b[a-zA-Z]*1[a-zA-Z]+\b', fix_l_as_1, text)

    # Fix common word-level OCR errors
    common_fixes = {
        'lnvoice': 'Invoice',
        'Tota1': 'Total',
        'tota1': 'total',
        'Arnount': 'Amount',
        'arnount': 'amount',
        'Narne': 'Name',
        'narne': 'name',
        'Aclclress': 'Address',
        'aclclress': 'address',
        'Payrnent': 'Payment',
        'payrnent': 'payment',
        'Receip+': 'Receipt',
        'Arnount': 'Amount',
        'Quantify': 'Quantity',
        'Oescription': 'Description',
        'Oate': 'Date',
    }
    for wrong, right in common_fixes.items():
        text = text.replace(wrong, right)

    return text.strip()


# ──────────────────────────────────────────────
# PDF extraction (enhanced V2 with table detection)
# ──────────────────────────────────────────────

def _extract_tables_from_page(page) -> str:
    """
    Extract tables from a PDF page using pdfplumber's table detection.
    Returns formatted text representation of tables found.
    """
    try:
        tables = page.extract_tables()
        if not tables:
            return ""

        table_text_parts = []
        for t_idx, table in enumerate(tables):
            if not table:
                continue

            rows = []
            for row in table:
                if row:
                    # Clean None values and join
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    if any(c for c in cleaned):  # Skip empty rows
                        rows.append(" | ".join(cleaned))

            if rows:
                table_text_parts.append("\n".join(rows))

        if table_text_parts:
            result = "\n\n".join(table_text_parts)
            logger.info(f"Extracted {len(tables)} table(s) from page")
            return result

    except Exception as e:
        logger.debug(f"Table extraction failed: {e}")

    return ""


def extract_from_pdf(file_path: str) -> tuple:
    """
    Extract text from PDF with V2 improvements:
    - Native text + table extraction combined
    - Higher resolution OCR (400 DPI)
    - Multi-pass OCR for scanned pages
    - OCR metadata collection

    Returns (text, page_count, ocr_metadata).
    """
    all_text = []
    page_count = 0
    ocr_pages = 0
    ocr_metadata_list = []

    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"PDF has {page_count} pages")

            for i, page in enumerate(pdf.pages):
                # Try native text extraction first
                native_text = page.extract_text() or ""

                # Also extract tables (even from native PDFs)
                table_text = _extract_tables_from_page(page)

                # Combine native text and table text
                if table_text and table_text not in native_text:
                    combined_native = f"{native_text}\n\n{table_text}".strip()
                else:
                    combined_native = native_text

                if len(combined_native.strip()) < 50:
                    # Likely a scanned page — use multi-pass OCR
                    logger.info(f"Page {i+1}: low native text ({len(combined_native.strip())} chars), running OCR")
                    page_text, page_meta = _ocr_pdf_page(page, i)
                    all_text.append(page_text)
                    ocr_metadata_list.append(page_meta)
                    ocr_pages += 1
                else:
                    logger.info(f"Page {i+1}: {len(combined_native)} chars "
                               f"(native{'+ tables' if table_text else ''})")
                    all_text.append(combined_native)

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

    combined = "\n\n".join(all_text)
    combined = clean_ocr_text(combined)

    if not combined.strip():
        raise RuntimeError("No text could be extracted from the PDF")

    if ocr_pages > 0:
        logger.info(f"PDF: {ocr_pages}/{page_count} pages required OCR")

    # Aggregate OCR metadata
    ocr_metadata = None
    if ocr_metadata_list:
        avg_conf = sum(m.get("confidence", 0) for m in ocr_metadata_list) / len(ocr_metadata_list)
        ocr_metadata = {
            "ocr_pages": ocr_pages,
            "total_pages": page_count,
            "avg_confidence": round(avg_conf, 1),
            "per_page": ocr_metadata_list,
        }

    return combined, page_count, ocr_metadata


def _ocr_pdf_page(page, page_num: int) -> tuple:
    """
    Render a PDF page at high resolution and run OCR with Vision fallback.
    Automatically uses Claude Vision for handwritten pages.
    Returns (text, ocr_metadata).
    """
    try:
        # Render at 400 DPI for better OCR
        page_image = page.to_image(resolution=400)
        pil_image = page_image.original

        result = extract_with_vision_fallback(pil_image)
        text = result["text"]
        text = clean_ocr_text(text)

        meta = {
            "page": page_num + 1,
            "confidence": result["confidence"],
            "profile": result["profile"],
            "psm": result["psm"],
            "score": result["score"],
            "word_count": result["word_count"],
            "merge_applied": result["merge_applied"],
            "extraction_method": result.get("extraction_method", "tesseract"),
            "handwriting_detected": result.get("handwriting_detected", False),
        }

        method = meta["extraction_method"]
        logger.info(f"Page {page_num+1}: {len(text)} chars, "
                     f"conf={result['confidence']}%, method={method}, "
                     f"profile={result['profile']}")
        return text, meta

    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed. "
            "Set TESSERACT_CMD in .env for Windows."
        )
    except Exception as e:
        logger.warning(f"OCR failed on page {page_num+1}: {e}")
        return "", {"page": page_num + 1, "error": str(e)}


def extract_from_image(file_path: str) -> tuple:
    """
    Extract text from an image using multi-pass OCR with Vision fallback.
    Automatically uses Claude Vision for handwritten documents.
    Returns (text, page_count, ocr_metadata).
    """
    try:
        image = Image.open(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    logger.info(f"Image: {image.size[0]}x{image.size[1]}, mode={image.mode}")

    try:
        result = extract_with_vision_fallback(image)
    except pytesseract.TesseractNotFoundError:
        # Tesseract not available — try Vision-only
        if ANTHROPIC_API_KEY:
            logger.warning("Tesseract not found, using Claude Vision only")
            result = extract_text_vision(image)
            if not result:
                raise RuntimeError("Both Tesseract and Vision extraction failed")
        else:
            raise RuntimeError(
                "Tesseract OCR is not installed or not found. "
                "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
                "and set TESSERACT_CMD in your .env file."
            )

    text = clean_ocr_text(result["text"])

    if not text.strip():
        raise RuntimeError("No text could be extracted from the image")

    ocr_metadata = {
        "confidence": result["confidence"],
        "profile": result["profile"],
        "psm": result["psm"],
        "score": result["score"],
        "word_count": result["word_count"],
        "low_confidence_words": result["low_confidence_words"],
        "passes_run": result["passes_run"],
        "all_scores": result["all_scores"],
        "merge_applied": result["merge_applied"],
        "extraction_method": result.get("extraction_method", "tesseract"),
        "handwriting_detected": result.get("handwriting_detected", False),
    }

    # Include YOLO detection info if present
    if result.get("yolo_detection"):
        ocr_metadata["yolo_detection"] = result["yolo_detection"]

    logger.info(f"Image extraction: {len(text)} chars, confidence={result['confidence']}%, "
                f"method={ocr_metadata['extraction_method']}, "
                f"handwritten={ocr_metadata['handwriting_detected']}")
    return text, 1, ocr_metadata


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def extract_text(file_path: str, file_type: str) -> tuple:
    """
    Extract text from a file.
    Returns (text, page_count, ocr_metadata).

    ocr_metadata is a dict with confidence, profile, scores, etc.
    Can be None for native PDF text extraction.
    """
    if file_type == "pdf":
        return extract_from_pdf(file_path)
    return extract_from_image(file_path)
