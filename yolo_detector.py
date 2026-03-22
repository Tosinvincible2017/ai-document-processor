"""
YOLO-based document region detection for handwriting identification.

Uses YOLOv8 to segment documents into regions, then classifies each
region as handwritten or printed text. This enables:

1. Mixed document handling — printed header + handwritten body
2. Region-level OCR routing — Vision API for handwriting, Tesseract for print
3. More accurate handwriting detection than global heuristics alone

Model priority:
1. Custom-trained model at models/doc_layout.pt (if you fine-tune one)
2. Auto-downloaded yolov8n-seg for general segmentation + per-region heuristics
"""

import os
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Optional

from config import BASE_DIR

logger = logging.getLogger(__name__)

# Path for custom-trained document layout model
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "doc_layout.pt")

# Cache the loaded model
_yolo_model = None
_model_type = None  # "custom" or "general"


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def _load_model():
    """Load YOLO model: custom doc layout model if available, else general YOLOv8n."""
    global _yolo_model, _model_type

    if _yolo_model is not None:
        return _yolo_model, _model_type

    try:
        from ultralytics import YOLO

        # Try custom document layout model first
        if os.path.exists(CUSTOM_MODEL_PATH):
            logger.info(f"Loading custom document layout model: {CUSTOM_MODEL_PATH}")
            _yolo_model = YOLO(CUSTOM_MODEL_PATH)
            _model_type = "custom"
        else:
            # Use general YOLOv8 nano model (fast, lightweight)
            # We'll use detection mode to find text-like regions,
            # then apply our heuristics per-region
            logger.info("Loading YOLOv8n for document region detection")
            _yolo_model = YOLO("yolov8n.pt")
            _model_type = "general"

        logger.info(f"YOLO model loaded: type={_model_type}")
        return _yolo_model, _model_type

    except Exception as e:
        logger.warning(f"Failed to load YOLO model: {e}")
        return None, None


def is_yolo_available() -> bool:
    """Check if YOLO/ultralytics is installed and a model can be loaded."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────
# Region detection and classification
# ──────────────────────────────────────────────

class DocumentRegion:
    """Represents a detected region in a document."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 region_type: str = "unknown", confidence: float = 0.0,
                 is_handwritten: bool = False, hw_score: float = 0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.region_type = region_type  # "text", "table", "figure", "handwriting"
        self.confidence = confidence
        self.is_handwritten = is_handwritten
        self.hw_score = hw_score  # handwriting score 0-1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def crop_from(self, image: np.ndarray) -> np.ndarray:
        """Extract this region from an image."""
        return image[self.y1:self.y2, self.x1:self.x2].copy()

    def crop_pil(self, pil_image: Image.Image) -> Image.Image:
        """Extract this region from a PIL image."""
        return pil_image.crop((self.x1, self.y1, self.x2, self.y2))

    def to_dict(self) -> dict:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "region_type": self.region_type,
            "confidence": round(self.confidence, 3),
            "is_handwritten": self.is_handwritten,
            "hw_score": round(self.hw_score, 3),
            "width": self.width,
            "height": self.height,
        }

    def __repr__(self):
        hw = " [HW]" if self.is_handwritten else ""
        return (f"Region({self.region_type}{hw}, "
                f"{self.x1},{self.y1}-{self.x2},{self.y2}, "
                f"conf={self.confidence:.2f})")


def _classify_region_handwriting(gray_region: np.ndarray) -> tuple:
    """
    Classify a single region as handwritten or printed using image analysis.

    Uses 5 features with weighted scoring:
    1. Stroke width variation (distance transform)
    2. Connected component irregularity (area/aspect CV)
    3. Contour complexity (circularity)
    4. Edge direction dispersion (Sobel-based) — strongest signal
    5. Baseline regularity (horizontal projection)

    Returns (is_handwritten: bool, score: float 0-1).
    """
    h, w = gray_region.shape[:2]

    # Too small to analyze
    if w < 30 or h < 30:
        return False, 0.0

    try:
        # Binarize the region
        _, binary = cv2.threshold(gray_region, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ink_pixels = cv2.countNonZero(binary)
        total_pixels = h * w

        # Skip mostly empty regions
        if ink_pixels < total_pixels * 0.01:
            return False, 0.0

        score = 0.0
        max_score = 0.0

        # 1. Stroke width variation (weight: 0.20)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        stroke_pixels = dist[dist > 0]
        if len(stroke_pixels) > 20:
            stroke_mean = float(np.mean(stroke_pixels))
            stroke_std = float(np.std(stroke_pixels))
            stroke_cv = stroke_std / max(stroke_mean, 0.01)

            # Handwriting: high variation (CV > 0.3)
            if stroke_cv > 0.5:
                score += 0.20
            elif stroke_cv > 0.3:
                score += 0.12
            elif stroke_cv > 0.15:
                score += 0.05
            max_score += 0.20

        # 2. Connected component regularity (weight: 0.15)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        valid_areas = []
        valid_aspects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            if 10 < area < (h * w * 0.5) and cw > 2 and ch > 2:
                valid_areas.append(area)
                valid_aspects.append(cw / ch)

        if len(valid_areas) > 3:
            area_cv = float(np.std(valid_areas)) / max(float(np.mean(valid_areas)), 1)
            aspect_cv = float(np.std(valid_aspects)) / max(float(np.mean(valid_aspects)), 0.01)

            if area_cv > 1.2:
                score += 0.08
            elif area_cv > 0.4:
                score += 0.04

            if aspect_cv > 0.6:
                score += 0.07
            elif aspect_cv > 0.3:
                score += 0.04
            max_score += 0.15

        # 3. Contour complexity (weight: 0.20)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        complexities = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if area > 20 and peri > 5:
                circularity = 4 * np.pi * area / (peri * peri)
                complexities.append(circularity)

        if complexities:
            avg_complexity = float(np.mean(complexities))
            # Handwriting: more complex contours (lower circularity)
            if avg_complexity < 0.15:
                score += 0.20
            elif avg_complexity < 0.30:
                score += 0.14
            elif avg_complexity < 0.40:
                score += 0.08
            max_score += 0.20

        # 4. Edge direction dispersion (weight: 0.30 — strongest feature)
        # Handwriting has strokes in many directions; printed text is mostly horizontal
        if w >= 40 and h >= 40:
            sobel_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            angles = np.arctan2(sobel_y, sobel_x)

            edge_mask = magnitude > np.percentile(magnitude, 70)
            edge_angles = angles[edge_mask]

            if len(edge_angles) > 50:
                sin_mean = float(np.mean(np.sin(2 * edge_angles)))
                cos_mean = float(np.mean(np.cos(2 * edge_angles)))
                R = np.sqrt(sin_mean**2 + cos_mean**2)
                dispersion = 1.0 - R  # 0=uniform direction, 1=all directions

                # Handwriting: high dispersion (>0.5), print: low (<0.3)
                if dispersion > 0.7:
                    score += 0.30
                elif dispersion > 0.5:
                    score += 0.22
                elif dispersion > 0.35:
                    score += 0.12
                max_score += 0.30

        # 5. Baseline regularity (weight: 0.15)
        v_proj = np.sum(binary, axis=1)
        v_proj_norm = v_proj / max(v_proj.max(), 1)

        transitions = 0
        above = False
        for val in v_proj_norm:
            if val > 0.05 and not above:
                transitions += 1
                above = True
            elif val <= 0.02:
                above = False

        if transitions > 1:
            peak_positions = np.where(v_proj_norm > 0.05)[0]
            if len(peak_positions) > 3:
                diffs = np.diff(peak_positions)
                line_regularity = float(np.std(diffs)) / max(float(np.mean(diffs)), 1)
                # Handwriting: irregular line spacing
                if line_regularity > 0.5:
                    score += 0.15
                elif line_regularity > 0.3:
                    score += 0.08
                elif line_regularity > 0.15:
                    score += 0.04
            max_score += 0.15

        # Normalize: score / max possible score
        if max_score > 0:
            normalized = min(score / max_score, 1.0)
        else:
            normalized = 0.0

        is_hw = normalized >= 0.45
        return is_hw, round(normalized, 3)

    except Exception as e:
        logger.debug(f"Region handwriting classification failed: {e}")
        return False, 0.0


def detect_text_regions(gray: np.ndarray, min_area: int = 500) -> list:
    """
    Detect text regions in a document using morphological operations.
    This works as a YOLO-independent fallback for region detection.

    Returns list of (x, y, w, h) bounding boxes.
    """
    h, w = gray.shape[:2]

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge nearby text into blocks
    # Horizontal kernel wider than vertical to merge words into lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 20, 15), 3))
    dilated = cv2.dilate(binary, h_kernel, iterations=2)

    # Then vertical dilation to merge lines into paragraphs
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(h // 40, 8)))
    dilated = cv2.dilate(dilated, v_kernel, iterations=2)

    # Find contours (text blocks)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        # Filter by size
        if area < min_area:
            continue
        if bw < 20 or bh < 10:
            continue
        # Skip regions that span the entire image (likely background)
        if bw > w * 0.95 and bh > h * 0.95:
            continue

        regions.append((x, y, bw, bh))

    # Sort top-to-bottom, left-to-right
    regions.sort(key=lambda r: (r[1], r[0]))

    return regions


def detect_document_regions(pil_image: Image.Image,
                            use_yolo: bool = True) -> dict:
    """
    Detect and classify regions in a document image.

    Uses YOLO for region detection when available, falls back to
    morphological region detection + per-region handwriting heuristics.

    Returns:
    {
        "regions": [DocumentRegion, ...],
        "has_handwriting": bool,
        "has_printed": bool,
        "is_mixed": bool,
        "handwriting_ratio": float,   # 0-1
        "detection_method": str,      # "yolo" or "morphological"
        "model_type": str,            # "custom", "general", or "heuristic"
    }
    """
    # Convert to OpenCV grayscale
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    img_cv = np.array(pil_image)
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_cv

    h, w = gray.shape[:2]
    regions = []
    detection_method = "morphological"
    model_type_used = "heuristic"

    # Try YOLO-based detection
    if use_yolo:
        model, model_type = _load_model()

        if model is not None:
            try:
                regions = _detect_with_yolo(model, model_type, pil_image, gray)
                detection_method = "yolo"
                model_type_used = model_type
                logger.info(f"YOLO detected {len(regions)} regions "
                            f"(model={model_type})")
            except Exception as e:
                logger.warning(f"YOLO detection failed, falling back: {e}")
                regions = []

    # Fallback: morphological region detection
    if not regions:
        raw_regions = detect_text_regions(gray)
        logger.info(f"Morphological detection found {len(raw_regions)} text blocks")

        for x, y, bw, bh in raw_regions:
            # Add margin for context
            margin = 5
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + bw + margin)
            y2 = min(h, y + bh + margin)

            # Classify each region
            region_gray = gray[y1:y2, x1:x2]
            is_hw, hw_score = _classify_region_handwriting(region_gray)

            region = DocumentRegion(
                x1=x1, y1=y1, x2=x2, y2=y2,
                region_type="handwriting" if is_hw else "text",
                confidence=hw_score,
                is_handwritten=is_hw,
                hw_score=hw_score,
            )
            regions.append(region)

    # If no regions detected at all, treat entire image as one region
    if not regions:
        is_hw, hw_score = _classify_region_handwriting(gray)
        regions = [DocumentRegion(
            x1=0, y1=0, x2=w, y2=h,
            region_type="handwriting" if is_hw else "text",
            confidence=hw_score,
            is_handwritten=is_hw,
            hw_score=hw_score,
        )]

    # Compute summary stats
    hw_regions = [r for r in regions if r.is_handwritten]
    printed_regions = [r for r in regions if not r.is_handwritten]

    total_area = sum(r.area for r in regions) or 1
    hw_area = sum(r.area for r in hw_regions)
    hw_ratio = hw_area / total_area

    has_hw = len(hw_regions) > 0
    has_printed = len(printed_regions) > 0
    is_mixed = has_hw and has_printed

    result = {
        "regions": regions,
        "has_handwriting": has_hw,
        "has_printed": has_printed,
        "is_mixed": is_mixed,
        "handwriting_ratio": round(hw_ratio, 3),
        "detection_method": detection_method,
        "model_type": model_type_used,
        "region_count": len(regions),
        "handwriting_regions": len(hw_regions),
        "printed_regions": len(printed_regions),
    }

    logger.info(f"Document analysis: {len(regions)} regions, "
                f"{len(hw_regions)} handwritten, {len(printed_regions)} printed, "
                f"hw_ratio={hw_ratio:.1%}, method={detection_method}")

    return result


def _detect_with_yolo(model, model_type: str,
                      pil_image: Image.Image,
                      gray: np.ndarray) -> list:
    """
    Run YOLO model and convert detections to DocumentRegions.

    For 'custom' models: expects document layout classes
    For 'general' models: uses YOLO to find object regions, then
    applies handwriting heuristics on the text areas between them
    """
    h, w = gray.shape[:2]
    regions = []

    # Run inference
    results = model(pil_image, verbose=False, conf=0.25)

    if model_type == "custom":
        # Custom model with document-specific classes
        # Expected classes: "printed_text", "handwritten_text", "table", "figure", etc.
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, "unknown")

                # Map class names to our types
                is_hw = cls_name.lower() in ("handwritten", "handwritten_text",
                                              "handwriting", "manuscript")
                region_type = "handwriting" if is_hw else cls_name.lower()

                # Also run heuristic for additional confidence
                region_gray = gray[max(0, y1):min(h, y2),
                                   max(0, x1):min(w, x2)]
                _, hw_score = _classify_region_handwriting(region_gray)

                # Combine YOLO confidence with heuristic
                if is_hw:
                    combined_hw = max(hw_score, conf)
                else:
                    combined_hw = hw_score

                regions.append(DocumentRegion(
                    x1=max(0, x1), y1=max(0, y1),
                    x2=min(w, x2), y2=min(h, y2),
                    region_type=region_type,
                    confidence=conf,
                    is_handwritten=is_hw or combined_hw > 0.5,
                    hw_score=combined_hw,
                ))

    else:
        # General YOLO model (yolov8n) — use it to find non-text objects,
        # then the remaining space is text regions to classify
        non_text_boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, "unknown")

                # Track non-text objects (photos, diagrams detected by YOLO)
                non_text_boxes.append((x1, y1, x2, y2, cls_name, conf))

        # Use morphological detection for text regions
        text_blocks = detect_text_regions(gray)

        for x, y, bw, bh in text_blocks:
            margin = 5
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + bw + margin)
            y2 = min(h, y + bh + margin)

            # Check if this region overlaps with YOLO-detected objects
            is_object = False
            for ox1, oy1, ox2, oy2, cls_name, _ in non_text_boxes:
                # Calculate overlap
                ix1 = max(x1, ox1)
                iy1 = max(y1, oy1)
                ix2 = min(x2, ox2)
                iy2 = min(y2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    region_area = (x2 - x1) * (y2 - y1)
                    if overlap_area > region_area * 0.5:
                        is_object = True
                        break

            if is_object:
                continue

            # Classify the text region
            region_gray = gray[y1:y2, x1:x2]
            is_hw, hw_score = _classify_region_handwriting(region_gray)

            regions.append(DocumentRegion(
                x1=x1, y1=y1, x2=x2, y2=y2,
                region_type="handwriting" if is_hw else "text",
                confidence=hw_score,
                is_handwritten=is_hw,
                hw_score=hw_score,
            ))

    # Sort regions top-to-bottom
    regions.sort(key=lambda r: (r.y1, r.x1))
    return regions


# ──────────────────────────────────────────────
# Region-based extraction helpers
# ──────────────────────────────────────────────

def get_handwritten_regions(pil_image: Image.Image) -> list:
    """Get only the handwritten regions from a document image."""
    result = detect_document_regions(pil_image)
    return [r for r in result["regions"] if r.is_handwritten]


def get_printed_regions(pil_image: Image.Image) -> list:
    """Get only the printed text regions from a document image."""
    result = detect_document_regions(pil_image)
    return [r for r in result["regions"] if not r.is_handwritten]


def merge_overlapping_regions(regions: list, overlap_threshold: float = 0.3) -> list:
    """Merge regions that significantly overlap."""
    if len(regions) <= 1:
        return regions

    merged = []
    used = set()

    for i, r1 in enumerate(regions):
        if i in used:
            continue

        current = r1
        for j, r2 in enumerate(regions):
            if j <= i or j in used:
                continue

            # Calculate overlap
            ix1 = max(current.x1, r2.x1)
            iy1 = max(current.y1, r2.y1)
            ix2 = min(current.x2, r2.x2)
            iy2 = min(current.y2, r2.y2)

            if ix1 < ix2 and iy1 < iy2:
                overlap_area = (ix2 - ix1) * (iy2 - iy1)
                min_area = min(current.area, r2.area)
                if overlap_area > min_area * overlap_threshold:
                    # Merge: expand bounding box, keep highest hw_score
                    current = DocumentRegion(
                        x1=min(current.x1, r2.x1),
                        y1=min(current.y1, r2.y1),
                        x2=max(current.x2, r2.x2),
                        y2=max(current.y2, r2.y2),
                        region_type=current.region_type if current.hw_score >= r2.hw_score else r2.region_type,
                        confidence=max(current.confidence, r2.confidence),
                        is_handwritten=current.is_handwritten or r2.is_handwritten,
                        hw_score=max(current.hw_score, r2.hw_score),
                    )
                    used.add(j)

        merged.append(current)

    return merged
