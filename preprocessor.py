"""
Advanced image preprocessing for OCR accuracy — V2.

Pipeline:  orient → perspective → deskew → shadow removal → denoise → binarize → clean → upscale

Key V2 improvements:
- Auto-rotation (0/90/180/270) using Tesseract OSD
- Perspective correction for phone-camera photos
- Shadow & illumination normalization
- Table/grid line removal for invoices
- Morphological cleanup (fix broken/touching chars)
- Diagnostic mode to save every preprocessing stage
- Smarter profile selection with image quality scoring
"""

import os
import uuid
import math
import logging
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, BASE_DIR

logger = logging.getLogger(__name__)

# Directory for diagnostic images (created on demand)
DIAG_DIR = os.path.join(BASE_DIR, "diagnostics")


# ──────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────

def validate_file(filename: str, file_bytes: bytes) -> tuple:
    """Validate file extension and size. Returns (ok, error_message)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"

    return True, ""


def detect_file_type(filename: str) -> str:
    """Returns 'pdf' or 'image' based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    return "pdf" if ext == ".pdf" else "image"


def save_upload(filename: str, file_bytes: bytes) -> str:
    """Save uploaded file to uploads/ with UUID prefix. Returns absolute path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved upload: {file_path} ({len(file_bytes)} bytes)")
    return file_path


# ──────────────────────────────────────────────
# PIL <-> OpenCV conversion
# ──────────────────────────────────────────────

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR/grayscale numpy array."""
    if pil_image.mode == "L":
        return np.array(pil_image)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV array back to PIL Image."""
    if len(cv_image.shape) == 2:
        return Image.fromarray(cv_image)
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────
# Diagnostic helper
# ──────────────────────────────────────────────

class DiagnosticTracker:
    """Saves intermediate preprocessing images for debugging."""

    def __init__(self, enabled: bool = False, job_id: str = ""):
        self.enabled = enabled
        self.job_id = job_id or uuid.uuid4().hex[:8]
        self.step = 0
        self.stages = []

    def save(self, name: str, img: np.ndarray):
        if not self.enabled:
            return
        os.makedirs(DIAG_DIR, exist_ok=True)
        self.step += 1
        fname = f"{self.job_id}_{self.step:02d}_{name}.png"
        path = os.path.join(DIAG_DIR, fname)
        cv2.imwrite(path, img)
        self.stages.append({"step": self.step, "name": name, "path": path})
        logger.debug(f"Diagnostic saved: {fname}")


# ──────────────────────────────────────────────
# Basic operations
# ──────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def estimate_noise(gray: np.ndarray) -> float:
    """Estimate image noise level using Laplacian variance."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def estimate_blur(gray: np.ndarray) -> float:
    """Estimate blur level. Lower = more blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_contrast(gray: np.ndarray) -> float:
    """Estimate contrast as standard deviation of pixel values."""
    return float(gray.std())


# ──────────────────────────────────────────────
# NEW: Auto-rotation (orientation detection)
# ──────────────────────────────────────────────

def auto_rotate(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct 90/180/270 degree rotation using Tesseract OSD.
    Falls back to no rotation if OSD fails.
    """
    try:
        import pytesseract
        from config import TESSERACT_CMD
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        # Need a PIL image for pytesseract
        pil_img = cv2_to_pil(gray)
        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)

        if rotation and rotation != 0:
            logger.info(f"OSD detected rotation: {rotation} degrees — correcting")
            if rotation == 90:
                return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                return cv2.rotate(gray, cv2.ROTATE_180)
            elif rotation == 270:
                return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    except Exception as e:
        logger.debug(f"OSD rotation detection failed (normal for small images): {e}")

    return gray


# ──────────────────────────────────────────────
# NEW: Perspective correction (for phone photos)
# ──────────────────────────────────────────────

def correct_perspective(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct perspective distortion from phone-camera photos.
    Finds the document quadrilateral and warps it to a rectangle.
    """
    h, w = gray.shape[:2]

    # Only attempt on larger images (phone photos are usually high-res)
    if w < 500 or h < 500:
        return gray

    # Blur and find edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    # Find the largest quadrilateral contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Must be a quadrilateral and cover >25% of image area
        if len(approx) == 4 and cv2.contourArea(approx) > (h * w * 0.25):
            doc_contour = approx
            break

    if doc_contour is None:
        return gray

    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = doc_contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)

    # Check if perspective correction is actually needed
    # (skip if the document is already mostly rectangular)
    if _is_mostly_rectangular(rect, w, h):
        return gray

    # Compute destination dimensions
    width_top = np.linalg.norm(rect[0] - rect[1])
    width_bot = np.linalg.norm(rect[3] - rect[2])
    height_left = np.linalg.norm(rect[0] - rect[3])
    height_right = np.linalg.norm(rect[1] - rect[2])

    new_w = int(max(width_top, width_bot))
    new_h = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [new_w - 1, 0],
        [new_w - 1, new_h - 1],
        [0, new_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    logger.info(f"Perspective corrected: {w}x{h} -> {new_w}x{new_h}")
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
    return rect


def _is_mostly_rectangular(rect: np.ndarray, img_w: int, img_h: int) -> bool:
    """Check if the quadrilateral is close enough to a rectangle to skip warping."""
    # Compare each corner to image corners
    img_corners = np.array([
        [0, 0], [img_w, 0], [img_w, img_h], [0, img_h]
    ], dtype=np.float32)

    # If all corners are within 5% of image dimensions, it's already rectangular
    threshold = max(img_w, img_h) * 0.05
    distances = np.linalg.norm(rect - img_corners, axis=1)
    return np.all(distances < threshold)


# ──────────────────────────────────────────────
# NEW: Shadow & illumination normalization
# ──────────────────────────────────────────────

def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Remove shadows and normalize uneven lighting using morphological background estimation.
    Critical for phone photos with shadows, desk lamp lighting, etc.
    """
    h, w = gray.shape[:2]

    # Estimate background illumination using large morphological closing
    kernel_size = max(w, h) // 10
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 51)  # minimum size

    # Use morphological closing to estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide original by background to normalize
    # This removes shadows while preserving text
    normalized = cv2.divide(gray, background, scale=255)

    # Check if normalization actually helped (higher contrast = better)
    if normalized.std() > gray.std() * 0.8:
        logger.info("Applied illumination normalization (shadow removal)")
        return normalized

    return gray


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


# ──────────────────────────────────────────────
# NEW: Table/grid line removal
# ──────────────────────────────────────────────

def remove_table_lines(gray: np.ndarray) -> np.ndarray:
    """
    Detect and remove horizontal/vertical table grid lines.
    Critical for invoice OCR — grid lines confuse Tesseract badly.
    """
    h, w = gray.shape[:2]

    # Binarize for line detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines
    horiz_kernel_len = max(w // 15, 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Detect vertical lines
    vert_kernel_len = max(h // 15, 40)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    # Combine all lines
    all_lines = cv2.add(horiz_lines, vert_lines)

    # Check if significant lines were found
    line_pixels = cv2.countNonZero(all_lines)
    total_pixels = h * w
    line_ratio = line_pixels / total_pixels

    if line_ratio < 0.005:  # Less than 0.5% — probably no table
        return gray

    # Remove lines by painting them white (inpainting would be better but slower)
    # Dilate lines slightly to catch anti-aliased edges
    dilated_lines = cv2.dilate(all_lines, np.ones((3, 3), np.uint8), iterations=1)

    # Where lines are, set to white in original image
    result = gray.copy()
    result[dilated_lines > 0] = 255

    # Repair text that was partially damaged by line removal
    # Use morphological closing to reconnect broken characters
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    _, repaired_binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    repaired_binary = cv2.morphologyEx(repaired_binary, cv2.MORPH_CLOSE, repair_kernel)
    result = cv2.bitwise_not(repaired_binary)

    logger.info(f"Removed table lines (coverage: {line_ratio*100:.1f}%)")
    return result


# ──────────────────────────────────────────────
# NEW: Morphological text cleanup
# ──────────────────────────────────────────────

def fix_broken_characters(binary: np.ndarray) -> np.ndarray:
    """
    Use morphological closing to reconnect broken character strokes.
    Common in low-DPI scans or after aggressive binarization.
    """
    # Small closing to bridge 1-2 pixel gaps within characters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Work on inverted (text = white)
    inv = cv2.bitwise_not(binary)
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)


def separate_touching_characters(binary: np.ndarray) -> np.ndarray:
    """
    Use morphological opening to separate characters that are touching.
    Common in bold or low-resolution text.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    inv = cv2.bitwise_not(binary)
    opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opened)


def remove_thin_noise_lines(gray: np.ndarray) -> np.ndarray:
    """Remove thin horizontal/vertical noise lines (scan artifacts, fax lines)."""
    # Detect thin horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 3, 1))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    if cv2.countNonZero(h_lines) > 0:
        # Remove the lines
        result = gray.copy()
        result[h_lines > 0] = 255
        logger.info("Removed thin horizontal noise lines")
        return result

    return gray


# ──────────────────────────────────────────────
# Existing preprocessing steps (refined)
# ──────────────────────────────────────────────

def detect_skew_angle(gray: np.ndarray) -> float:
    """Detect skew angle using Hough line transform. Returns degrees."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 8,
                            maxLineGap=20)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return 0.0

    logger.info(f"Detected skew angle: {median_angle:.2f} degrees")
    return median_angle


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct image skew."""
    angle = detect_skew_angle(gray)
    if angle == 0.0:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    bg = 255 if gray.mean() > 128 else 0
    rotated = cv2.warpAffine(gray, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=bg)
    logger.info(f"Deskewed by {angle:.2f} degrees")
    return rotated


def remove_noise_light(gray: np.ndarray) -> np.ndarray:
    """Light denoising for clean or slightly noisy images."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def remove_noise_medium(gray: np.ndarray) -> np.ndarray:
    """Medium denoising: bilateral filter preserves edges."""
    return cv2.bilateralFilter(gray, 9, 75, 75)


def remove_noise_heavy(gray: np.ndarray) -> np.ndarray:
    """Heavy denoising for very noisy scans: Non-local means."""
    return cv2.fastNlMeansDenoising(gray, None, h=15,
                                     templateWindowSize=7,
                                     searchWindowSize=21)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding — best for uneven lighting."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Otsu's binarization — best for bimodal histograms (clean scans)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def sauvola_threshold(gray: np.ndarray, window_size: int = 25,
                      k: float = 0.2) -> np.ndarray:
    """Sauvola binarization — excellent for documents with varying background."""
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
    mean_sq = cv2.blur((gray.astype(np.float64)) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def remove_borders(gray: np.ndarray) -> np.ndarray:
    """Remove black borders/scan artifacts from edges."""
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return gray

    all_points = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_points)

    margin = 10
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(w, x + bw + margin)
    y_end = min(h, y + bh + margin)

    if (x_start > w * 0.05 or y_start > h * 0.05 or
            x_end < w * 0.95 or y_end < h * 0.95):
        cropped = gray[y_start:y_end, x_start:x_end]
        logger.info(f"Removed borders: {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    return gray


def remove_small_components(binary: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Remove tiny connected components (specks/dust) from binary image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(binary), connectivity=8
    )
    cleaned = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 255
    return cleaned


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask sharpening to recover blurry text edges."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def upscale_if_needed(gray: np.ndarray, min_dim: int = 1500) -> np.ndarray:
    """Upscale image if too small for reliable OCR."""
    h, w = gray.shape[:2]
    if w >= min_dim and h >= min_dim:
        return gray

    scale = max(min_dim / w, min_dim / h, 1.0)
    if scale <= 1.0:
        return gray

    # Use INTER_CUBIC for upscaling (better than INTER_LINEAR for text)
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.info(f"Upscaled: {w}x{h} -> {new_w}x{new_h} ({scale:.1f}x)")
    return upscaled


def invert_if_dark_background(gray: np.ndarray) -> np.ndarray:
    """Invert image if background is dark (white text on dark bg)."""
    if gray.mean() < 128:
        logger.info("Dark background detected — inverting image")
        return cv2.bitwise_not(gray)
    return gray


# ──────────────────────────────────────────────
# Image quality analysis (enhanced)
# ──────────────────────────────────────────────

def analyze_image(pil_image: Image.Image) -> dict:
    """Comprehensive image quality analysis to guide preprocessing."""
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)
    h, w = gray.shape

    noise_level = estimate_noise(gray)
    blur_level = estimate_blur(gray)
    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())

    # Detect shadows by checking brightness variance across quadrants
    quad_h, quad_w = h // 2, w // 2
    quadrants = [
        gray[:quad_h, :quad_w],
        gray[:quad_h, quad_w:],
        gray[quad_h:, :quad_w],
        gray[quad_h:, quad_w:],
    ]
    quad_means = [float(q.mean()) for q in quadrants]
    shadow_variance = float(np.std(quad_means))
    has_shadows = shadow_variance > 30

    # Detect if it's a photo (vs scan) by checking for high-frequency content at edges
    edge_region_top = gray[:max(h // 20, 5), :]
    edge_region_bot = gray[-max(h // 20, 5):, :]
    edge_noise = float(np.std(edge_region_top)) + float(np.std(edge_region_bot))
    likely_photo = edge_noise > 60 or has_shadows

    # Detect table lines
    _, bin_check = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 15, 40), 1))
    h_lines = cv2.morphologyEx(bin_check, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 15, 40)))
    v_lines = cv2.morphologyEx(bin_check, cv2.MORPH_OPEN, v_kernel, iterations=2)
    has_table_lines = (cv2.countNonZero(h_lines) + cv2.countNonZero(v_lines)) > (h * w * 0.005)

    # Detect handwriting characteristics using heuristics
    # (YOLO region detection is done separately in ocr_engine to avoid
    # double inference — it handles mixed document routing there)
    likely_handwritten = _detect_handwriting(gray)

    low_contrast = std_brightness < 40
    low_res = w < 1000 or h < 1000
    is_blurry = blur_level < 100

    analysis = {
        "width": w,
        "height": h,
        "noise_level": round(noise_level, 1),
        "blur_level": round(blur_level, 1),
        "mean_brightness": round(mean_brightness, 1),
        "std_brightness": round(std_brightness, 1),
        "shadow_variance": round(shadow_variance, 1),
        "low_contrast": low_contrast,
        "low_resolution": low_res,
        "is_blurry": is_blurry,
        "has_shadows": has_shadows,
        "has_table_lines": has_table_lines,
        "likely_photo": likely_photo,
        "likely_handwritten": likely_handwritten,
        "likely_dark_bg": mean_brightness < 128,
        "likely_noisy": noise_level > 2000,
        "likely_clean": noise_level < 500 and not low_contrast and not has_shadows,
    }

    logger.info(f"Image analysis: {w}x{h}, noise={noise_level:.0f}, blur={blur_level:.0f}, "
                f"brightness={mean_brightness:.0f}+/-{std_brightness:.0f}, "
                f"shadows={'YES' if has_shadows else 'no'}, "
                f"table_lines={'YES' if has_table_lines else 'no'}, "
                f"photo={'YES' if likely_photo else 'no'}, "
                f"handwritten={'YES' if likely_handwritten else 'no'}")
    return analysis


def _detect_handwriting_smart(gray: np.ndarray, pil_image: Image.Image = None) -> bool:
    """
    Smart handwriting detection: uses YOLO region analysis when available,
    falls back to heuristic analysis.

    YOLO provides region-level detection (can identify mixed documents),
    while heuristics provide a global assessment.
    """
    # Try YOLO-based detection first
    try:
        from yolo_detector import detect_document_regions, is_yolo_available
        if is_yolo_available() and pil_image is not None:
            result = detect_document_regions(pil_image)
            if result["region_count"] > 0:
                has_hw = result["has_handwriting"]
                hw_ratio = result["handwriting_ratio"]
                method = result["detection_method"]
                logger.info(f"YOLO handwriting detection: has_hw={has_hw}, "
                           f"ratio={hw_ratio:.1%}, method={method}")
                # Consider handwritten if >30% of area is handwritten
                return hw_ratio > 0.3
    except ImportError:
        logger.debug("YOLO detector not available, using heuristic detection")
    except Exception as e:
        logger.debug(f"YOLO detection failed, falling back to heuristics: {e}")

    # Fallback: existing heuristic detection
    return _detect_handwriting(gray)


def _detect_handwriting(gray: np.ndarray) -> bool:
    """
    Detect if an image likely contains handwriting rather than printed text.

    Heuristics:
    1. Stroke width variation — handwriting has much more variation than print
    2. Baseline irregularity — handwritten lines are not perfectly horizontal
    3. Connected component shape — handwriting has more complex, irregular shapes
    4. Angle distribution — handwriting has diverse stroke angles vs print's regularity
    """
    h, w = gray.shape[:2]

    # Too small to analyze reliably
    if w < 200 or h < 200:
        return False

    try:
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 1. Stroke width variation using distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        stroke_pixels = dist[dist > 0]
        if len(stroke_pixels) < 100:
            return False

        stroke_mean = float(np.mean(stroke_pixels))
        stroke_std = float(np.std(stroke_pixels))
        stroke_cv = stroke_std / max(stroke_mean, 0.01)  # coefficient of variation

        # 2. Connected component analysis — handwriting has more complex shapes
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels < 5:
            return False

        # Filter out tiny noise and huge blobs
        valid_areas = []
        valid_aspects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            if 50 < area < (h * w * 0.3) and cw > 3 and ch > 3:
                valid_areas.append(area)
                valid_aspects.append(cw / ch)

        if len(valid_areas) < 5:
            return False

        # Area variation — handwriting components are very irregular in size
        area_cv = float(np.std(valid_areas)) / max(float(np.mean(valid_areas)), 1)

        # Aspect ratio variation — printed chars are uniform, handwriting varies
        aspect_cv = float(np.std(valid_aspects)) / max(float(np.mean(valid_aspects)), 0.01)

        # 3. Contour complexity — handwriting has smoother, more complex curves
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        complexities = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if area > 50 and peri > 10:
                # Circularity ratio — complex shapes have lower values
                circularity = 4 * np.pi * area / (peri * peri)
                complexities.append(circularity)

        avg_complexity = float(np.mean(complexities)) if complexities else 0.5

        # 4. Horizontal line regularity — printed text has regular baselines
        # Project vertically and check for regular peaks
        v_proj = np.sum(binary, axis=1)
        v_proj_norm = v_proj / max(v_proj.max(), 1)
        # Count transitions (peaks indicate text lines)
        transitions = 0
        above = False
        for val in v_proj_norm:
            if val > 0.05 and not above:
                transitions += 1
                above = True
            elif val <= 0.02:
                above = False

        # For printed text, peaks are more regular
        if transitions > 2:
            peak_positions = np.where(v_proj_norm > 0.05)[0]
            if len(peak_positions) > 5:
                diffs = np.diff(peak_positions)
                line_regularity = float(np.std(diffs)) / max(float(np.mean(diffs)), 1)
            else:
                line_regularity = 0.5
        else:
            line_regularity = 0.3

        # Score handwriting likelihood
        hw_score = 0

        # Stroke width variation: handwriting > 0.5, print < 0.3
        if stroke_cv > 0.6:
            hw_score += 3
        elif stroke_cv > 0.45:
            hw_score += 2
        elif stroke_cv > 0.35:
            hw_score += 1

        # Component area variation
        if area_cv > 2.0:
            hw_score += 2
        elif area_cv > 1.2:
            hw_score += 1

        # Aspect ratio variation
        if aspect_cv > 0.8:
            hw_score += 2
        elif aspect_cv > 0.5:
            hw_score += 1

        # Shape complexity — handwriting typically < 0.3
        if avg_complexity < 0.2:
            hw_score += 2
        elif avg_complexity < 0.35:
            hw_score += 1

        # Line irregularity — handwriting > 0.5
        if line_regularity > 0.7:
            hw_score += 2
        elif line_regularity > 0.5:
            hw_score += 1

        is_handwritten = hw_score >= 6

        logger.debug(f"Handwriting detection: score={hw_score}/12, "
                     f"stroke_cv={stroke_cv:.2f}, area_cv={area_cv:.2f}, "
                     f"aspect_cv={aspect_cv:.2f}, complexity={avg_complexity:.2f}, "
                     f"line_reg={line_regularity:.2f} -> {'HANDWRITTEN' if is_handwritten else 'printed'}")

        return is_handwritten

    except Exception as e:
        logger.debug(f"Handwriting detection failed: {e}")
        return False


# ──────────────────────────────────────────────
# Preprocessing profiles (enhanced V2)
# ──────────────────────────────────────────────

def preprocess_clean(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: CLEAN
    For clear, well-lit documents with minimal noise.
    Pipeline: grayscale -> auto-rotate -> deskew -> light denoise -> Otsu -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)
    diag.save("clean_01_gray", gray)

    gray = invert_if_dark_background(gray)
    gray = auto_rotate(gray)
    diag.save("clean_02_rotated", gray)

    gray = deskew(gray)
    gray = remove_noise_light(gray)
    binary = otsu_threshold(gray)
    diag.save("clean_03_binary", binary)

    binary = upscale_if_needed(binary)
    return cv2_to_pil(binary)


def preprocess_noisy(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: NOISY
    For scanned documents with noise, uneven lighting, stains.
    Pipeline: grayscale -> auto-rotate -> deskew -> shadow removal -> heavy denoise
              -> adaptive threshold -> clean specks -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    gray = auto_rotate(gray)
    gray = deskew(gray)
    gray = remove_borders(gray)
    diag.save("noisy_01_borders_removed", gray)

    # NEW: shadow removal before denoising
    gray = normalize_illumination(gray)
    diag.save("noisy_02_illumination", gray)

    gray = remove_noise_heavy(gray)
    binary = adaptive_threshold(gray)
    diag.save("noisy_03_binary", binary)

    binary = remove_small_components(binary, min_area=20)
    binary = fix_broken_characters(binary)
    binary = upscale_if_needed(binary)
    return cv2_to_pil(binary)


def preprocess_receipt(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: RECEIPT
    For receipts/thermal paper — often low contrast, small text.
    Pipeline: grayscale -> sharpen -> bilateral denoise -> Sauvola -> morph cleanup -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    gray = auto_rotate(gray)
    gray = deskew(gray)
    diag.save("receipt_01_deskewed", gray)

    # NEW: shadow removal critical for receipts
    gray = normalize_illumination(gray)
    gray = sharpen(gray)
    gray = remove_noise_medium(gray)
    diag.save("receipt_02_denoised", gray)

    binary = sauvola_threshold(gray, window_size=25, k=0.15)
    binary = fix_broken_characters(binary)
    binary = remove_small_components(binary, min_area=15)
    binary = upscale_if_needed(binary, min_dim=2000)
    return cv2_to_pil(binary)


def preprocess_grayscale_only(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: GRAYSCALE
    Minimal processing — sometimes Tesseract does better on non-binarized images.
    Pipeline: grayscale -> auto-rotate -> deskew -> CLAHE -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    gray = auto_rotate(gray)
    gray = deskew(gray)
    gray = upscale_if_needed(gray)
    enhanced = apply_clahe(gray, clip_limit=2.0, grid_size=(8, 8))
    return cv2_to_pil(enhanced)


def preprocess_photo(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: PHOTO (NEW)
    For phone-camera photos of documents.
    Pipeline: grayscale -> perspective correction -> auto-rotate -> deskew
              -> shadow removal -> CLAHE -> medium denoise -> adaptive threshold
              -> morph cleanup -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    diag.save("photo_01_gray", gray)

    # Perspective correction first (before rotation/deskew)
    gray = correct_perspective(gray)
    diag.save("photo_02_perspective", gray)

    gray = auto_rotate(gray)
    gray = deskew(gray)
    diag.save("photo_03_deskewed", gray)

    # Shadow removal is critical for photos
    gray = normalize_illumination(gray)
    diag.save("photo_04_illumination", gray)

    # CLAHE for contrast, then denoise
    gray = apply_clahe(gray, clip_limit=3.0, grid_size=(8, 8))
    gray = remove_noise_medium(gray)
    diag.save("photo_05_enhanced", gray)

    binary = adaptive_threshold(gray)
    binary = fix_broken_characters(binary)
    binary = remove_small_components(binary, min_area=25)
    binary = upscale_if_needed(binary, min_dim=2000)
    diag.save("photo_06_final", binary)
    return cv2_to_pil(binary)


def preprocess_handwriting(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: HANDWRITING
    For handwritten documents — preserve stroke details, avoid aggressive binarization.
    Pipeline: grayscale -> auto-rotate -> perspective correction -> deskew
              -> shadow removal -> CLAHE (gentle) -> light denoise -> Sauvola (gentle)
              -> upscale (higher res for stroke details)

    Key differences from printed text profiles:
    - Gentler binarization (Sauvola with lower k to preserve thin strokes)
    - No aggressive noise removal (strokes can look like noise)
    - Higher upscaling (handwriting needs more resolution)
    - No morphological opening (would break connected handwriting)
    - CLAHE with lower clip limit to avoid amplifying pen pressure variations
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    diag.save("hw_01_gray", gray)

    # Perspective correction (common for phone photos of handwritten notes)
    gray = correct_perspective(gray)
    gray = auto_rotate(gray)
    gray = deskew(gray)
    diag.save("hw_02_deskewed", gray)

    # Gentle shadow removal
    gray = normalize_illumination(gray)
    diag.save("hw_03_illumination", gray)

    # Gentle contrast enhancement — don't amplify pen pressure variations
    gray = apply_clahe(gray, clip_limit=1.5, grid_size=(16, 16))

    # Very light denoising — preserve stroke edges
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    diag.save("hw_04_enhanced", gray)

    # Sauvola with low k value — preserves thin strokes better than Otsu
    binary = sauvola_threshold(gray, window_size=31, k=0.1)
    diag.save("hw_05_binary", binary)

    # Only fix broken chars, NO touching char separation (would break cursive)
    binary = fix_broken_characters(binary)

    # Remove only large noise blobs, keep small strokes
    binary = remove_small_components(binary, min_area=10)

    # Higher resolution for handwriting — strokes need detail
    binary = upscale_if_needed(binary, min_dim=2500)
    diag.save("hw_06_final", binary)
    return cv2_to_pil(binary)


def preprocess_table(pil_image: Image.Image, diag: DiagnosticTracker = None) -> Image.Image:
    """
    Profile: TABLE (NEW)
    For documents with table/grid lines (invoices, spreadsheets).
    Pipeline: grayscale -> auto-rotate -> deskew -> remove table lines
              -> denoise -> adaptive threshold -> morph cleanup -> upscale
    """
    diag = diag or DiagnosticTracker()
    img = pil_to_cv2(pil_image)
    gray = to_grayscale(img)

    gray = invert_if_dark_background(gray)
    gray = auto_rotate(gray)
    gray = deskew(gray)
    diag.save("table_01_deskewed", gray)

    # Remove table grid lines BEFORE binarization
    gray = remove_table_lines(gray)
    diag.save("table_02_lines_removed", gray)

    gray = normalize_illumination(gray)
    gray = remove_noise_light(gray)

    binary = adaptive_threshold(gray)
    binary = fix_broken_characters(binary)
    binary = remove_small_components(binary, min_area=20)
    binary = upscale_if_needed(binary)
    diag.save("table_03_final", binary)
    return cv2_to_pil(binary)


# ──────────────────────────────────────────────
# Smart profile selection (enhanced V2)
# ──────────────────────────────────────────────

def get_preprocessing_profiles(pil_image: Image.Image,
                                enable_diagnostics: bool = False,
                                job_id: str = "") -> list:
    """
    Return ordered list of (name, preprocessed_image) tuples.
    Uses comprehensive image analysis to pick the best profiles first.

    V2 improvements:
    - Photo profile for camera-captured documents
    - Table profile for documents with grid lines
    - Smarter ordering based on multi-factor analysis
    """
    analysis = analyze_image(pil_image)
    diag = DiagnosticTracker(enabled=enable_diagnostics, job_id=job_id)
    profiles = []

    # Priority 0: Handwriting-specific handling
    if analysis["likely_handwritten"]:
        profiles.append(("handwriting", preprocess_handwriting(pil_image, diag)))

    # Priority 1: Photo-specific handling
    if analysis["likely_photo"]:
        profiles.append(("photo", preprocess_photo(pil_image, diag)))

    # Priority 2: Table line handling
    if analysis["has_table_lines"]:
        profiles.append(("table", preprocess_table(pil_image, diag)))

    # Priority 3: Based on noise/quality analysis
    if analysis["likely_clean"]:
        profiles.append(("clean", preprocess_clean(pil_image, diag)))
        profiles.append(("grayscale", preprocess_grayscale_only(pil_image, diag)))
    elif analysis["low_contrast"]:
        profiles.append(("receipt", preprocess_receipt(pil_image, diag)))
        profiles.append(("noisy", preprocess_noisy(pil_image, diag)))
    elif analysis["likely_noisy"]:
        profiles.append(("noisy", preprocess_noisy(pil_image, diag)))
        profiles.append(("clean", preprocess_clean(pil_image, diag)))
    else:
        profiles.append(("grayscale", preprocess_grayscale_only(pil_image, diag)))
        profiles.append(("clean", preprocess_clean(pil_image, diag)))
        profiles.append(("noisy", preprocess_noisy(pil_image, diag)))

    # Always include grayscale as fallback if not already present
    profile_names = [p[0] for p in profiles]
    if "grayscale" not in profile_names:
        profiles.append(("grayscale", preprocess_grayscale_only(pil_image, diag)))

    # Deduplicate by name
    seen = set()
    unique_profiles = []
    for name, img in profiles:
        if name not in seen:
            seen.add(name)
            unique_profiles.append((name, img))

    return unique_profiles


# ──────────────────────────────────────────────
# Legacy compatibility
# ──────────────────────────────────────────────

def enhance_image(image: Image.Image) -> Image.Image:
    """Legacy single-image enhancement (kept for backward compat)."""
    return preprocess_grayscale_only(image)
