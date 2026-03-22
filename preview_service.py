"""
Document preview generator.

Generates thumbnail images from uploaded PDFs and images.
- PDFs: renders first page at medium DPI
- Images: resizes to thumbnail
- Multi-page PDFs: generates preview for each page

Previews are saved as PNG files in the previews/ directory.
"""

import os
import logging
from PIL import Image
import pdfplumber

from config import PREVIEW_DIR

logger = logging.getLogger(__name__)

# Preview settings
PREVIEW_MAX_WIDTH = 800
PREVIEW_MAX_HEIGHT = 1100
THUMBNAIL_WIDTH = 280
THUMBNAIL_HEIGHT = 400
PDF_RENDER_DPI = 150


def generate_preview(file_path: str, job_id: str, file_type: str) -> dict:
    """
    Generate preview thumbnail(s) for a document.

    Args:
        file_path: Path to the uploaded file
        job_id: Processing job ID (used as filename)
        file_type: 'pdf' or 'image'

    Returns:
        {
            "preview_available": True/False,
            "pages": [{"page": 1, "path": "...", "width": ..., "height": ...}],
            "thumbnail": "path to small thumbnail"
        }
    """
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    try:
        if file_type == "pdf":
            return _preview_pdf(file_path, job_id)
        else:
            return _preview_image(file_path, job_id)
    except Exception as e:
        logger.warning(f"Preview generation failed for {job_id}: {e}")
        return {"preview_available": False, "pages": [], "thumbnail": None}


def _preview_pdf(file_path: str, job_id: str) -> dict:
    """Generate preview images from PDF pages."""
    pages_info = []

    try:
        with pdfplumber.open(file_path) as pdf:
            # Preview up to 10 pages
            max_pages = min(len(pdf.pages), 10)

            for i in range(max_pages):
                page = pdf.pages[i]

                # Render page as image
                page_img = page.to_image(resolution=PDF_RENDER_DPI)
                pil_image = page_img.original

                # Save full preview
                preview_path = os.path.join(PREVIEW_DIR, f"{job_id}_page{i+1}.png")
                resized = _resize_to_fit(pil_image, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
                resized.save(preview_path, "PNG", optimize=True)

                pages_info.append({
                    "page": i + 1,
                    "path": preview_path,
                    "width": resized.width,
                    "height": resized.height,
                })

            # Generate small thumbnail from first page
            thumb_path = os.path.join(PREVIEW_DIR, f"{job_id}_thumb.png")
            if pages_info:
                first_page = Image.open(pages_info[0]["path"])
                thumb = _resize_to_fit(first_page, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
                thumb.save(thumb_path, "PNG", optimize=True)
            else:
                thumb_path = None

            logger.info(f"Generated {len(pages_info)} preview page(s) for PDF {job_id}")

            return {
                "preview_available": True,
                "pages": pages_info,
                "thumbnail": thumb_path,
                "total_pages": len(pdf.pages),
            }

    except Exception as e:
        logger.warning(f"PDF preview failed for {job_id}: {e}")
        return {"preview_available": False, "pages": [], "thumbnail": None}


def _preview_image(file_path: str, job_id: str) -> dict:
    """Generate preview from an image file."""
    try:
        img = Image.open(file_path)

        # Convert RGBA to RGB for PNG saving
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Save full preview
        preview_path = os.path.join(PREVIEW_DIR, f"{job_id}_page1.png")
        resized = _resize_to_fit(img, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
        resized.save(preview_path, "PNG", optimize=True)

        # Save thumbnail
        thumb_path = os.path.join(PREVIEW_DIR, f"{job_id}_thumb.png")
        thumb = _resize_to_fit(img, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        thumb.save(thumb_path, "PNG", optimize=True)

        logger.info(f"Generated preview for image {job_id} ({img.size[0]}x{img.size[1]})")

        return {
            "preview_available": True,
            "pages": [{
                "page": 1,
                "path": preview_path,
                "width": resized.width,
                "height": resized.height,
            }],
            "thumbnail": thumb_path,
            "total_pages": 1,
        }

    except Exception as e:
        logger.warning(f"Image preview failed for {job_id}: {e}")
        return {"preview_available": False, "pages": [], "thumbnail": None}


def _resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize image to fit within max dimensions, preserving aspect ratio."""
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img.copy()

    ratio = min(max_w / w, max_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)


def get_preview_path(job_id: str, page: int = 1) -> str:
    """Get the path to a preview image. Returns None if not found."""
    path = os.path.join(PREVIEW_DIR, f"{job_id}_page{page}.png")
    if os.path.exists(path):
        return path
    return None


def get_thumbnail_path(job_id: str) -> str:
    """Get the path to a thumbnail. Returns None if not found."""
    path = os.path.join(PREVIEW_DIR, f"{job_id}_thumb.png")
    if os.path.exists(path):
        return path
    return None


def delete_previews(job_id: str):
    """Delete all preview files for a job."""
    import glob
    pattern = os.path.join(PREVIEW_DIR, f"{job_id}_*.png")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass


def get_preview_page_count(job_id: str) -> int:
    """Count how many preview pages exist for a job."""
    import glob
    pattern = os.path.join(PREVIEW_DIR, f"{job_id}_page*.png")
    return len(glob.glob(pattern))
