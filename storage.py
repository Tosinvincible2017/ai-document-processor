import os
import json
import logging
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def save_result(job_id: str, result: dict) -> str:
    """Save processing result to output/{job_id}.json. Returns file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved result: {file_path}")
    return file_path


def load_result(job_id: str) -> dict | None:
    """Load a result by job_id. Returns None if not found."""
    file_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load result {job_id}: {e}")
        return None


def list_results(limit: int = 50, offset: int = 0) -> list:
    """List all results sorted by created_at descending."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".json"):
            continue
        file_path = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by created_at descending
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results[offset:offset + limit]


def delete_result(job_id: str) -> bool:
    """Delete a result file. Returns True if found and deleted."""
    file_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted result: {job_id}")
        return True
    return False
