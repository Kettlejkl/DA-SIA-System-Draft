"""File upload helpers."""

import os
from pathlib import Path
from werkzeug.datastructures import FileStorage


def allowed_file(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def save_upload(file: FileStorage, upload_dir: str, job_id: str) -> str:
    """Save an uploaded file; returns the full path."""
    os.makedirs(upload_dir, exist_ok=True)
    ext  = Path(file.filename).suffix.lower()
    name = f"{job_id}{ext}"
    path = os.path.join(upload_dir, name)
    file.save(path)
    return path
