"""
/api/ocr — Persian text detection + recognition endpoints
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os, uuid

from app.services.ocr_service import OCRService
from app.utils.file_utils import allowed_file, save_upload

ocr_bp = Blueprint("ocr", __name__)
_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(
            yolo_weights=current_app.config["YOLO_WEIGHTS"],
            crnn_weights=current_app.config["CRNN_WEIGHTS"],
            cnn_weights=current_app.config["CNN_WEIGHTS"],
            conf_threshold=current_app.config["YOLO_CONF_THRESHOLD"],
            img_size=current_app.config["YOLO_IMG_SIZE"],
            cnn_confidence_threshold=current_app.config["CNN_CONF_THRESHOLD"],
        )
    return _ocr_service


# ── POST /api/ocr/recognize ───────────────────────────────────────────────────
@ocr_bp.route("/recognize", methods=["POST"])
def recognize():
    """
    Accepts a multipart image upload.
    Returns detected bounding boxes + recognized Persian text for each region.

    Response shape:
    {
      "success": true,
      "persian_text": "سلام دنیا",
      "regions": [
        {
          "box": [x1, y1, x2, y2],
          "confidence": 0.91,
          "text": "سلام"
        }, ...
      ],
      "annotated_image_url": "/api/ocr/result/<job_id>"
    }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(
        file.filename, current_app.config["ALLOWED_EXTENSIONS"]
    ):
        return jsonify({"success": False, "error": "Invalid or unsupported file"}), 400

    job_id   = str(uuid.uuid4())
    img_path = save_upload(file, current_app.config["UPLOAD_FOLDER"], job_id)

    try:
        result = get_ocr_service().run(img_path, job_id)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        current_app.logger.exception("OCR pipeline failed")
        return jsonify({"success": False, "error": str(exc)}), 500


# ── GET /api/ocr/result/<job_id> ──────────────────────────────────────────────
@ocr_bp.route("/result/<job_id>", methods=["GET"])
def get_result_image(job_id: str):
    """Serve the annotated output image for a given job."""
    from flask import send_from_directory
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    filename   = f"{job_id}_annotated.jpg"
    return send_from_directory(upload_dir, filename)