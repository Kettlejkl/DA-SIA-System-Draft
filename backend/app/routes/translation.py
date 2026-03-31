"""
/api/translate — Translation middleware layer
Supports multiple providers; target languages are extensible (en, tl, …)
"""

from flask import Blueprint, request, jsonify, current_app
from app.services.translation_service import TranslationService

translation_bp  = Blueprint("translation", __name__)
_trans_service: TranslationService | None = None


def get_translation_service() -> TranslationService:
    global _trans_service
    if _trans_service is None:
        _trans_service = TranslationService(
            provider=current_app.config["TRANSLATION_PROVIDER"],
            libretranslate_url=current_app.config["LIBRETRANSLATE_URL"],
            libretranslate_key=current_app.config["LIBRETRANSLATE_API_KEY"],
            google_key=current_app.config["GOOGLE_TRANSLATE_KEY"],
            deepl_key=current_app.config["DEEPL_API_KEY"],
        )
    return _trans_service


# ── POST /api/translate ───────────────────────────────────────────────────────
@translation_bp.route("/", methods=["POST"])
def translate():
    """
    Body: { "text": "سلام", "target": "en" }
    Returns: { "success": true, "translated": "Hello", "source": "fa", "target": "en" }
    """
    body = request.get_json(silent=True) or {}
    text   = body.get("text", "").strip()
    target = body.get("target", "en").strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    supported = current_app.config["SUPPORTED_TARGETS"]
    if target not in supported:
        return jsonify({
            "success": False,
            "error": f"Unsupported target. Choose from: {supported}"
        }), 400

    try:
        translated = get_translation_service().translate(text, source="fa", target=target)
        return jsonify({
            "success": True,
            "translated": translated,
            "source": "fa",
            "target": target,
        }), 200
    except Exception as exc:
        current_app.logger.exception("Translation failed")
        return jsonify({"success": False, "error": str(exc)}), 500


# ── GET /api/translate/languages ─────────────────────────────────────────────
@translation_bp.route("/languages", methods=["GET"])
def list_languages():
    """Returns the list of supported target languages."""
    return jsonify({
        "source": "fa",
        "targets": current_app.config["SUPPORTED_TARGETS"],
    })
