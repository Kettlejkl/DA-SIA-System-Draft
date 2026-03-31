"""
Flask Configuration
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ── General ──────────────────────────────────────────────
    SECRET_KEY          = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
    DEBUG               = os.getenv("FLASK_DEBUG", "0") == "1"

    # ── Upload ────────────────────────────────────────────────
    UPLOAD_FOLDER       = str(BASE_DIR / "uploads")
    MAX_CONTENT_LENGTH  = 16 * 1024 * 1024          # 16 MB
    ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}

    # ── CORS ──────────────────────────────────────────────────
    ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

    # ── ML Models ─────────────────────────────────────────────
    YOLO_WEIGHTS        = str(BASE_DIR / "models" / "weights" / "persian_text_detector.pt")
    CRNN_WEIGHTS        = str(BASE_DIR / "models" / "weights" / "persian_crnn.pt")
    YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.4"))
    YOLO_IMG_SIZE       = int(os.getenv("YOLO_IMG_SIZE", "640"))

    # ── Translation API ───────────────────────────────────────
    # Plug in any provider: LibreTranslate, Google, DeepL, etc.
    TRANSLATION_PROVIDER  = os.getenv("TRANSLATION_PROVIDER", "libretranslate")
    LIBRETRANSLATE_URL    = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com")
    LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "")

    GOOGLE_TRANSLATE_KEY  = os.getenv("GOOGLE_TRANSLATE_KEY", "")
    DEEPL_API_KEY         = os.getenv("DEEPL_API_KEY", "")

    # ── Target Languages (extendable) ─────────────────────────
    SUPPORTED_TARGETS = ["en", "tl"]   # English, Tagalog


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
