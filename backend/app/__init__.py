"""
Persian OCR - Flask Application Factory
CIT 404A Final Project - TIP Manila 2026
Team: Carbonel, Estacio, Rosales
"""

from flask import Flask
from flask_cors import CORS

from app.routes.ocr import ocr_bp
from app.routes.translation import translation_bp
from app.routes.health import health_bp
from app.middleware.request_logger import RequestLogger
from app.config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # --- CORS (allow React dev server) ---
    CORS(app, resources={r"/api/*": {"origins": app.config["ALLOWED_ORIGINS"]}})

    # --- Middleware ---
    RequestLogger(app)

    # --- Blueprints ---
    app.register_blueprint(health_bp,      url_prefix="/api")
    app.register_blueprint(ocr_bp,         url_prefix="/api/ocr")
    app.register_blueprint(translation_bp, url_prefix="/api/translate")

    return app
