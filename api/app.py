# api/app.py
from app import create_app
from app.config import DevelopmentConfig, ProductionConfig
import os

config = ProductionConfig if os.getenv("FLASK_ENV") == "production" else DevelopmentConfig
app = create_app(config)