# api/app.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import create_app
from app.config import ProductionConfig

app = create_app(ProductionConfig)