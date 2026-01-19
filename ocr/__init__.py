"""OCR module."""
from .models import OcrResult, PassportData
from .provider import OcrProvider, get_ocr_provider

__all__ = ["OcrResult", "PassportData", "OcrProvider", "get_ocr_provider"]
