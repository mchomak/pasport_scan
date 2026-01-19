"""OCR provider interface."""
from abc import ABC, abstractmethod
from typing import Literal
from ocr.models import OcrResult
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OcrProvider(ABC):
    """Abstract OCR provider interface."""

    @abstractmethod
    async def recognize_passport(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG", "PDF"]
    ) -> OcrResult:
        """
        Recognize passport from image bytes.

        Args:
            image_bytes: Image data in bytes
            mime_type: MIME type (JPEG, PNG, or PDF)

        Returns:
            OcrResult with extracted passport data
        """
        pass


def get_ocr_provider() -> OcrProvider:
    """
    Get OCR provider instance based on configuration.

    Returns:
        OcrProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_model = settings.ocr_provider_model

    if provider_model == "yandex":
        from ocr.yandex import YandexOcrProvider
        return YandexOcrProvider()
    else:
        raise ValueError(f"OCR provider '{provider_model}' is not supported")
