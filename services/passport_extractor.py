"""Passport extraction service with rotation heuristics."""
from typing import Literal
from ocr.provider import OcrProvider
from ocr.models import OcrResult, PassportData
from services.image_processor import ImageProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class PassportExtractor:
    """Service for extracting passport data with smart rotation."""

    def __init__(self, ocr_provider: OcrProvider):
        self.ocr_provider = ocr_provider
        self.image_processor = ImageProcessor()

    async def extract_with_rotation(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG"]
    ) -> OcrResult:
        """
        Extract passport data with rotation heuristic.

        If the initial OCR doesn't find the passport number or has few fields,
        try rotating the image by 90° and 270° and pick the best result.

        Args:
            image_bytes: Normalized image bytes
            mime_type: Image MIME type

        Returns:
            Best OCR result
        """
        logger.info("Starting passport extraction with rotation heuristic")

        # Try original orientation
        result_0 = await self.ocr_provider.recognize_passport(image_bytes, mime_type)
        score_0 = result_0.passport_data.count_filled_fields()

        logger.info("OCR result for original orientation", score=score_0)

        # Check if we need to try rotations
        # Heuristic: if passport_number is empty or score is low, try rotations
        if not result_0.passport_data.passport_number or score_0 < 3:
            logger.info("Trying rotated orientations due to low confidence")

            # Try 90° rotation
            try:
                rotated_90 = self.image_processor.rotate_image(image_bytes, 90)
                result_90 = await self.ocr_provider.recognize_passport(rotated_90, mime_type)
                score_90 = result_90.passport_data.count_filled_fields()
                logger.info("OCR result for 90° rotation", score=score_90)
            except Exception as e:
                logger.error("Failed to process 90° rotation", error=str(e))
                result_90 = result_0
                score_90 = 0

            # Try 270° rotation
            try:
                rotated_270 = self.image_processor.rotate_image(image_bytes, 270)
                result_270 = await self.ocr_provider.recognize_passport(rotated_270, mime_type)
                score_270 = result_270.passport_data.count_filled_fields()
                logger.info("OCR result for 270° rotation", score=score_270)
            except Exception as e:
                logger.error("Failed to process 270° rotation", error=str(e))
                result_270 = result_0
                score_270 = 0

            # Pick the best result
            results = [
                (result_0, score_0, 0),
                (result_90, score_90, 90),
                (result_270, score_270, 270),
            ]

            best_result, best_score, best_angle = max(results, key=lambda x: x[1])

            logger.info(
                "Selected best orientation",
                angle=best_angle,
                score=best_score
            )

            return best_result
        else:
            logger.info("Original orientation is good enough", score=score_0)
            return result_0

    async def extract(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG"]
    ) -> OcrResult:
        """
        Extract passport data without rotation.

        Args:
            image_bytes: Image bytes
            mime_type: Image MIME type

        Returns:
            OCR result
        """
        return await self.ocr_provider.recognize_passport(image_bytes, mime_type)
