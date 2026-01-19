"""Image processing service."""
import io
from typing import Literal
from PIL import Image, ImageOps
import cv2
import numpy as np
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Service for processing and preparing images for OCR."""

    @staticmethod
    def normalize_image(image_bytes: bytes) -> tuple[bytes, Literal["JPEG", "PNG"]]:
        """
        Normalize image: fix orientation, convert to RGB, resize if needed.

        Args:
            image_bytes: Input image bytes

        Returns:
            Tuple of (processed_bytes, mime_type)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))

            # Fix EXIF orientation
            image = ImageOps.exif_transpose(image) or image

            # Convert to RGB
            if image.mode not in ["RGB", "L"]:
                image = image.convert("RGB")

            # Check megapixels
            width, height = image.size
            megapixels = (width * height) / 1_000_000

            if megapixels > settings.ocr_max_megapixels:
                # Calculate new dimensions
                scale = (settings.ocr_max_megapixels / megapixels) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)

                logger.info(
                    "Resizing image",
                    original_size=f"{width}x{height}",
                    new_size=f"{new_width}x{new_height}",
                    megapixels=megapixels
                )

                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to bytes with quality adjustment
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95, optimize=True)
            processed_bytes = output.getvalue()

            # Check file size
            if len(processed_bytes) > settings.ocr_max_file_bytes:
                logger.info(
                    "Compressing image",
                    original_size=len(processed_bytes),
                    max_size=settings.ocr_max_file_bytes
                )

                # Try lower quality
                for quality in [85, 75, 65, 55]:
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=quality, optimize=True)
                    processed_bytes = output.getvalue()

                    if len(processed_bytes) <= settings.ocr_max_file_bytes:
                        logger.info("Compressed successfully", quality=quality, size=len(processed_bytes))
                        break

            return processed_bytes, "JPEG"

        except Exception as e:
            logger.error("Image normalization failed", error=str(e))
            raise

    @staticmethod
    def rotate_image(image_bytes: bytes, angle: int) -> bytes:
        """
        Rotate image by specified angle.

        Args:
            image_bytes: Input image bytes
            angle: Rotation angle (90, 180, 270)

        Returns:
            Rotated image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            if angle == 90:
                rotated = image.rotate(-90, expand=True)
            elif angle == 270:
                rotated = image.rotate(90, expand=True)
            elif angle == 180:
                rotated = image.rotate(180, expand=True)
            else:
                rotated = image

            output = io.BytesIO()
            rotated.save(output, format="JPEG", quality=95)
            return output.getvalue()

        except Exception as e:
            logger.error("Image rotation failed", error=str(e), angle=angle)
            raise

    @staticmethod
    def enhance_for_ocr(image_bytes: bytes) -> bytes:
        """
        Apply minimal preprocessing to enhance OCR accuracy.

        Args:
            image_bytes: Input image bytes

        Returns:
            Enhanced image bytes
        """
        try:
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply slight gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Encode back to bytes
            _, encoded = cv2.imencode('.jpg', blurred)
            return encoded.tobytes()

        except Exception as e:
            logger.error("Image enhancement failed", error=str(e))
            # Return original on error
            return image_bytes
