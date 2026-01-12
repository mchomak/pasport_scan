"""PDF processing service."""
import io
from typing import List
import fitz  # PyMuPDF
from PIL import Image
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PdfProcessor:
    """Service for processing PDF files and extracting pages as images."""

    @staticmethod
    def extract_pages_as_images(pdf_bytes: bytes) -> List[tuple[bytes, int]]:
        """
        Extract all pages from PDF as JPEG images.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            List of tuples (image_bytes, page_index)
        """
        pages = []

        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            logger.info("Processing PDF", page_count=len(pdf_document))

            for page_index in range(len(pdf_document)):
                try:
                    page = pdf_document[page_index]

                    # Render page to pixmap with specified DPI
                    # DPI affects quality: higher = better quality but larger file
                    zoom = settings.pdf_render_dpi / 72  # 72 is default DPI
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)

                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Save as JPEG
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=95)
                    image_bytes = output.getvalue()

                    pages.append((image_bytes, page_index))

                    logger.info(
                        "Extracted PDF page",
                        page=page_index + 1,
                        size=len(image_bytes),
                        dimensions=f"{pix.width}x{pix.height}"
                    )

                except Exception as e:
                    logger.error("Failed to extract PDF page", page=page_index, error=str(e))
                    continue

            pdf_document.close()

        except Exception as e:
            logger.error("Failed to process PDF", error=str(e))
            raise

        return pages

    @staticmethod
    def is_valid_pdf(pdf_bytes: bytes) -> bool:
        """
        Check if bytes represent a valid PDF.

        Args:
            pdf_bytes: Bytes to check

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(pdf_document)
            pdf_document.close()
            return page_count > 0
        except Exception:
            return False
