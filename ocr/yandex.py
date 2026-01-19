"""Yandex Cloud OCR provider implementation."""
import base64
import asyncio
from typing import Literal, Optional
from datetime import datetime
import httpx
from ocr.provider import OcrProvider
from ocr.models import OcrResult, PassportData
from config import settings
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class YandexOcrProvider(OcrProvider):
    """Yandex Cloud OCR provider."""

    def __init__(self):
        self.endpoint = settings.yc_ocr_endpoint
        self.folder_id = settings.yc_folder_id
        self.iam_token = settings.yc_iam_token
        self.rate_limiter = RateLimiter(rate=settings.ocr_rate_limit_rps)
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers - copied from ocr_test.py"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id
        }

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime.date]:
        """Parse date from various formats."""
        if not date_str:
            return None

        date_str = date_str.strip()
        if not date_str:
            return None

        # Try different date formats
        formats = [
            "%d.%m.%Y",  # 01.01.2000
            "%Y-%m-%d",  # 2000-01-01
            "%d/%m/%Y",  # 01/01/2000
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        logger.warning("Failed to parse date", date_str=date_str)
        return None

    def _extract_passport_data(self, entities: list[dict]) -> PassportData:
        """Extract passport data from Yandex OCR entities."""
        # Create a mapping of entity names to their text values
        entity_map = {}
        for entity in entities:
            name = entity.get("name", "")
            text = entity.get("text", "")
            if name and text:
                entity_map[name] = text.strip()

        # Map Yandex OCR entity names to our passport fields
        # Field names match Yandex OCR API response structure (see ocr_test.py)
        passport_data = PassportData(
            passport_number=entity_map.get("number"),
            issued_by=entity_map.get("issued_by"),  # Fixed: was "issue_place"
            issue_date=self._parse_date(entity_map.get("issue_date")),
            subdivision_code=entity_map.get("subdivision"),
            surname=entity_map.get("surname"),
            name=entity_map.get("name"),
            middle_name=entity_map.get("middle_name"),
            gender=entity_map.get("gender"),
            birth_date=self._parse_date(entity_map.get("birth_date")),
            birth_place=entity_map.get("birth_place"),
        )

        return passport_data

    async def _make_request(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG", "PDF"]
    ) -> dict:
        """Make OCR API request with retries."""
        # Rate limiting
        await self.rate_limiter.acquire()

        # Prepare request body
        content_b64 = base64.b64encode(image_bytes).decode("utf-8")
        language_codes = settings.ocr_language_codes.split(",") if settings.ocr_language_codes != "*" else ["*"]

        body = {
            "mimeType": mime_type,
            "languageCodes": language_codes,
            "model": settings.ocr_document_model,
            "content": content_b64,
        }

        headers = self._get_headers()

        # Retry logic: 2 retries with backoff
        max_retries = 2
        backoff_delays = [0.5, 1.5]

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(
                    self.endpoint,
                    json=body,
                    headers=headers,
                )

                # Handle rate limiting and server errors
                if response.status_code in [429, 500, 502, 503, 504]:
                    if attempt < max_retries:
                        delay = backoff_delays[attempt]
                        logger.warning(
                            "Retrying OCR request",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            delay=delay
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise httpx.HTTPStatusError(
                            f"OCR request failed after {max_retries} retries",
                            request=response.request,
                            response=response
                        )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt == max_retries:
                    logger.error("OCR request failed", error=str(e), status_code=e.response.status_code)
                    raise
            except Exception as e:
                if attempt == max_retries:
                    logger.error("OCR request failed", error=str(e))
                    raise
                await asyncio.sleep(backoff_delays[attempt])

        raise RuntimeError("Should not reach here")

    async def recognize_passport(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG", "PDF"]
    ) -> OcrResult:
        """Recognize passport from image bytes."""
        try:
            logger.info("Starting OCR recognition", mime_type=mime_type, size_bytes=len(image_bytes))

            response_data = await self._make_request(image_bytes, mime_type)

            # Extract entities from response - correct path according to Yandex OCR API
            # Structure: response["result"]["textAnnotation"]["entities"]
            entities = response_data.get("result", {}).get("textAnnotation", {}).get("entities", [])

            passport_data = self._extract_passport_data(entities)

            logger.info(
                "OCR recognition completed",
                entities_count=len(entities),
                filled_fields=passport_data.count_filled_fields()
            )

            return OcrResult(
                passport_data=passport_data,
                raw_response=response_data,
                success=True
            )

        except Exception as e:
            logger.error("OCR recognition failed", error=str(e))
            return OcrResult(
                passport_data=PassportData(),
                raw_response={"error": str(e)},
                success=False,
                error=str(e)
            )

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()