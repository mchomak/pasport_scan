"""OpenRouter vision LLM provider for passport OCR."""
import base64
import json
import re
from datetime import datetime
from typing import Optional, Literal

import httpx

from ocr.models import OcrResult, PassportData
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a passport data extraction assistant.
Extract all visible data from the passport image and return it as JSON.
Return ONLY a valid JSON object, no markdown, no explanation."""

_USER_PROMPT = """Extract passport data from this image.

Return a JSON object with these exact fields:
{
  "surname": "family name in Latin uppercase, e.g. IVANOV",
  "name": "first name in Latin uppercase, e.g. IVAN",
  "middle_name": "patronymic/middle name in Latin uppercase, or null",
  "passport_number": "series+number without spaces, e.g. fa3009783 or 4619709685",
  "birth_date": "date of birth in YYYY-MM-DD format, or null",
  "expiry_date": "passport expiry/validity date in YYYY-MM-DD format, or null",
  "gender": "male or female, or null",
  "birth_place": "place of birth in English lowercase, or null"
}

Rules:
- All name fields (surname, name, middle_name) must be transliterated to Latin uppercase
- Dates must be in YYYY-MM-DD format
- expiry_date is the date until which the passport is valid (Date of expiry / Дата окончания срока действия / Amal qilish muddati)
- gender must be "male" or "female" (not M/F)
- If a field is not visible or unclear, use null
- Return ONLY the JSON object"""


class OpenRouterProvider:
    """OpenRouter vision LLM provider for passport OCR."""

    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime.date]:
        if not date_str:
            return None
        date_str = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_response(self, content: str) -> PassportData:
        """Parse LLM JSON response into PassportData."""
        content = content.strip()
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                logger.warning("OpenRouter: could not parse JSON", content=content[:200])
                return PassportData()

        def get_str(key: str) -> Optional[str]:
            val = data.get(key)
            if val is None or str(val).strip().lower() in ('null', 'none', ''):
                return None
            return str(val).strip()

        return PassportData(
            surname=get_str("surname"),
            name=get_str("name"),
            middle_name=get_str("middle_name"),
            passport_number=get_str("passport_number"),
            birth_date=self._parse_date(get_str("birth_date")),
            expiry_date=self._parse_date(get_str("expiry_date")),
            gender=get_str("gender"),
            birth_place=get_str("birth_place"),
        )

    async def recognize_passport(
        self,
        image_bytes: bytes,
        mime_type: Literal["JPEG", "PNG", "PDF"] = "JPEG",
    ) -> OcrResult:
        """Recognize passport using OpenRouter vision model."""
        try:
            if not self.api_key:
                return OcrResult(
                    passport_data=PassportData(),
                    raw_response={"error": "OPENROUTER_API_KEY not set"},
                    success=False,
                    error="OPENROUTER_API_KEY not configured",
                )

            media_type = "image/jpeg" if mime_type == "JPEG" else "image/png"
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": _USER_PROMPT},
                        ],
                    },
                ],
                "max_tokens": 512,
                "temperature": 0,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pasport-scan",
            }

            logger.info("OpenRouter: sending request", model=self.model)
            response = await self.client.post(
                self.endpoint, json=payload, headers=headers
            )
            response.raise_for_status()
            resp_json = response.json()

            content = (
                resp_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            logger.info("OpenRouter: response received", content_len=len(content))

            passport_data = self._parse_response(content)
            logger.info(
                "OpenRouter: parsed fields",
                filled=passport_data.count_filled_fields(),
            )

            return OcrResult(
                passport_data=passport_data,
                raw_response={"response": resp_json, "content": content},
                success=True,
            )

        except Exception as e:
            logger.error("OpenRouter recognition failed", error=str(e))
            return OcrResult(
                passport_data=PassportData(),
                raw_response={"error": str(e)},
                success=False,
                error=str(e),
            )