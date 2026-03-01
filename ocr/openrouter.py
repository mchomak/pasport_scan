"""OpenRouter vision LLM provider for passport OCR."""
import base64
import json
import re
from datetime import datetime, date
from typing import Any, Optional, Literal

import httpx

from ocr.models import OcrResult, PassportData
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a passport data extraction assistant.
Extract all visible data from the passport image and return it as JSON.
Return ONLY a valid JSON object, no markdown, no explanation."""

_USER_PROMPT = """Extract passport data from this image.

Return a JSON object with these required fields:
{
  "surname": "family name in Latin uppercase, e.g. IVANOV",
  "name": "first name in Latin uppercase, e.g. IVAN",
  "middle_name": "patronymic/middle name in Latin uppercase, or null",
  "passport_number": "series+number without spaces, e.g. FA3009783 or 4619709685",
  "birth_date": "date of birth in YYYY-MM-DD format, or null",
  "expiry_date": "passport expiry/validity date in YYYY-MM-DD format, or null",
  "gender": "male or female, or null",
  "birth_place": "place of birth in English lowercase, or null"
}

You may also include these optional debug fields (recommended):
{
  "passport_number_source": "visual | mrz | mixed | unknown",
  "mrz_document_number_raw": "document number read from MRZ including MRZ check digit if visible, no spaces, or null",
  "mrz_document_number_check_digit": "single MRZ check digit for document number, or null"
}

Rules:
- All name fields (surname, name, middle_name) must be transliterated to Latin uppercase.
- Dates must be in YYYY-MM-DD format.
- expiry_date is the date until which the passport is valid (Date of expiry / Дата окончания срока действия / Amal qilish muddati).
- gender must be "male" or "female" (not M/F).
- If a field is not visible or unclear, use null.
- If top-right passport number is partially visible, you may use MRZ as fallback.
- IMPORTANT (MRZ): The document number and the MRZ check digit are separate values.
- If passport_number is taken from MRZ, DO NOT include the MRZ check digit (control character) immediately to the right of the document number.
- Never append MRZ check digits to passport_number, birth_date, or expiry_date.
- Return ONLY the JSON object.
"""


class OpenRouterProvider:
    """OpenRouter vision LLM provider for passport OCR."""

    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=60.0)


    async def close(self):
        await self.client.aclose()


    @staticmethod
    def _apply_uz_passport_number_rule(passport_number: Optional[str]) -> Optional[str]:
        """
        Apply Uzbekistan passport number heuristic:
        expected format is 2 letters + 7 digits.
        If model returns 2 letters + 8 digits (extra MRZ check digit), trim last digit.
        """
        if not passport_number:
            return None

        value = passport_number.strip().upper()
        value = re.sub(r"\s+", "", value)

        # Match two Latin letters + digits
        m = re.fullmatch(r"([A-Z]{2})(\d+)", value)
        if not m:
            return value

        series, digits = m.groups()

        # Expected: 9 digits
        if len(digits) == 7:
            return f"{series}{digits}"

        # Common MRZ issue: extra check digit appended -> 8 digits
        if len(digits) == 8:
            fixed = f"{series}{digits[:-1]}"
            logger.info(
                "OpenRouter: applied UZ passport heuristic (trim extra digit)",
                before=value,
                after=fixed,
            )
            return fixed

        # Any other length: keep as is, but you may log for QA
        logger.warning(
            "OpenRouter: unexpected UZ passport number length",
            value=value,
            digits_len=len(digits),
        )
        return value


    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        if not date_str:
            return None
        date_str = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def _clean_response_text(self, content: str) -> str:
        """Remove markdown fences and surrounding noise from model output."""
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return content.strip()

    def _extract_json_dict(self, content: str) -> dict[str, Any]:
        """Extract JSON object from model response text."""
        cleaned = self._clean_response_text(content)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                raise
            data = json.loads(m.group(0))

        if not isinstance(data, dict):
            raise ValueError("Model response JSON is not an object")

        return data

    @staticmethod
    def _get_str(data: dict[str, Any], key: str) -> Optional[str]:
        """Return normalized optional string field from JSON."""
        val = data.get(key)
        if val is None:
            return None

        text = str(val).strip()
        if text.lower() in {"null", "none", ""}:
            return None
        return text

    @staticmethod
    def _normalize_alnum(value: Optional[str], keep_angle_brackets: bool = False) -> Optional[str]:
        """Normalize MRZ/number-like strings by removing whitespace."""
        if not value:
            return None

        out = re.sub(r"\s+", "", value)
        if not keep_angle_brackets:
            out = out.replace("<", "")
        return out or None

    def _normalize_passport_number(self, data: dict[str, Any]) -> Optional[str]:
        """
        Normalize passport number with MRZ-aware postprocessing.

        Strategy:
        - Prefer model's passport_number as base value.
        - If source indicates MRZ and a separate MRZ check digit is present, strip it when appended.
        - If mrz_document_number_raw is present and includes check digit, derive a clean MRZ document number.
        - Apply changes only for MRZ/mixed scenarios to avoid damaging valid visual reads.
        """
        passport_number = self._normalize_alnum(self._get_str(data, "passport_number"))
        if not passport_number:
            return None

        source = (self._get_str(data, "passport_number_source") or "unknown").strip().lower()
        mrz_raw = self._normalize_alnum(self._get_str(data, "mrz_document_number_raw"))
        mrz_check_digit = self._get_str(data, "mrz_document_number_check_digit")

        # Normalize check digit to a single character if provided.
        if mrz_check_digit is not None:
            mrz_check_digit = mrz_check_digit.strip()
            if mrz_check_digit == "":
                mrz_check_digit = None
            elif len(mrz_check_digit) > 1:
                mrz_check_digit = mrz_check_digit[-1]

        if source not in {"mrz", "mixed"}:
            return self._apply_uz_passport_number_rule(passport_number)

        original = passport_number
        normalized = passport_number

        # Case 1: The model explicitly provided MRZ raw doc number + check digit.
        # Example:
        #   mrz_document_number_raw = "FA30097835"
        #   mrz_document_number_check_digit = "5"
        #   passport_number should be "FA3009783"
        if mrz_raw and mrz_check_digit and mrz_raw.endswith(mrz_check_digit) and len(mrz_raw) > 1:
            mrz_number_wo_check = mrz_raw[:-1]

            # If the returned passport_number equals raw MRZ (with check digit), fix it.
            if normalized == mrz_raw:
                normalized = mrz_number_wo_check
            # If the returned passport_number ends with the same check digit and starts with MRZ core, fix it.
            elif normalized.endswith(mrz_check_digit) and normalized[:-1] == mrz_number_wo_check:
                normalized = mrz_number_wo_check
            # If source is MRZ and the core looks more reliable than the returned value, prefer it.
            elif source == "mrz" and mrz_number_wo_check:
                normalized = mrz_number_wo_check

        # Case 2: No explicit raw MRZ, but source says MRZ and check digit is available separately.
        # If model appended the check digit to passport_number, strip it.
        elif mrz_check_digit and len(normalized) > 1 and normalized.endswith(mrz_check_digit):
            normalized = normalized[:-1]

        # Keep safe fallback behavior: do NOT blindly strip the last character
        # when there is no explicit signal (check digit / MRZ raw fields).

        if normalized != original:
            logger.info(
                "OpenRouter: normalized passport_number using MRZ rules",
                source=source,
                before=original,
                after=normalized,
                mrz_raw=mrz_raw,
                mrz_check_digit=mrz_check_digit,
            )

        normalized = self._apply_uz_passport_number_rule(normalized)
        return normalized
    

    def _parse_response(self, content: str) -> PassportData:
        """Parse LLM JSON response into PassportData."""
        try:
            data = self._extract_json_dict(content)
        except Exception:
            logger.warning("OpenRouter: could not parse JSON", content=content[:300])
            return PassportData()

        try:
            return PassportData(
                surname=self._get_str(data, "surname"),
                name=self._get_str(data, "name"),
                middle_name=self._get_str(data, "middle_name"),
                passport_number=self._normalize_passport_number(data),
                birth_date=self._parse_date(self._get_str(data, "birth_date")),
                expiry_date=self._parse_date(self._get_str(data, "expiry_date")),
                gender=self._get_str(data, "gender"),
                birth_place=self._get_str(data, "birth_place"),
            )
        except Exception:
            logger.exception("OpenRouter: failed to build PassportData")
            return PassportData()

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

            # NOTE: If your pipeline sends PDF, it should ideally be rasterized to image before this call.
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
                "max_tokens": 700,  # a bit more room for optional debug fields
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