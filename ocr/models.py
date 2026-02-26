"""OCR data models."""
from datetime import date
from typing import Optional
from pydantic import BaseModel


class PassportData(BaseModel):
    """Extracted passport data."""

    passport_number: Optional[str] = None
    surname: Optional[str] = None
    name: Optional[str] = None
    middle_name: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[date] = None
    birth_place: Optional[str] = None
    expiry_date: Optional[date] = None

    def count_filled_fields(self) -> int:
        """Count non-empty fields."""
        count = 0
        for field_name, field_value in self.model_dump().items():
            if field_value is not None and str(field_value).strip():
                count += 1
        return count


class OcrResult(BaseModel):
    """OCR recognition result."""

    passport_data: PassportData
    raw_response: dict
    success: bool = True
    error: Optional[str] = None