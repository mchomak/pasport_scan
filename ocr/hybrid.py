"""
Hybrid OCR module combining three recognition engines with priority chain:

  1. rupasportread  (Tesseract MRZ) - fast, free, Latin names directly
  2. EasyOCR        (enhanced)      - deeper analysis, MRZ + Russian text
  3. Yandex OCR     (cloud API)     - most accurate, paid

Each subsequent engine only fills in fields that are still missing.
"""
import asyncio
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional, List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

from ocr.models import OcrResult, PassportData
from utils.logger import get_logger, get_file_logger
from utils.passport_formatter import transliterate_to_latin

logger = get_logger(__name__)
debug_log = get_file_logger("hybrid.debug")

# Lazy-loaded EasyOCR reader (expensive to init, keep in memory)
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        logger.info("Initializing EasyOCR reader (first use)...")
        _easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=False)
        logger.info("EasyOCR reader ready")
    return _easyocr_reader


class HybridResult:
    """Result from hybrid OCR pipeline."""

    def __init__(
        self,
        passport_data: PassportData,
        modules_used: List[str],
        raw_response: dict,
        field_providers: Optional[dict] = None,
        per_module_data: Optional[dict] = None,
    ):
        self.passport_data = passport_data
        self.modules_used = modules_used
        self.raw_response = raw_response
        self.field_providers = field_providers or {}
        # {module_name: PassportData} — full data each module found independently
        self.per_module_data: dict[str, PassportData] = per_module_data or {}


class HybridRecognizer:
    """
    Hybrid passport recognizer with 3-tier priority:
      1) rupasportread  - Tesseract MRZ reader
      2) EasyOCR        - enhanced OCR with document detection
      3) Yandex OCR     - cloud API (last resort)
    """

    ESSENTIAL_FIELDS = [
        'surname', 'name', 'passport_number',
        'birth_date', 'gender', 'issue_date',
    ]

    def __init__(self, yandex_provider=None):
        self.yandex_provider = yandex_provider

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _is_valid_name(self, name: Optional[str]) -> bool:
        if not name:
            return False
        name = name.strip()
        return len(name) >= 2 and name.replace('-', '').isalpha()

    def _is_valid_passport_number(self, number: Optional[str]) -> bool:
        if not number:
            return False
        digits = number.replace(" ", "")
        return len(digits) >= 9 and digits.isdigit()

    def _parse_date_dmy(self, date_str: Optional[str]):
        if not date_str:
            return None
        for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _count_essential(self, data: PassportData) -> int:
        count = 0
        for f in self.ESSENTIAL_FIELDS:
            val = getattr(data, f, None)
            if val is not None and str(val).strip():
                count += 1
        return count

    # ------------------------------------------------------------------
    # Priority 1: rupasportread
    # ------------------------------------------------------------------
    def _run_rupasportread(self, image_bytes: bytes) -> Optional[dict]:
        try:
            import rupasportread
            result = rupasportread.recognize_from_bytes(image_bytes)
            return result
        except Exception as e:
            logger.warning("rupasportread failed", error=str(e))
            return None

    def _rupasportread_to_passport_data(self, data: dict) -> PassportData:
        passport_number = None
        series = data.get('Series')
        number = data.get('Number')
        if series and number:
            passport_number = f"{series} {number}"
        birth_date = self._parse_date_dmy(data.get('Date'))
        return PassportData(
            surname=data.get('Surname'),
            name=data.get('Name'),
            middle_name=data.get('Mid'),
            birth_date=birth_date,
            passport_number=passport_number,
        )

    # ------------------------------------------------------------------
    # Priority 2: EasyOCR enhanced
    # ------------------------------------------------------------------
    def _run_easyocr(self, image_bytes: bytes) -> Optional[PassportData]:
        try:
            from test_easyocr_enhanced import (
                detect_and_crop_document,
                preprocess_full,
                preprocess_edge_strips,
                preprocess_mrz,
                extract_series_number,
                extract_fio_from_mrz,
                extract_meta_from_mrz,
                extract_from_russian_text,
                extract_fio_russian,
            )
            import cv2
            import numpy as np

            temp_dir = tempfile.mkdtemp(prefix='hybrid_ocr_')
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None

                temp_image = os.path.join(temp_dir, 'input.jpg')
                cv2.imwrite(temp_image, img)
                preproc_dir = os.path.join(temp_dir, 'preprocessed')

                # Document detection & crop
                cropped_path = detect_and_crop_document(temp_image, preproc_dir)
                if cropped_path is None:
                    return None

                # Preprocessing
                full_path = preprocess_full(cropped_path, preproc_dir)
                strip_paths = preprocess_edge_strips(cropped_path, preproc_dir)
                mrz_paths = preprocess_mrz(cropped_path, preproc_dir)

                # OCR
                reader = _get_easyocr_reader()
                lines_full = reader.readtext(full_path)

                strip_ocr_variants = []
                for sp in strip_paths:
                    lines = reader.readtext(sp)
                    strip_ocr_variants.append(lines)

                lines_mrz = []
                for mp in mrz_paths:
                    lines_mrz.extend(reader.readtext(mp))

                all_mrz_lines = lines_mrz + lines_full

                # Extract data
                series, number = extract_series_number(
                    strip_ocr_variants, lines_full
                )
                surname_lat, name_lat, middle_lat = extract_fio_from_mrz(
                    all_mrz_lines
                )
                mrz_meta = extract_meta_from_mrz(all_mrz_lines)
                birth_date_mrz = mrz_meta.get('birth_date')
                gender_mrz = mrz_meta.get('gender')

                birth_date_ru, issue_date_ru, gender_ru = \
                    extract_from_russian_text(lines_full)

                (surname_lat_fb, name_lat_fb, middle_lat_fb,
                 _surname_ru, _name_ru, _middle_ru) = extract_fio_russian(
                    lines_full
                )

                # Merge EasyOCR-internal sources (MRZ > transliteration)
                fio_surname = surname_lat or surname_lat_fb
                fio_name = name_lat or name_lat_fb
                fio_middle = middle_lat or middle_lat_fb
                birth_date = birth_date_ru or birth_date_mrz
                issue_date = issue_date_ru
                gender = gender_ru or gender_mrz

                passport_number = None
                if series and number:
                    passport_number = f"{series} {number}"

                return PassportData(
                    surname=fio_surname,
                    name=fio_name,
                    middle_name=fio_middle,
                    birth_date=birth_date,
                    issue_date=issue_date,
                    gender=gender,
                    passport_number=passport_number,
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.warning("EasyOCR failed", error=str(e))
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Merge helper
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(base: PassportData, supplement: PassportData) -> PassportData:
        """Merge two PassportData; `base` values take priority."""
        merged = {}
        for field_name in base.model_fields:
            base_val = getattr(base, field_name)
            supp_val = getattr(supplement, field_name)
            if base_val is not None and str(base_val).strip():
                merged[field_name] = base_val
            else:
                merged[field_name] = supp_val
        return PassportData(**merged)

    @staticmethod
    def _get_filled_fields(data: PassportData) -> set:
        """Return set of field names that have non-empty values."""
        filled = set()
        for field_name in data.model_fields:
            val = getattr(data, field_name)
            if val is not None and str(val).strip():
                filled.add(field_name)
        return filled

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _passport_data_to_debug_dict(data: PassportData) -> dict:
        """Serialize PassportData for debug logging (dates -> str)."""
        return {k: str(v) if v is not None else None
                for k, v in data.model_dump().items()}

    async def recognize(
        self,
        image_bytes: bytes,
        mime_type: str = "JPEG",
    ) -> HybridResult:
        """
        Run the hybrid recognition pipeline.

        Returns HybridResult with passport_data, modules_used list,
        raw_response dict, field_providers mapping,
        and per_module_data with full data each module found.
        """
        modules_used: List[str] = []
        current_data = PassportData()
        raw_responses: dict = {}
        field_providers: dict = {}
        per_module_data: dict[str, PassportData] = {}

        debug_log.debug("=" * 60)
        debug_log.debug("HYBRID PIPELINE START  image_size=%d  mime=%s",
                        len(image_bytes), mime_type)

        # ---- Priority 1: rupasportread (Tesseract MRZ) ----
        logger.info("Hybrid: [1/3] rupasportread...")
        rpr_result = await asyncio.to_thread(
            self._run_rupasportread, image_bytes
        )

        if rpr_result:
            rpr_data = self._rupasportread_to_passport_data(rpr_result)
            per_module_data["rupasportread"] = rpr_data

            debug_log.debug("[rupasportread] raw_result=%s",
                           json.dumps(rpr_result, ensure_ascii=False, default=str))
            debug_log.debug("[rupasportread] parsed=%s",
                           json.dumps(self._passport_data_to_debug_dict(rpr_data),
                                      ensure_ascii=False))

            filled_before = self._get_filled_fields(current_data)
            current_data = self._merge(current_data, rpr_data)
            filled_after = self._get_filled_fields(current_data)
            new_fields = filled_after - filled_before
            for f in new_fields:
                field_providers[f] = "rupasportread"
            modules_used.append("rupasportread")
            raw_responses['rupasportread'] = rpr_result
            logger.info(
                "rupasportread done",
                filled=current_data.count_filled_fields(),
                essential=self._count_essential(current_data),
                new_fields=list(new_fields),
            )
        else:
            debug_log.debug("[rupasportread] returned None")
            logger.info("rupasportread: no result")

        # ---- Priority 2: EasyOCR (if essential fields still missing) ----
        if self._count_essential(current_data) < len(self.ESSENTIAL_FIELDS):
            logger.info("Hybrid: [2/3] EasyOCR...")
            easyocr_data = await asyncio.to_thread(
                self._run_easyocr, image_bytes
            )

            if easyocr_data:
                per_module_data["easyocr"] = easyocr_data

                debug_log.debug("[easyocr] parsed=%s",
                               json.dumps(self._passport_data_to_debug_dict(easyocr_data),
                                          ensure_ascii=False))

                filled_before = self._get_filled_fields(current_data)
                current_data = self._merge(current_data, easyocr_data)
                filled_after = self._get_filled_fields(current_data)
                new_fields = filled_after - filled_before
                for f in new_fields:
                    field_providers[f] = "easyocr"
                modules_used.append("easyocr")
                raw_responses['easyocr'] = easyocr_data.model_dump(
                    mode='json'
                )
                logger.info(
                    "EasyOCR done",
                    filled=current_data.count_filled_fields(),
                    essential=self._count_essential(current_data),
                    new_fields=list(new_fields),
                )
            else:
                debug_log.debug("[easyocr] returned None")
                logger.info("EasyOCR: no result")
        else:
            debug_log.debug("[easyocr] SKIPPED — all essential fields filled")

        # ---- Priority 3: Yandex OCR (last resort) ----
        if (
            self._count_essential(current_data) < len(self.ESSENTIAL_FIELDS)
            and self.yandex_provider
        ):
            logger.info("Hybrid: [3/3] Yandex OCR...")
            try:
                yandex_result = await self.yandex_provider.recognize_passport(
                    image_bytes, mime_type
                )
                if yandex_result.success and yandex_result.passport_data:
                    yd = yandex_result.passport_data
                    # Transliterate Cyrillic names to Latin for consistency
                    if yd.surname and any(
                        c in yd.surname for c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
                    ):
                        yd.surname = transliterate_to_latin(yd.surname)
                    if yd.name and any(
                        c in yd.name for c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
                    ):
                        yd.name = transliterate_to_latin(yd.name)
                    if yd.middle_name and any(
                        c in yd.middle_name for c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
                    ):
                        yd.middle_name = transliterate_to_latin(yd.middle_name)

                    per_module_data["yandex_ocr"] = yd

                    debug_log.debug("[yandex_ocr] parsed=%s",
                                   json.dumps(self._passport_data_to_debug_dict(yd),
                                              ensure_ascii=False))
                    debug_log.debug("[yandex_ocr] raw_entities=%s",
                                   json.dumps(
                                       yandex_result.raw_response.get("result", {})
                                       .get("textAnnotation", {})
                                       .get("entities", []),
                                       ensure_ascii=False, default=str))

                    filled_before = self._get_filled_fields(current_data)
                    current_data = self._merge(current_data, yd)
                    filled_after = self._get_filled_fields(current_data)
                    new_fields = filled_after - filled_before
                    for f in new_fields:
                        field_providers[f] = "yandex_ocr"
                    modules_used.append("yandex_ocr")
                    raw_responses['yandex_ocr'] = yandex_result.raw_response
                    logger.info(
                        "Yandex OCR done",
                        filled=current_data.count_filled_fields(),
                        new_fields=list(new_fields),
                    )
            except Exception as e:
                debug_log.debug("[yandex_ocr] EXCEPTION: %s", e, exc_info=True)
                logger.error("Yandex OCR failed", error=str(e))
        else:
            if self._count_essential(current_data) >= len(self.ESSENTIAL_FIELDS):
                debug_log.debug("[yandex_ocr] SKIPPED — all essential fields filled")
            elif not self.yandex_provider:
                debug_log.debug("[yandex_ocr] SKIPPED — no provider configured")

        if not modules_used:
            modules_used.append("none")

        # Final debug summary
        debug_log.debug("MERGED RESULT: %s",
                        json.dumps(self._passport_data_to_debug_dict(current_data),
                                   ensure_ascii=False))
        debug_log.debug("FIELD PROVIDERS: %s", field_providers)
        debug_log.debug("MODULES USED: %s", modules_used)
        debug_log.debug("=" * 60)

        return HybridResult(
            passport_data=current_data,
            modules_used=modules_used,
            raw_response=raw_responses,
            field_providers=field_providers,
            per_module_data=per_module_data,
        )