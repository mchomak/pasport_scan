"""
Hybrid OCR module combining three recognition engines with priority chain:

  1. rupasportread  (Tesseract MRZ) - fast, free, Latin names directly
  2. EasyOCR        (enhanced)      - deeper analysis, MRZ + Russian text
  3. Yandex OCR     (cloud API)     - most accurate, paid

Each subsequent engine only fills in fields that are still missing.
"""
import asyncio
import os
import re
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
    # Latin-name cleanup (common MRZ OCR artifacts)
    # ------------------------------------------------------------------
    NAME_FIELDS = ('surname', 'name', 'middle_name')

    @staticmethod
    def _clean_latin_name(name: Optional[str]) -> Optional[str]:
        """Fix common OCR artifacts in Latin names from MRZ.

        Typical errors:
          3  → CH   (Ч in MRZ → ICAO "CH", Tesseract reads as "3")
          9  → strip (soft-sign artifact, e.g. RAMIL9 → RAMIL)
          Q  → Y    (common tail confusion, DMITRIQ → DMITRIY)
          0  → O    (zero → letter O inside a word)
          8  → B    (eight → B inside a word)
          5  → S    (five → S inside a word)
        """
        if not name:
            return None
        original = name

        # Upper-case for uniform processing
        name = name.upper()

        # 3 → CH  (patronymic -VICH: ANDREEVI3 → ANDREEVICH)
        name = re.sub(r'3', 'CH', name)

        # Q → Y at the end  (DMITRIQ → DMITRIY)
        name = re.sub(r'Q$', 'Y', name)
        # Q → Y also before consonant cluster (less common but safe)
        name = re.sub(r'Q(?=[BCDFGHJKLMNPQRSTVWXYZ])', 'Y', name)

        # Digit substitutions inside a word (surrounded by letters)
        name = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', name)
        name = re.sub(r'(?<=[A-Z])8(?=[A-Z])', 'B', name)
        name = re.sub(r'(?<=[A-Z])5(?=[A-Z])', 'S', name)

        # Strip remaining leading/trailing digits  (RAMIL9 → RAMIL)
        name = re.sub(r'^\d+', '', name)
        name = re.sub(r'\d+$', '', name)

        # Remove any remaining non-alpha chars except hyphen
        name = re.sub(r'[^A-Z\-]', '', name)

        if not name:
            return None

        debug_log.debug("_clean_latin_name: %s -> %s", original, name)
        return name

    @staticmethod
    def _trim_patronymic(middle_name: Optional[str]) -> Optional[str]:
        """Trim trailing garbage after -CH ending in patronymics.

        Russian patronymics end with -VICH / -OVICH / -EVICH (male)
        or -OVNA / -EVNA (female).
        If extra chars appear after the valid ending, strip them.
        E.g. KHOSHIMJONOVICHSS → KHOSHIMJONOVICH
        """
        if not middle_name:
            return None
        # Male patronymic: cut everything after last ...VICH / ...NICH
        m = re.match(r'^(.+(?:VICH|NICH|MICH))(.+)?$', middle_name, re.IGNORECASE)
        if m and m.group(2):
            tail = m.group(2)
            # Keep tail only if it starts with another CH (double-Ч is impossible)
            if not tail.upper().startswith('CH'):
                debug_log.debug("_trim_patronymic: %s -> %s (cut '%s')",
                                middle_name, m.group(1), tail)
                return m.group(1).upper()
        # Female patronymic: ...OVNA / ...EVNA
        m = re.match(r'^(.+(?:OVNA|EVNA))(.+)?$', middle_name, re.IGNORECASE)
        if m and m.group(2):
            debug_log.debug("_trim_patronymic: %s -> %s (cut '%s')",
                            middle_name, m.group(1), m.group(2))
            return m.group(1).upper()
        return middle_name

    @staticmethod
    def _infer_gender_from_name(name: Optional[str], middle_name: Optional[str]) -> Optional[str]:
        """Infer gender from first name / patronymic endings.

        Male indicators:  name ending in consonant/IY/EY/OV,
                          patronymic ending in -VICH / -OVICH
        Female indicators: name ending in -A/-YA/-IA,
                           patronymic ending in -OVNA / -EVNA
        """
        # Patronymic is the most reliable signal
        if middle_name:
            mn = middle_name.upper()
            if re.search(r'(?:VICH|NICH|MICH|UGLI|OGLI|ZODA)$', mn):
                return 'M'
            if re.search(r'(?:OVNA|EVNA|ICHNA|QIZI|KIZI)$', mn):
                return 'F'
        # Fallback to first name ending
        if name:
            n = name.upper()
            if re.search(r'(?:A|YA|IA|INNA|ALLA)$', n):
                return 'F'
            # Most male names end in consonant or IY/EY
            if re.search(r'(?:IY|EY|IL|AN|IM|AM|ER|IR|AR|UR|EL|AD|ED|AT|AV|EV|OV|ON|IN|OR|UR)$', n):
                return 'M'
        return None

    @staticmethod
    def _name_quality(name: Optional[str]) -> int:
        """Score a Latin name: higher is better.

        +10  base (non-empty)
        +len  longer names are usually more complete
        -5   per digit remaining
        -3   per non-alpha non-hyphen char
        """
        if not name or not name.strip():
            return 0
        score = 10 + len(name)
        for ch in name:
            if ch.isdigit():
                score -= 5
            elif not ch.isalpha() and ch != '-':
                score -= 3
        return max(score, 1)

    @classmethod
    def _clean_passport_data(cls, data: PassportData) -> PassportData:
        """Apply _clean_latin_name + _trim_patronymic to name fields."""
        updates = {}
        for f in cls.NAME_FIELDS:
            raw = getattr(data, f, None)
            if raw:
                cleaned = cls._clean_latin_name(raw)
                if f == 'middle_name' and cleaned:
                    cleaned = cls._trim_patronymic(cleaned)
                if cleaned != raw:
                    updates[f] = cleaned
        if updates:
            return data.model_copy(update=updates)
        return data

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
    @classmethod
    def _merge(cls, base: PassportData, supplement: PassportData,
               ) -> tuple[PassportData, set]:
        """Merge two PassportData.

        For regular fields: base wins if non-empty.
        For name fields (surname, name, middle_name): pick the higher-quality
        value when both are present (longer, no digit artifacts).

        Returns (merged PassportData, set of field names won by supplement).
        """
        merged = {}
        supplement_wins: set = set()
        for field_name in base.model_fields:
            base_val = getattr(base, field_name)
            supp_val = getattr(supplement, field_name)

            base_filled = base_val is not None and str(base_val).strip()
            supp_filled = supp_val is not None and str(supp_val).strip()

            if field_name in cls.NAME_FIELDS and base_filled and supp_filled:
                # Both modules found this name → pick better one
                bq = cls._name_quality(str(base_val))
                sq = cls._name_quality(str(supp_val))
                if sq > bq:
                    debug_log.debug(
                        "merge: prefer supplement for %s: %s (q=%d) > %s (q=%d)",
                        field_name, supp_val, sq, base_val, bq)
                    merged[field_name] = supp_val
                    supplement_wins.add(field_name)
                else:
                    merged[field_name] = base_val
            elif base_filled:
                merged[field_name] = base_val
            else:
                merged[field_name] = supp_val
        return PassportData(**merged), supplement_wins

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
            rpr_data = self._clean_passport_data(rpr_data)
            per_module_data["rupasportread"] = rpr_data

            debug_log.debug("[rupasportread] raw_result=%s",
                           json.dumps(rpr_result, ensure_ascii=False, default=str))
            debug_log.debug("[rupasportread] parsed=%s",
                           json.dumps(self._passport_data_to_debug_dict(rpr_data),
                                      ensure_ascii=False))

            filled_before = self._get_filled_fields(current_data)
            current_data, _ = self._merge(current_data, rpr_data)
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
                easyocr_data = self._clean_passport_data(easyocr_data)
                per_module_data["easyocr"] = easyocr_data

                debug_log.debug("[easyocr] parsed=%s",
                               json.dumps(self._passport_data_to_debug_dict(easyocr_data),
                                          ensure_ascii=False))

                filled_before = self._get_filled_fields(current_data)
                current_data, supp_wins = self._merge(current_data, easyocr_data)
                filled_after = self._get_filled_fields(current_data)
                new_fields = filled_after - filled_before
                for f in new_fields | supp_wins:
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

                    yd = self._clean_passport_data(yd)
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
                    current_data, supp_wins = self._merge(current_data, yd)
                    filled_after = self._get_filled_fields(current_data)
                    new_fields = filled_after - filled_before
                    for f in new_fields | supp_wins:
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

        # ---- Post-processing: Yandex priority for passport_number ----
        if "yandex_ocr" in per_module_data:
            yd = per_module_data["yandex_ocr"]
            if yd.passport_number and yd.passport_number.strip():
                old_pn = current_data.passport_number
                if old_pn != yd.passport_number:
                    debug_log.debug(
                        "Yandex passport_number override: %s -> %s",
                        old_pn, yd.passport_number)
                    current_data = current_data.model_copy(
                        update={"passport_number": yd.passport_number})
                    field_providers["passport_number"] = "yandex_ocr"

        # ---- Post-processing: infer gender from name/patronymic ----
        if not current_data.gender or not current_data.gender.strip():
            inferred = self._infer_gender_from_name(
                current_data.name, current_data.middle_name)
            if inferred:
                debug_log.debug("Gender inferred from name: %s", inferred)
                current_data = current_data.model_copy(
                    update={"gender": inferred})
                field_providers["gender"] = "inferred"

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