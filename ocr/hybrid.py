"""
Hybrid OCR module with configurable module priority.

Module priority is read from settings.ocr_module_priority.
Available modules: openrouter, yandex_ocr, rupasportread.
Higher-priority modules run first; subsequent modules only fill missing fields.
"""
import asyncio
import re
import json
from datetime import datetime
from typing import Optional, List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ocr.models import OcrResult, PassportData
from config import settings
from utils.logger import get_logger, get_file_logger
from utils.passport_formatter import transliterate_to_latin

logger = get_logger(__name__)
debug_log = get_file_logger("hybrid.debug")


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
        self.per_module_data: dict[str, PassportData] = per_module_data or {}


class HybridRecognizer:
    """
    Hybrid passport recognizer with configurable module pipeline.

    Module priority is determined by settings.ocr_module_priority.
    Default: openrouter -> yandex_ocr -> rupasportread
    """

    ESSENTIAL_FIELDS = [
        'surname', 'name', 'passport_number',
        'birth_date', 'gender', 'expiry_date',
    ]

    def __init__(self, yandex_provider=None, openrouter_provider=None):
        self.yandex_provider = yandex_provider
        self.openrouter_provider = openrouter_provider

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
        if not name:
            return None
        original = name
        name = name.upper()

        name = re.sub(r'3', 'CH', name)
        name = re.sub(r'Q$', 'Y', name)
        name = re.sub(r'Q(?=[BCDFGHJKLMNPQRSTVWXYZ])', 'Y', name)
        name = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', name)
        name = re.sub(r'(?<=[A-Z])8(?=[A-Z])', 'B', name)
        name = re.sub(r'(?<=[A-Z])5(?=[A-Z])', 'S', name)
        name = re.sub(r'^\d+', '', name)
        name = re.sub(r'\d+$', '', name)
        name = re.sub(r'[^A-Z\-]', '', name)

        if not name:
            return None

        debug_log.debug("_clean_latin_name: %s -> %s", original, name)
        return name

    @staticmethod
    def _trim_patronymic(middle_name: Optional[str]) -> Optional[str]:
        if not middle_name:
            return None
        m = re.match(r'^(.+(?:VICH|NICH|MICH))(.+)?$', middle_name, re.IGNORECASE)
        if m and m.group(2):
            tail = m.group(2)
            if not tail.upper().startswith('CH'):
                debug_log.debug("_trim_patronymic: %s -> %s (cut '%s')",
                                middle_name, m.group(1), tail)
                return m.group(1).upper()
        m = re.match(r'^(.+(?:OVNA|EVNA))(.+)?$', middle_name, re.IGNORECASE)
        if m and m.group(2):
            debug_log.debug("_trim_patronymic: %s -> %s (cut '%s')",
                            middle_name, m.group(1), m.group(2))
            return m.group(1).upper()
        return middle_name

    @staticmethod
    def _infer_gender_from_name(
        name: Optional[str],
        middle_name: Optional[str],
        surname: Optional[str] = None,
    ) -> Optional[str]:
        """Infer gender from patronymic, surname, or first name endings."""
        if middle_name:
            mn = middle_name.upper()
            if re.search(r'(?:VICH|NICH|MICH|UGLI|OGLI|ZODA)$', mn):
                return 'M'
            if re.search(r'(?:OVNA|EVNA|ICHNA|QIZI|KIZI)$', mn):
                return 'F'
        if surname:
            sn = surname.upper()
            if re.search(r'(?:OVA|EVA|INA|SKAYA|CKAYA)$', sn):
                return 'F'
            if re.search(r'(?:OV|EV|IN|SKIY|CKIY|SKOY|CKOY)$', sn):
                return 'M'
        if name:
            n = name.upper()
            if re.search(r'(?:A|YA|IA|INNA|ALLA)$', n):
                return 'F'
            if re.search(r'(?:IY|EY|IL|AN|IM|AM|ER|IR|AR|UR|EL|AD|ED|AT|AV|EV|OV|ON|IN|OR|UR)$', n):
                return 'M'
        return None

    @staticmethod
    def _name_quality(name: Optional[str]) -> int:
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
    # Module: rupasportread (Tesseract MRZ)
    # ------------------------------------------------------------------
    def _run_rupasportread(self, image_bytes: bytes) -> Optional[dict]:
        try:
            import utils.rupasportread as rupasportread
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
    # Merge helper
    # ------------------------------------------------------------------
    @classmethod
    def _merge(cls, base: PassportData, supplement: PassportData,
               ) -> tuple[PassportData, set]:
        merged = {}
        supplement_wins: set = set()
        for field_name in base.model_fields:
            base_val = getattr(base, field_name)
            supp_val = getattr(supplement, field_name)

            base_filled = base_val is not None and str(base_val).strip()
            supp_filled = supp_val is not None and str(supp_val).strip()

            if field_name in cls.NAME_FIELDS and base_filled and supp_filled:
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
        filled = set()
        for field_name in data.model_fields:
            val = getattr(data, field_name)
            if val is not None and str(val).strip():
                filled.add(field_name)
        return filled

    # ------------------------------------------------------------------
    # Module runners
    # ------------------------------------------------------------------
    async def _run_module_openrouter(
        self, image_bytes: bytes, mime_type: str
    ) -> Optional[PassportData]:
        """Run OpenRouter vision LLM module."""
        if not self.openrouter_provider:
            debug_log.debug("[openrouter] SKIPPED — no provider configured")
            return None
        try:
            result = await self.openrouter_provider.recognize_passport(
                image_bytes, mime_type
            )
            if result.success and result.passport_data:
                return result.passport_data
        except Exception as e:
            logger.error("OpenRouter failed", error=str(e))
        return None

    async def _run_module_yandex(
        self, image_bytes: bytes, mime_type: str
    ) -> Optional[PassportData]:
        """Run Yandex OCR module."""
        if not self.yandex_provider:
            debug_log.debug("[yandex_ocr] SKIPPED — no provider configured")
            return None
        try:
            result = await self.yandex_provider.recognize_passport(
                image_bytes, mime_type
            )
            if result.success and result.passport_data:
                yd = result.passport_data
                # Transliterate Cyrillic names to Latin
                _cyr = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
                if yd.surname and any(c in yd.surname for c in _cyr):
                    yd.surname = transliterate_to_latin(yd.surname)
                if yd.name and any(c in yd.name for c in _cyr):
                    yd.name = transliterate_to_latin(yd.name)
                if yd.middle_name and any(c in yd.middle_name for c in _cyr):
                    yd.middle_name = transliterate_to_latin(yd.middle_name)
                return yd
        except Exception as e:
            logger.error("Yandex OCR failed", error=str(e))
        return None

    async def _run_module_rupasportread(
        self, image_bytes: bytes, mime_type: str
    ) -> Optional[PassportData]:
        """Run rupasportread (Tesseract MRZ) module."""
        try:
            raw = await asyncio.to_thread(self._run_rupasportread, image_bytes)
            if raw:
                return self._rupasportread_to_passport_data(raw)
        except Exception as e:
            logger.error("rupasportread failed", error=str(e))
        return None

    # Module dispatcher
    _MODULE_MAP = {
        'openrouter': '_run_module_openrouter',
        'yandex_ocr': '_run_module_yandex',
        'rupasportread': '_run_module_rupasportread',
    }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _passport_data_to_debug_dict(data: PassportData) -> dict:
        return {k: str(v) if v is not None else None
                for k, v in data.model_dump().items()}

    async def recognize(
        self,
        image_bytes: bytes,
        mime_type: str = "JPEG",
    ) -> HybridResult:
        """Run the hybrid recognition pipeline with configurable priority."""
        modules_used: List[str] = []
        current_data = PassportData()
        raw_responses: dict = {}
        field_providers: dict = {}
        per_module_data: dict[str, PassportData] = {}

        priority = settings.get_module_priority()
        total = len(priority)

        debug_log.debug("=" * 60)
        debug_log.debug("HYBRID PIPELINE START  image_size=%d  mime=%s  priority=%s",
                        len(image_bytes), mime_type, priority)

        for idx, module_key in enumerate(priority):
            method_name = self._MODULE_MAP.get(module_key)
            if not method_name:
                logger.warning("Unknown OCR module: %s, skipping", module_key)
                continue

            # Skip if all essential fields already filled (except first module)
            if idx > 0 and self._count_essential(current_data) >= len(self.ESSENTIAL_FIELDS):
                debug_log.debug("[%s] SKIPPED — all essential fields filled", module_key)
                continue

            logger.info("Hybrid: [%d/%d] %s...", idx + 1, total, module_key)

            method = getattr(self, method_name)
            mod_data = await method(image_bytes, mime_type)

            if mod_data:
                mod_data = self._clean_passport_data(mod_data)
                per_module_data[module_key] = mod_data

                debug_log.debug("[%s] parsed=%s", module_key,
                               json.dumps(self._passport_data_to_debug_dict(mod_data),
                                          ensure_ascii=False))

                filled_before = self._get_filled_fields(current_data)
                current_data, supp_wins = self._merge(current_data, mod_data)
                filled_after = self._get_filled_fields(current_data)
                new_fields = filled_after - filled_before
                for f in new_fields | supp_wins:
                    field_providers[f] = module_key
                modules_used.append(module_key)

                logger.info(
                    "%s done", module_key,
                    filled=current_data.count_filled_fields(),
                    essential=self._count_essential(current_data),
                    new_fields=list(new_fields),
                )
            else:
                debug_log.debug("[%s] returned None", module_key)
                logger.info("%s: no result", module_key)

        # ---- Post-processing: infer gender from name/patronymic ----
        if not current_data.gender or not current_data.gender.strip():
            inferred = self._infer_gender_from_name(
                current_data.name, current_data.middle_name,
                current_data.surname)
            if inferred:
                debug_log.debug("Gender inferred from name: %s", inferred)
                current_data = current_data.model_copy(
                    update={"gender": inferred})
                field_providers["gender"] = "inferred"

        if not modules_used:
            modules_used.append("none")

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