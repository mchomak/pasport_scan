import os
from datetime import date
import unittest
from unittest.mock import AsyncMock, patch


os.environ["BOT_TOKEN"] = "fake-telegram-token"
os.environ["ADMIN_IDS"] = "1"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://fake:fake@localhost/fake"
os.environ["OCR_MODULE_PRIORITY"] = "openrouter,rupasportread"

from ocr.hybrid import HybridRecognizer, settings
from ocr.models import OcrResult, PassportData


class _FakeOpenRouterProvider:
    def __init__(self, passport_data):
        self.passport_data = passport_data

    async def recognize_passport(self, image_bytes, mime_type):
        return OcrResult(
            passport_data=self.passport_data,
            raw_response={},
            success=True,
        )


class HybridRecognizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_local_recognizer_after_openrouter_fills_every_essential_field(self):
        openrouter_data = PassportData(
            surname="PRIMARY",
            name="FIRST",
            passport_number="AA1234567",
            birth_date=date(1990, 1, 2),
            gender="male",
            expiry_date=date(2030, 1, 2),
        )
        local_data = PassportData(birth_place="TASHKENT")
        recognizer = HybridRecognizer(
            openrouter_provider=_FakeOpenRouterProvider(openrouter_data)
        )
        local_runner = AsyncMock(return_value=local_data)
        recognizer._run_module_rupasportread = local_runner

        with patch.object(settings, "ocr_module_priority", "openrouter,rupasportread"):
            result = await recognizer.recognize(b"image", "JPEG")

        local_runner.assert_awaited_once_with(b"image", "JPEG")
        self.assertEqual(result.passport_data.birth_place, "TASHKENT")

    async def test_local_recognizer_does_not_overwrite_openrouter_name(self):
        openrouter_data = PassportData(surname="IVAN")
        local_data = PassportData(surname="ALEXANDEROVICH")
        recognizer = HybridRecognizer(
            openrouter_provider=_FakeOpenRouterProvider(openrouter_data)
        )
        recognizer._run_module_rupasportread = AsyncMock(return_value=local_data)

        with patch.object(settings, "ocr_module_priority", "openrouter,rupasportread"):
            result = await recognizer.recognize(b"image", "JPEG")

        self.assertEqual(result.passport_data.surname, "IVAN")
        self.assertEqual(result.field_providers["surname"], "openrouter")

    async def test_local_recognizer_fills_missing_openrouter_field(self):
        openrouter_data = PassportData(surname="IVAN")
        local_data = PassportData(birth_place="TASHKENT")
        recognizer = HybridRecognizer(
            openrouter_provider=_FakeOpenRouterProvider(openrouter_data)
        )
        recognizer._run_module_rupasportread = AsyncMock(return_value=local_data)

        with patch.object(settings, "ocr_module_priority", "openrouter,rupasportread"):
            result = await recognizer.recognize(b"image", "JPEG")

        self.assertEqual(result.passport_data.birth_place, "TASHKENT")
        self.assertEqual(result.field_providers["birth_place"], "rupasportread")


if __name__ == "__main__":
    unittest.main()
