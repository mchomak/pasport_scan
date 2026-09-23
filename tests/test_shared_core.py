import os
from datetime import date
from types import SimpleNamespace
import unittest


os.environ["BOT_TOKEN"] = "fake-telegram-token"
os.environ["ADMIN_IDS"] = "1"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://fake:fake@localhost/fake"
os.environ["OCR_MODULE_PRIORITY"] = ""

from core.messaging import IncomingImage, MessengerSource
from ocr.models import PassportData
from services.passport_processing import PassportProcessingService


class _FakeImageProcessor:
    def normalize_image(self, image_bytes):
        return image_bytes, "JPEG"


class _FakeRecognizer:
    async def recognize(self, image_bytes, mime_type):
        return SimpleNamespace(
            passport_data=PassportData(
                passport_number="4619709685",
                surname="IVANOV",
                name="IVAN",
                gender="male",
                birth_date=date(1990, 1, 2),
                expiry_date=date(2030, 1, 2),
            ),
            modules_used=["fake"],
            raw_response={"response": "private"},
            field_providers={},
            per_module_data={},
        )


class _FakeRepository:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(id="record-id")


class IncomingImageTests(unittest.TestCase):
    def test_external_identifiers_are_normalized_to_strings(self):
        incoming = IncomingImage(
            content=b"image",
            filename=None,
            source=MessengerSource.MAX,
            external_user_id=123,
            external_chat_id=456,
            external_message_id=789,
        )

        self.assertIs(incoming.source, MessengerSource.MAX)
        self.assertEqual(incoming.external_user_id, "123")
        self.assertEqual(incoming.external_chat_id, "456")
        self.assertEqual(incoming.external_message_id, "789")


class PassportProcessingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_image_dual_writes_telegram_metadata(self):
        repository = _FakeRepository()
        service = PassportProcessingService(
            repository_factory=lambda: repository,
            image_processor=_FakeImageProcessor(),
            recognizer=_FakeRecognizer(),
        )

        result = await service.process_image(
            IncomingImage(
                content=b"image",
                filename="passport.jpg",
                source=MessengerSource.TELEGRAM,
                external_user_id="123",
                external_chat_id="456",
                external_message_id="789",
                external_username="alice",
            )
        )

        self.assertEqual(result.record_id, "record-id")
        self.assertEqual(repository.kwargs["source"], "telegram")
        self.assertEqual(repository.kwargs["tg_user_id"], 123)
        self.assertEqual(repository.kwargs["source_message_id"], 789)
        self.assertEqual(repository.kwargs["external_user_id"], "123")
        self.assertEqual(repository.kwargs["external_chat_id"], "456")
        self.assertEqual(repository.kwargs["external_message_id"], "789")

    async def test_process_image_keeps_max_metadata_as_strings(self):
        repository = _FakeRepository()
        service = PassportProcessingService(
            repository_factory=lambda: repository,
            image_processor=_FakeImageProcessor(),
            recognizer=_FakeRecognizer(),
        )

        await service.process_image(
            IncomingImage(
                content=b"image",
                filename="passport.jpg",
                source=MessengerSource.MAX,
                external_user_id=123,
                external_chat_id=456,
                external_message_id=789,
                external_username="alice",
            )
        )

        self.assertIsNone(repository.kwargs["tg_user_id"])
        self.assertIsNone(repository.kwargs["tg_username"])
        self.assertIsNone(repository.kwargs["source_message_id"])
        self.assertEqual(repository.kwargs["source"], "max")
        self.assertEqual(repository.kwargs["external_user_id"], "123")
        self.assertEqual(repository.kwargs["external_chat_id"], "456")
        self.assertEqual(repository.kwargs["external_message_id"], "789")


if __name__ == "__main__":
    unittest.main()
