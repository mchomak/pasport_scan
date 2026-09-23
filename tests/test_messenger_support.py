import os
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace


for _name, _value in {
    "BOT_TOKEN": "fake-telegram-token",
    "MAX_BOT_TOKEN": "fake-max-token",
    "ADMIN_IDS": "1",
    "DATABASE_URL": "postgresql+asyncpg://fake:fake@localhost/fake",
    "OCR_MODULE_PRIORITY": "",
}.items():
    os.environ.setdefault(_name, _value)


from bot.max_adapter import (  # noqa: E402
    UNSUPPORTED_ATTACHMENT_MESSAGE,
    build_incoming_image,
    extract_image_attachments,
    extract_unsupported_attachments,
    handle_message,
)
from core.messaging import IncomingImage, MessengerSource  # noqa: E402
from ocr.models import PassportData  # noqa: E402
from services import passport_processing  # noqa: E402
from services.passport_processing import PassportProcessingService  # noqa: E402


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
        self.records = []

    async def create(self, **kwargs):
        self.records.append(kwargs)
        return SimpleNamespace(id=f"record-{len(self.records)}")


class _FakeResponse:
    def __init__(self, content):
        self.status = 200
        self.headers = {"content-length": str(len(content))}
        self._content = content

    async def read(self):
        return self._content


class _FakeResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []

    def get(self, url, *, timeout):
        self.requests.append((url, timeout))
        return _FakeResponseContext(_FakeResponse(self.payloads[url]))


class _FakeBot:
    def __init__(self, session):
        self.session = session
        self.ensure_session_calls = 0

    async def ensure_session(self):
        self.ensure_session_calls += 1
        return self.session


class _FakeMessage:
    def __init__(self, attachments):
        self.body = SimpleNamespace(attachments=attachments)
        self.sender = SimpleNamespace(user_id=42, username="max-user")
        self.recipient = SimpleNamespace(chat_id=99)
        self.mid = "message-100"
        self.answers = []

    async def answer(self, text):
        self.answers.append(text)


class _FakeProcessingService:
    def __init__(self):
        self.incoming = []

    async def process_image(self, incoming):
        self.incoming.append(incoming)
        number = len(self.incoming)
        return SimpleNamespace(
            success=True,
            format1=f"format1-{number}",
            format2=f"format2-{number}",
            details=f"details-{number}",
        )


def _image_attachment(identifier, url):
    return {
        "type": "image",
        "payload": {
            "url": url,
            "photo_id": identifier,
            "filename": f"{identifier}.jpg",
        },
    }


class SharedContractTests(unittest.TestCase):
    def test_shared_service_imports_without_messenger_sdks(self):
        script = """
import sys
sys.modules[\"aiogram\"] = None
sys.modules[\"maxapi\"] = None
import services.passport_processing
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )
        self.assertNotIn("aiogram", passport_processing.__dict__)
        self.assertNotIn("maxapi", passport_processing.__dict__)

    def test_incoming_image_normalizes_source_and_identifiers(self):
        incoming = IncomingImage(
            content=bytearray(b"image"),
            filename="passport.jpg",
            source="max",
            external_user_id=42,
            external_chat_id=99,
            external_message_id=100,
            external_username=123,
            source_file_id=456,
        )

        self.assertIs(incoming.source, MessengerSource.MAX)
        self.assertIsInstance(incoming.content, bytes)
        self.assertEqual(incoming.external_user_id, "42")
        self.assertEqual(incoming.external_chat_id, "99")
        self.assertEqual(incoming.external_message_id, "100")
        self.assertEqual(incoming.external_username, "123")
        self.assertEqual(incoming.source_file_id, "456")


class SourceMetadataContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_metadata_keeps_equal_looking_ids_separate(self):
        repository = _FakeRepository()
        service = PassportProcessingService(
            repository_factory=lambda: repository,
            image_processor=_FakeImageProcessor(),
            recognizer=_FakeRecognizer(),
        )
        shared_ids = {
            "external_user_id": "42",
            "external_chat_id": "99",
            "external_message_id": "100",
        }

        await service.process_image(
            IncomingImage(
                content=b"telegram-image",
                filename="passport.jpg",
                source=MessengerSource.TELEGRAM,
                external_username="telegram-user",
                source_file_id="telegram-file",
                **shared_ids,
            )
        )
        await service.process_image(
            IncomingImage(
                content=b"max-image",
                filename="passport.jpg",
                source=MessengerSource.MAX,
                external_username="max-user",
                source_file_id="max-file",
                **shared_ids,
            )
        )

        telegram, max_record = repository.records
        self.assertEqual(telegram["source"], "telegram")
        self.assertEqual(telegram["source_type"], "photo")
        self.assertEqual(telegram["source_file_id"], "telegram-file")
        self.assertEqual(telegram["tg_user_id"], 42)
        self.assertEqual(telegram["source_message_id"], 100)
        self.assertEqual(telegram["external_user_id"], "42")
        self.assertEqual(telegram["external_chat_id"], "99")
        self.assertEqual(telegram["external_message_id"], "100")

        self.assertEqual(max_record["source"], "max")
        self.assertEqual(max_record["source_type"], "photo")
        self.assertEqual(max_record["source_file_id"], "max-file")
        self.assertIsNone(max_record["tg_user_id"])
        self.assertIsNone(max_record["source_message_id"])
        self.assertEqual(max_record["external_user_id"], "42")
        self.assertEqual(max_record["external_chat_id"], "99")
        self.assertEqual(max_record["external_message_id"], "100")

        identity_keys = {
            (
                record["source"],
                record["external_user_id"],
                record["external_chat_id"],
                record["external_message_id"],
            )
            for record in repository.records
        }
        self.assertEqual(
            identity_keys,
            {
                ("telegram", "42", "99", "100"),
                ("max", "42", "99", "100"),
            },
        )


class MaxExtractionContractTests(unittest.TestCase):
    def test_normal_and_forwarded_images_produce_same_common_input_shape(self):
        attachment = _image_attachment("photo-1", "https://example.test/photo-1")
        normal_message = SimpleNamespace(
            body=SimpleNamespace(attachments=[attachment]),
            sender=SimpleNamespace(user_id=42, username="max-user"),
            recipient=SimpleNamespace(chat_id=99),
            mid="message-100",
        )
        forwarded_message = SimpleNamespace(
            sender=SimpleNamespace(user_id=42, username="max-user"),
            recipient=SimpleNamespace(chat_id=99),
            mid="message-100",
            link=SimpleNamespace(
                message=SimpleNamespace(attachments=[attachment])
            ),
        )

        normal_event = SimpleNamespace(message=normal_message)
        forwarded_event = SimpleNamespace(message=forwarded_message)
        self.assertEqual(extract_image_attachments(normal_event), [attachment])
        self.assertEqual(
            extract_image_attachments(forwarded_event),
            [attachment],
        )

        normal = build_incoming_image(normal_event, attachment, b"image-bytes")
        forwarded = build_incoming_image(
            forwarded_event,
            attachment,
            b"image-bytes",
        )

        self.assertEqual(normal, forwarded)
        self.assertIs(normal.source, MessengerSource.MAX)
        self.assertEqual(normal.content, b"image-bytes")
        self.assertEqual(normal.external_user_id, "42")
        self.assertEqual(normal.external_chat_id, "99")
        self.assertEqual(normal.external_message_id, "message-100")
        self.assertEqual(normal.source_file_id, "photo-1")

    def test_unsupported_and_multiple_attachments_are_deterministic(self):
        first = _image_attachment("photo-1", "https://example.test/photo-1")
        unsupported = {"type": "file", "payload": {"url": "not-an-image"}}
        second = _image_attachment("photo-2", "https://example.test/photo-2")
        message = SimpleNamespace(
            body=SimpleNamespace(attachments=[first, unsupported, second])
        )

        self.assertEqual(extract_image_attachments(message), [first, second])
        self.assertEqual(extract_unsupported_attachments(message), [unsupported])
        self.assertEqual(extract_image_attachments(message), [first, second])
        self.assertEqual(extract_unsupported_attachments(message), [unsupported])

    def test_mixed_attachments_use_one_fake_session_and_preserve_order(self):
        first = _image_attachment("photo-1", "https://example.test/photo-1")
        unsupported = {"type": "file", "payload": {"url": "not-an-image"}}
        second = _image_attachment("photo-2", "https://example.test/photo-2")
        message = _FakeMessage([first, unsupported, second])
        session = _FakeSession(
            {
                "https://example.test/photo-1": b"first-bytes",
                "https://example.test/photo-2": b"second-bytes",
            }
        )
        bot = _FakeBot(session)
        service = _FakeProcessingService()

        processed = self._run(
            handle_message(message, service, bot),
        )

        self.assertEqual(len(processed), 2)
        self.assertEqual(bot.ensure_session_calls, 1)
        self.assertEqual(
            [request[0] for request in session.requests],
            [
                "https://example.test/photo-1",
                "https://example.test/photo-2",
            ],
        )
        self.assertEqual(
            [incoming.source_file_id for incoming in service.incoming],
            ["photo-1", "photo-2"],
        )
        self.assertEqual(
            [incoming.content for incoming in service.incoming],
            [b"first-bytes", b"second-bytes"],
        )
        self.assertEqual(
            message.answers,
            [
                UNSUPPORTED_ATTACHMENT_MESSAGE,
                "format1-1\n\nformat2-1\ndetails-1",
                "format1-2\n\nformat2-2\ndetails-2",
            ],
        )

    def test_only_unsupported_attachment_does_not_open_a_session(self):
        message = _FakeMessage([{"type": "file", "payload": {}}])
        session = _FakeSession({})
        bot = _FakeBot(session)
        service = _FakeProcessingService()

        processed = self._run(handle_message(message, service, bot))

        self.assertEqual(processed, [])
        self.assertEqual(bot.ensure_session_calls, 0)
        self.assertEqual(service.incoming, [])
        self.assertEqual(message.answers, [UNSUPPORTED_ATTACHMENT_MESSAGE])

    @staticmethod
    def _run(awaitable):
        import asyncio

        return asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
