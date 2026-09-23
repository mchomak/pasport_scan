import os
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


for _name, _value in {
    "BOT_TOKEN": "fake-telegram-token",
    "MAX_BOT_TOKEN": "fake-max-token",
    "ADMIN_IDS": "1",
    "DATABASE_URL": "postgresql+asyncpg://fake:fake@localhost/fake",
    "OCR_MODULE_PRIORITY": "",
}.items():
    os.environ.setdefault(_name, _value)


from bot.max_adapter import (  # noqa: E402
    PROCESSING_ERROR_MESSAGE,
    UNSUPPORTED_ATTACHMENT_MESSAGE,
    build_incoming_image,
    download_image,
    extract_image_attachments,
    extract_unsupported_attachments,
    handle_message,
    MaxDownloadError,
)
import bot.max_adapter as max_adapter  # noqa: E402
import bot.max_main as max_main  # noqa: E402
from core.messaging import IncomingImage, MessengerSource  # noqa: E402
import main as telegram_main  # noqa: E402
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
    def __init__(self, payloads, headers=None):
        self.payloads = payloads
        self.headers = dict(headers or {})
        self.requests = []
        self.closed = False

    def get(self, url, *, timeout):
        self.requests.append((url, timeout, dict(self.headers)))
        return _FakeResponseContext(_FakeResponse(self.payloads[url]))

    async def close(self):
        self.closed = True


class _FakeBot:
    def __init__(self, session, sdk_session=None):
        self.session = session
        self._passport_download_session = session
        self.sdk_session = sdk_session or session
        self.headers = {"Authorization": "fake-max-token"}
        self.ensure_session_calls = 0
        self.close_session_calls = 0

    async def ensure_session(self):
        self.ensure_session_calls += 1
        return self.sdk_session

    async def close_session(self):
        self.close_session_calls += 1
        await self.sdk_session.close()


class _FreshFakeBot:
    def __init__(self):
        self.close_session_calls = 0

    async def close_session(self):
        self.close_session_calls += 1


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


class _EmptyRecognizer:
    async def recognize(self, image_bytes, mime_type):
        return SimpleNamespace(
            passport_data=PassportData(),
            modules_used=["fake"],
            raw_response={"response": "empty"},
            field_providers={},
            per_module_data={},
        )


class _InferredGenderRecognizer:
    async def recognize(self, image_bytes, mime_type):
        return SimpleNamespace(
            passport_data=PassportData(
                passport_number="4619709685",
                surname="IVANOVA",
                name="ANNA",
                birth_date=date(1990, 1, 2),
                expiry_date=date(2030, 1, 2),
            ),
            modules_used=["fake"],
            raw_response={"response": "private"},
            field_providers={},
            per_module_data={},
        )


class _RedirectResponse:
    status = 302
    headers = {"location": "http://127.0.0.1/private"}

    async def read(self):
        return b"redirect"


class _RedirectContext:
    async def __aenter__(self):
        return _RedirectResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RedirectSession:
    def __init__(self):
        self.headers = {}
        self.requests = []
        self.closed = False

    def get(self, url, *, timeout, allow_redirects=True):
        self.requests.append(
            {"url": url, "timeout": timeout, "allow_redirects": allow_redirects}
        )
        return _RedirectContext()

    async def close(self):
        self.closed = True


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
        forwarded_with_empty_body = SimpleNamespace(
            body=SimpleNamespace(attachments=[]),
            sender=SimpleNamespace(user_id=42, username="max-user"),
            recipient=SimpleNamespace(chat_id=99),
            mid="message-100",
            link=SimpleNamespace(
                message=SimpleNamespace(attachments=[attachment])
            ),
        )

        normal_event = SimpleNamespace(message=normal_message)
        forwarded_event = SimpleNamespace(message=forwarded_message)
        empty_body_event = SimpleNamespace(message=forwarded_with_empty_body)
        self.assertEqual(extract_image_attachments(normal_event), [attachment])
        self.assertEqual(
            extract_image_attachments(forwarded_event),
            [attachment],
        )
        self.assertEqual(
            extract_image_attachments(empty_body_event),
            [attachment],
        )

        normal = build_incoming_image(normal_event, attachment, b"image-bytes")
        forwarded = build_incoming_image(
            forwarded_event,
            attachment,
            b"image-bytes",
        )
        empty_body_forwarded = build_incoming_image(
            empty_body_event,
            attachment,
            b"image-bytes",
        )

        self.assertEqual(normal, forwarded)
        self.assertEqual(normal, empty_body_forwarded)
        self.assertIs(normal.source, MessengerSource.MAX)
        self.assertEqual(normal.content, b"image-bytes")
        self.assertEqual(normal.external_user_id, "42")
        self.assertEqual(normal.external_chat_id, "99")
        self.assertEqual(normal.external_message_id, "message-100")
        self.assertEqual(normal.source_file_id, "photo-1")

    def test_forwarded_event_uses_dispatcher_user_without_recipient_fallback(self):
        attachment = _image_attachment("photo-1", "https://example.test/photo-1")
        event = SimpleNamespace(
            from_user=SimpleNamespace(user_id="forwarded-user", username="forwarded"),
            message=SimpleNamespace(
                body=SimpleNamespace(attachments=[]),
                sender=None,
                recipient=SimpleNamespace(chat_id="chat-99", user_id="bot-1"),
                mid="message-100",
                link=SimpleNamespace(message=SimpleNamespace(attachments=[attachment])),
            ),
        )

        incoming = build_incoming_image(event, attachment, b"image-bytes")

        self.assertEqual(incoming.external_user_id, "forwarded-user")
        self.assertEqual(incoming.external_username, "forwarded")
        self.assertEqual(incoming.external_chat_id, "chat-99")
        self.assertEqual(incoming.external_message_id, "message-100")

    def test_download_uses_unauthenticated_session_and_closes_at_shutdown(self):
        image_session = _FakeSession(
            {"https://example.test/photo-1": b"image-bytes"},
            headers={},
        )
        sdk_session = _FakeSession(
            {"https://example.test/photo-1": b"image-bytes"},
            headers={"Authorization": "fake-max-token"},
        )
        bot = _FakeBot(image_session, sdk_session=sdk_session)
        message = _FakeMessage(
            [_image_attachment("photo-1", "https://example.test/photo-1")]
        )

        processed = self._run(
            handle_message(message, _FakeProcessingService(), bot),
        )
        self._run(bot.close_session())

        self.assertEqual(len(processed), 1)
        self.assertEqual(len(image_session.requests), 1)
        self.assertNotIn("Authorization", image_session.requests[0][2])
        self.assertEqual(sdk_session.requests, [])
        self.assertTrue(image_session.closed)
        self.assertEqual(bot.close_session_calls, 1)

    def test_fresh_download_session_is_empty_reused_and_closed(self):
        created = []

        def session_factory(*, headers):
            session = _FakeSession(
                {"https://example.test/photo-1": b"image-bytes"},
                headers=headers,
            )
            created.append(session)
            return session

        original_client_session = max_adapter.ClientSession
        max_adapter.ClientSession = session_factory
        try:
            bot = _FreshFakeBot()
            first = self._run(
                max_adapter.download_image(
                    bot,
                    "https://example.test/photo-1",
                )
            )
            second_session = self._run(max_adapter.get_download_session(bot))
            self._run(bot.close_session())
        finally:
            max_adapter.ClientSession = original_client_session

        self.assertEqual(first, b"image-bytes")
        self.assertEqual(len(created), 1)
        self.assertIs(second_session, created[0])
        self.assertEqual(created[0].headers, {})
        self.assertTrue(created[0].closed)
        self.assertEqual(bot.close_session_calls, 1)

    def test_download_rejects_caller_supplied_authenticated_session(self):
        bot = _FreshFakeBot()
        authenticated = _FakeSession(
            {"https://example.test/photo-1": b"secret"},
            headers={"Authorization": "fake-max-token"},
        )

        with self.assertRaises(max_adapter.MaxDownloadError):
            self._run(
                max_adapter.download_image(
                    bot,
                    "https://example.test/photo-1",
                    session=authenticated,
                )
            )

        self.assertEqual(authenticated.requests, [])

    def test_download_rejects_private_and_loopback_urls_before_request(self):
        session = _FakeSession({})
        bot = _FakeBot(session)

        for url in (
            "http://127.0.0.1/image.jpg",
            "http://localhost/image.jpg",
            "http://[::1]/image.jpg",
            "http://169.254.169.254/latest/meta-data",
        ):
            with self.subTest(url=url):
                with self.assertRaises(MaxDownloadError):
                    self._run(download_image(bot, url))

        self.assertEqual(session.requests, [])

    def test_safe_resolver_rejects_any_unsafe_resolved_address(self):
        class _StaticResolver:
            def __init__(self, results):
                self.results = results

            async def resolve(self, host, port, family):
                return self.results

            async def close(self):
                return None

        resolver = max_adapter._PublicAddressResolver(
            _StaticResolver(
                [
                    {"host": "93.184.216.34", "port": 443},
                    {"host": "127.0.0.1", "port": 443},
                ]
            )
        )

        with self.assertRaises(OSError):
            self._run(resolver.resolve("cdn.example", 443, 0))

    def test_real_download_session_uses_non_cached_safe_connector(self):
        if max_adapter._AIOHTTP_CLIENT_SESSION_TYPE is None:
            self.skipTest("aiohttp is unavailable")

        bot = _FreshFakeBot()
        session = self._run(max_adapter.get_download_session(bot))
        try:
            connector = session.connector
            self.assertIsInstance(connector, max_adapter.TCPConnector)
            self.assertFalse(connector.use_dns_cache)
            self.assertIsInstance(
                connector._resolver,
                max_adapter._PublicAddressResolver,
            )
        finally:
            self._run(bot.close_session())

    def test_download_disables_redirects(self):
        session = _RedirectSession()
        bot = _FakeBot(session)

        with self.assertRaises(MaxDownloadError):
            self._run(download_image(bot, "https://example.test/photo-1"))

        self.assertEqual(len(session.requests), 1)
        self.assertFalse(session.requests[0]["allow_redirects"])

    def test_inferred_gender_is_persisted_with_returned_result(self):
        repository = _FakeRepository()
        service = PassportProcessingService(
            repository_factory=lambda: repository,
            image_processor=_FakeImageProcessor(),
            recognizer=_InferredGenderRecognizer(),
        )

        result = self._run(
            service.process_image(
                IncomingImage(
                    content=b"image",
                    filename="passport.jpg",
                    source=MessengerSource.MAX,
                    external_user_id="max-user",
                    external_chat_id="max-chat",
                    external_message_id="max-message",
                )
            )
        )

        self.assertEqual(result.structured_details["fields"]["gender"], "female")
        self.assertEqual(repository.records[0]["gender"], "female")

    def test_max_polling_error_is_propagated_after_cleanup(self):
        class _FailingDispatcher:
            def __init__(self):
                self.stopped = False

            async def start_polling(self, bot):
                raise RuntimeError("polling failed")

            async def stop_polling(self):
                self.stopped = True

        class _Runtime:
            def __init__(self):
                self.closed = False

            async def start(self):
                return object()

            async def close(self):
                self.closed = True

        class _Bot:
            def __init__(self):
                self.closed = False

            async def close_session(self):
                self.closed = True

        dispatcher = _FailingDispatcher()
        runtime = _Runtime()
        bot = _Bot()
        with patch.object(max_main, "register_handlers"):
            with self.assertRaisesRegex(RuntimeError, "polling failed"):
                self._run(
                    max_main.main(
                        settings_obj=SimpleNamespace(
                            max_bot_enabled=True,
                            max_bot_token="fake",
                        ),
                        bot=bot,
                        dispatcher=dispatcher,
                        runtime=runtime,
                    )
                )

        self.assertTrue(dispatcher.stopped)
        self.assertTrue(bot.closed)
        self.assertTrue(runtime.closed)

    def test_telegram_disabled_returns_before_runtime_start(self):
        with patch.object(
            telegram_main,
            "settings",
            SimpleNamespace(telegram_bot_enabled=False),
        ):
            self._run(telegram_main.main())

    def test_zero_field_ocr_is_not_persisted_or_formatted_as_success(self):
        repository = _FakeRepository()
        service = PassportProcessingService(
            repository_factory=lambda: repository,
            image_processor=_FakeImageProcessor(),
            recognizer=_EmptyRecognizer(),
        )
        image_session = _FakeSession(
            {"https://example.test/photo-1": b"image-bytes"},
        )
        bot = _FakeBot(image_session)
        message = _FakeMessage(
            [_image_attachment("photo-1", "https://example.test/photo-1")]
        )

        processed = self._run(handle_message(message, service, bot))

        self.assertEqual(processed, [])
        self.assertEqual(repository.records, [])
        self.assertEqual(message.answers, [PROCESSING_ERROR_MESSAGE])
        self.assertNotIn("unknown", "\n".join(message.answers))
        self.assertNotIn("000000", "\n".join(message.answers))

        result = self._run(service.recognize_image(b"image-bytes"))
        self.assertFalse(result.success)
        self.assertEqual(result.format1, "")
        self.assertEqual(result.format2, "")
        self.assertNotIn("unknown", result.format1.lower())
        self.assertNotIn("000000", result.format2)

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
        self.assertEqual(bot.ensure_session_calls, 0)
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
