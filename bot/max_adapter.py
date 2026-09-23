"""MAX messenger adapter for the shared passport processing service."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

try:
    from aiohttp import ClientError, ClientSession, ClientTimeout
except ImportError:  # pragma: no cover - maxapi brings aiohttp in production.
    ClientError = Exception  # type: ignore[assignment,misc]
    ClientSession = None  # type: ignore[assignment,misc]
    ClientTimeout = None  # type: ignore[assignment,misc]

from core.messaging import IncomingImage, MessengerSource
def _get_logger() -> Any:
    try:
        from utils.logger import get_logger

        return get_logger(__name__)
    except Exception:
        return logging.getLogger(__name__)


logger = _get_logger()

DOWNLOAD_TIMEOUT_SECONDS = 30.0
DOWNLOAD_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024

_DOWNLOAD_SESSION_ATTR = "_passport_download_session"
_DOWNLOAD_SESSION_CLOSE_WRAPPER_ATTR = "_passport_download_close_wrapper"

NO_IMAGE_MESSAGE = "Пожалуйста, отправьте изображение паспорта."
UNSUPPORTED_ATTACHMENT_MESSAGE = (
    "Поддерживаются только изображения паспорта (JPEG или PNG)."
)
DOWNLOAD_ERROR_MESSAGE = "Не удалось скачать изображение. Попробуйте отправить его ещё раз."
PROCESSING_ERROR_MESSAGE = (
    "Не удалось распознать паспорт. Попробуйте отправить более чёткое изображение."
)


NO_RESULT_MESSAGE = PROCESSING_ERROR_MESSAGE


class MaxAdapterError(Exception):
    """Base class for errors that belong to one MAX message or attachment."""


class MaxDownloadError(MaxAdapterError):
    """A remote image could not be downloaded safely."""


class MaxFileTooLargeError(MaxDownloadError):
    """A remote image exceeded the configured in-memory size limit."""


def _log_error(message: str, exc: BaseException) -> None:
    error_type = type(exc).__name__
    try:
        logger.error(message, error_type=error_type)
    except TypeError:
        logger.error("%s (error_type=%s)", message, error_type)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _message_from_event(message_or_event: Any) -> Any:
    message = _value(message_or_event, "message")
    return message if message is not None else message_or_event


def _linked_body(message: Any) -> Any:
    link = _value(message, "link")
    return _value(link, "message")


def extract_attachments(message_or_event: Any) -> list[Any]:
    """Return attachments from a normal or forwarded MAX message.

    MAX omits ``message.body`` for a forwarded-only message.  In that case the
    linked message body is the only attachment source.  The helper deliberately
    returns SDK/duck-typed attachment objects unchanged so it is pure and easy
    to exercise without a live MAX connection.
    """

    message = _message_from_event(message_or_event)
    body_attachments = _as_list(
        _value(_value(message, "body"), "attachments")
    )
    linked_attachments = _as_list(
        _value(_linked_body(message), "attachments")
    )
    return body_attachments + linked_attachments


def _attachment_type(attachment: Any) -> str:
    value = _value(attachment, "type", "")
    value = _value(value, "value", value)
    return str(value).lower()


def extract_image_attachments(message_or_event: Any) -> list[Any]:
    """Return only MAX attachments whose type is ``image``."""

    return [
        attachment
        for attachment in extract_attachments(message_or_event)
        if _attachment_type(attachment) == "image"
    ]


def extract_unsupported_attachments(message_or_event: Any) -> list[Any]:
    """Return attachments that cannot be processed as MAX images."""

    return [
        attachment
        for attachment in extract_attachments(message_or_event)
        if _attachment_type(attachment) != "image"
    ]


def attachment_url(attachment: Any) -> str | None:
    """Get an image URL from typed MAX payload/photo metadata."""

    payload = _value(attachment, "payload")
    photo = _value(payload, "photo")
    url = _first_present(
        _value(payload, "url"),
        _value(photo, "url"),
        _value(attachment, "url"),
    )
    if not isinstance(url, str) or not url.strip():
        return None
    return url.strip()


def attachment_identifier(attachment: Any) -> str | None:
    """Return a non-secret attachment identifier suitable for persistence."""

    payload = _value(attachment, "payload")
    photo = _value(payload, "photo")
    identifier = _first_present(
        _value(payload, "photo_id"),
        _value(payload, "file_id"),
        _value(photo, "photo_id"),
        _value(photo, "id"),
        _value(attachment, "photo_id"),
        _value(attachment, "file_id"),
        _value(attachment, "id"),
    )
    return None if identifier is None else str(identifier)


def attachment_filename(attachment: Any) -> str | None:
    """Return an optional payload-provided filename without exposing its URL."""

    payload = _value(attachment, "payload")
    filename = _first_present(
        _value(payload, "filename"),
        _value(payload, "file_name"),
        _value(payload, "name"),
        _value(attachment, "filename"),
        _value(attachment, "file_name"),
    )
    return filename if isinstance(filename, str) and filename.strip() else None


def _message_metadata(
    message: Any,
    event: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    sender = _value(message, "sender")
    event_user = _value(event, "from_user")
    recipient = _value(message, "recipient")
    link = _value(message, "link")
    linked_body = _value(link, "message")

    user_id = _first_present(
        _value(sender, "user_id"),
        _value(event_user, "user_id"),
    )
    username = _first_present(
        _value(sender, "username"),
        _value(event_user, "username"),
    )
    chat_id = _first_present(
        _value(recipient, "chat_id"),
        _value(recipient, "user_id"),
        _value(link, "chat_id"),
    )
    message_id = _first_present(
        _value(message, "mid"),
        _value(_value(message, "body"), "mid"),
        _value(linked_body, "mid"),
    )
    return user_id, chat_id, message_id, username


def build_incoming_image(
    message_or_event: Any,
    attachment: Any,
    content: bytes,
) -> IncomingImage:
    """Build the shared input contract for one MAX image attachment."""

    message = _message_from_event(message_or_event)
    user_id, chat_id, message_id, username = _message_metadata(
        message,
        message_or_event,
    )
    return IncomingImage(
        content=content,
        filename=attachment_filename(attachment),
        source=MessengerSource.MAX,
        external_user_id=user_id,
        external_chat_id=chat_id,
        external_message_id=message_id,
        external_username=username,
        source_type="photo",
        source_file_id=attachment_identifier(attachment),
    )


make_incoming_image = build_incoming_image


def _max_file_limit_bytes(settings_obj: Any | None = None) -> int:
    if settings_obj is not None:
        configured = getattr(settings_obj, "ocr_max_file_bytes", None)
        if isinstance(configured, int) and configured > 0:
            return configured
        megabytes = getattr(settings_obj, "ocr_max_file_mb", None)
        if isinstance(megabytes, int) and megabytes > 0:
            return megabytes * 1024 * 1024

    try:
        from config import settings

        configured = getattr(settings, "ocr_max_file_bytes", None)
        if isinstance(configured, int) and configured > 0:
            return configured
    except Exception:
        pass
    return DEFAULT_MAX_FILE_BYTES


def _request_timeout(seconds: float) -> Any:
    if ClientTimeout is not None:
        return ClientTimeout(total=seconds)
    return seconds


def _session_is_closed(session: Any) -> bool:
    return bool(_value(session, "closed", False))


async def close_download_session(bot: Any) -> None:
    """Close the unauthenticated image-download session owned by ``bot``."""

    session = _value(bot, _DOWNLOAD_SESSION_ATTR)
    if session is None:
        return
    try:
        setattr(bot, _DOWNLOAD_SESSION_ATTR, None)
    except (AttributeError, TypeError):
        pass
    close = _value(session, "close")
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def _install_download_session_cleanup(bot: Any) -> None:
    if _value(bot, _DOWNLOAD_SESSION_CLOSE_WRAPPER_ATTR, False):
        return
    original_close = _value(bot, "close_session")
    if not callable(original_close):
        return

    async def close_session() -> None:
        try:
            await close_download_session(bot)
        finally:
            result = original_close()
            if inspect.isawaitable(result):
                await result

    try:
        setattr(bot, "close_session", close_session)
        setattr(bot, _DOWNLOAD_SESSION_CLOSE_WRAPPER_ATTR, True)
    except (AttributeError, TypeError):
        return


async def get_download_session(bot: Any) -> Any:
    """Return one reusable unauthenticated session for arbitrary image URLs."""

    session = _value(bot, _DOWNLOAD_SESSION_ATTR)
    if session is None or _session_is_closed(session):
        if ClientSession is None:
            raise MaxDownloadError("image download client is unavailable")
        session = ClientSession(headers={})
        setattr(bot, _DOWNLOAD_SESSION_ATTR, session)
    _install_download_session_cleanup(bot)
    return session


def _header_value(headers: Any, name: str) -> Any:
    if not isinstance(headers, Mapping):
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value
    return None


async def _iter_response_chunks(response: Any) -> AsyncIterator[bytes]:
    content = _value(response, "content")
    iter_chunked = _value(content, "iter_chunked")
    if callable(iter_chunked):
        async for chunk in iter_chunked(DOWNLOAD_CHUNK_SIZE):
            yield bytes(chunk)
        return

    if content is not None and hasattr(content, "__aiter__"):
        async for chunk in content:
            yield bytes(chunk)
        return

    read = _value(response, "read")
    if not callable(read):
        raise MaxDownloadError("response has no readable body")
    chunk = read()
    if inspect.isawaitable(chunk):
        chunk = await chunk
    if chunk:
        yield bytes(chunk)


async def _read_bounded_response(response: Any, max_bytes: int) -> bytes:
    status = _first_present(_value(response, "status"), _value(response, "status_code"))
    if status is None or not 200 <= int(status) < 300:
        raise MaxDownloadError("remote image returned a non-success status")

    content_length = _header_value(_value(response, "headers"), "content-length")
    try:
        declared_size = int(content_length) if content_length is not None else None
    except (TypeError, ValueError):
        declared_size = None
    if declared_size is not None and declared_size > max_bytes:
        raise MaxFileTooLargeError("remote image exceeds the configured size limit")

    data = bytearray()
    async for chunk in _iter_response_chunks(response):
        if len(data) + len(chunk) > max_bytes:
            raise MaxFileTooLargeError("remote image exceeds the configured size limit")
        data.extend(chunk)

    if not data:
        raise MaxDownloadError("remote image body is empty")
    return bytes(data)


async def download_image(
    bot: Any,
    url: str,
    *,
    session: Any | None = None,
    settings_obj: Any | None = None,
    max_bytes: int | None = None,
    timeout_seconds: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> bytes:
    """Download one image through the reusable unauthenticated session."""

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise MaxDownloadError("image URL is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MaxDownloadError("image URL is invalid")

    if session is None:
        session = await get_download_session(bot)
    request = getattr(session, "get", None)
    if request is None:
        request = getattr(session, "request", None)
        request_args = ("GET", url)
    else:
        request_args = (url,)
    if request is None:
        raise MaxDownloadError("MAX session cannot issue HTTP requests")

    try:
        pending_response = request(
            *request_args,
            timeout=_request_timeout(timeout_seconds),
        )

        if hasattr(pending_response, "__aenter__"):
            async with pending_response as response:
                return await _read_bounded_response(
                    response,
                    max_bytes or _max_file_limit_bytes(settings_obj),
                )

        response = pending_response
        if inspect.isawaitable(response):
            response = await response
        try:
            return await _read_bounded_response(
                response,
                max_bytes or _max_file_limit_bytes(settings_obj),
            )
        finally:
            release = _value(response, "release")
            if callable(release):
                result = release()
                if inspect.isawaitable(result):
                    await result
    except MaxDownloadError:
        raise
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise MaxDownloadError("image download timed out") from exc
    except ClientError as exc:
        raise MaxDownloadError("image download failed") from exc
    except Exception as exc:
        raise MaxDownloadError("image download failed") from exc


async def download_attachment(
    bot: Any,
    attachment: Any,
    *,
    session: Any | None = None,
    settings_obj: Any | None = None,
    max_bytes: int | None = None,
    timeout_seconds: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> bytes:
    """Resolve and download one typed MAX image attachment."""

    url = attachment_url(attachment)
    if url is None:
        raise MaxDownloadError("image attachment has no URL")
    return await download_image(
        bot,
        url,
        session=session,
        settings_obj=settings_obj,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _safe_answer(message: Any, text: str, bot: Any | None = None) -> bool:
    try:
        answer = _value(message, "answer")
        if callable(answer):
            await _maybe_await(answer(text))
            return True

        sender = _value(bot, "send_message")
        if callable(sender):
            _, chat_id, _, _ = _message_metadata(message)
            await _maybe_await(sender(chat_id=chat_id, text=text))
            return True
    except Exception as exc:
        _log_error("MAX response failed", exc)
    return False


def format_max_result(result: Any) -> str:
    """Render the shared result as plain text, without Telegram markup."""

    if _value(result, "success", True) is False:
        return NO_RESULT_MESSAGE

    details = _first_present(
        _value(result, "details_text"),
        _value(result, "details"),
        "",
    )
    return f"{_value(result, 'format1', '')}\n\n{_value(result, 'format2', '')}\n{details}".strip()


async def handle_message(
    event: Any,
    service: Any,
    bot: Any | None = None,
    *,
    settings_obj: Any | None = None,
) -> list[Any]:
    """Process one MAX event without allowing it to stop polling."""

    message = _message_from_event(event)
    bot = bot or _first_present(_value(event, "bot"), _value(message, "bot"))
    processed: list[Any] = []

    try:
        attachments = extract_attachments(message)
        if not attachments:
            await _safe_answer(message, NO_IMAGE_MESSAGE, bot)
            return processed

        image_attachments = [
            attachment
            for attachment in attachments
            if _attachment_type(attachment) == "image"
        ]
        if not image_attachments:
            await _safe_answer(message, UNSUPPORTED_ATTACHMENT_MESSAGE, bot)
            return processed

        if len(image_attachments) != len(attachments):
            await _safe_answer(message, UNSUPPORTED_ATTACHMENT_MESSAGE, bot)

        try:
            session = await get_download_session(bot)
        except Exception as exc:
            _log_error("MAX session acquisition failed", exc)
            await _safe_answer(message, DOWNLOAD_ERROR_MESSAGE, bot)
            return processed

        for attachment in image_attachments:
            try:
                content = await download_attachment(
                    bot,
                    attachment,
                    session=session,
                    settings_obj=settings_obj,
                )
            except MaxDownloadError as exc:
                _log_error("MAX image download failed", exc)
                await _safe_answer(message, DOWNLOAD_ERROR_MESSAGE, bot)
                continue

            try:
                incoming = build_incoming_image(event, attachment, content)
                result = await service.process_image(incoming)
                if _value(result, "success", True) is False:
                    await _safe_answer(message, format_max_result(result), bot)
                    continue
                processed.append(result)
                await _safe_answer(message, format_max_result(result), bot)
            except Exception as exc:
                _log_error("MAX image processing failed", exc)
                await _safe_answer(message, PROCESSING_ERROR_MESSAGE, bot)
    except Exception as exc:
        _log_error("MAX message handling failed", exc)
        await _safe_answer(message, PROCESSING_ERROR_MESSAGE, bot)

    return processed


def register_handlers(
    dispatcher: Any,
    service: Any,
    bot: Any,
    *,
    settings_obj: Any | None = None,
) -> Callable[..., Awaitable[Any]]:
    """Register the MAX message handler and return the registered callable."""

    async def on_message(event: Any) -> list[Any]:
        return await handle_message(
            event,
            service,
            bot,
            settings_obj=settings_obj,
        )

    dispatcher.message_created()(on_message)
    return on_message


__all__ = [
    "DOWNLOAD_TIMEOUT_SECONDS",
    "MaxAdapterError",
    "MaxDownloadError",
    "MaxFileTooLargeError",
    "NO_RESULT_MESSAGE",
    "attachment_filename",
    "attachment_identifier",
    "attachment_url",
    "build_incoming_image",
    "download_attachment",
    "download_image",
    "close_download_session",
    "extract_attachments",
    "extract_image_attachments",
    "extract_unsupported_attachments",
    "format_max_result",
    "get_download_session",
    "handle_message",
    "make_incoming_image",
    "register_handlers",
]
