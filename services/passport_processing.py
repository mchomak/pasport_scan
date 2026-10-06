"""Shared passport recognition and persistence service."""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Awaitable, Callable

from core.messaging import IncomingImage, MessengerSource
from db.repository import PassportRepository
from ocr.hybrid import HybridRecognizer
from ocr.models import PassportData
from ocr.openrouter import OpenRouterProvider
from ocr.provider import get_ocr_provider
from services.image_processor import ImageProcessor
from utils.passport_formatter import (
    format_passport_type1,
    format_passport_type2,
    infer_gender,
)
from utils.rate_limiter import MinuteRateLimiter


NotifyWait = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PassportResult:
    """Platform-neutral presentation data returned by the shared service."""

    format1: str
    format2: str
    details: str
    structured_details: dict[str, Any]
    recognition_state: str
    quality_score: int
    success: bool = True
    error: str | None = None
    record_id: str | None = None
    source: MessengerSource | None = None
    modules_used: tuple[str, ...] = ()

    @property
    def details_text(self) -> str:
        """Compatibility alias for adapters that name the field explicitly."""
        return self.details


class PassportProcessingService:
    """Own image normalization, OCR provider lifecycle, and persistence."""

    _FIELD_LABELS = {
        "surname": "Surname",
        "name": "Name",
        "middle_name": "Middle name",
        "passport_number": "Passport number",
        "birth_date": "Birth date",
        "expiry_date": "Expiry date",
        "gender": "Gender",
        "birth_place": "Birth place",
    }
    _PROVIDER_LABELS = {
        "yandex_ocr": "Yandex OCR",
        "openrouter": "OpenRouter LLM",
        "rupasportread": "Tesseract MRZ",
        "inferred": "Inferred",
    }

    def __init__(
        self,
        repository_factory: Callable[[], Any] | None = None,
        image_processor: ImageProcessor | None = None,
        recognizer: Any | None = None,
        settings_obj: Any | None = None,
        recognizer_factory: Callable[..., HybridRecognizer] = HybridRecognizer,
        yandex_provider_factory: Callable[[], Any] = get_ocr_provider,
        openrouter_provider_factory: Callable[[], Any] = OpenRouterProvider,
    ) -> None:
        from config import settings

        self._settings = settings_obj or settings
        self._repository_factory = repository_factory
        self._image_processor = image_processor or ImageProcessor()
        self._recognizer = recognizer
        self._owns_recognizer = recognizer is None
        self._recognizer_factory = recognizer_factory
        self._yandex_provider_factory = yandex_provider_factory
        self._openrouter_provider_factory = openrouter_provider_factory
        self._yandex_provider: Any | None = None
        self._openrouter_provider: Any | None = None
        self._limiter = MinuteRateLimiter(
            rpm=getattr(self._settings, "openrouter_rpm", 0)
        )
        self._started = False

    async def start(self) -> None:
        """Create provider clients and the hybrid recognizer once."""
        if self._started:
            return

        try:
            priority = self._settings.get_module_priority()
            if self._recognizer is None:
                if "yandex_ocr" in priority:
                    self._yandex_provider = self._yandex_provider_factory()
                if "openrouter" in priority and self._settings.openrouter_api_key:
                    self._openrouter_provider = self._openrouter_provider_factory()
                self._recognizer = self._recognizer_factory(
                    yandex_provider=self._yandex_provider,
                    openrouter_provider=self._openrouter_provider,
                )
            self._started = True
        except Exception:
            await self._close_provider(self._yandex_provider)
            await self._close_provider(self._openrouter_provider)
            self._yandex_provider = None
            self._openrouter_provider = None
            raise

    async def close(self) -> None:
        """Close provider HTTP clients; safe to call more than once."""
        if not self._started and not (
            self._yandex_provider or self._openrouter_provider
        ):
            return
        await self._close_provider(self._yandex_provider)
        await self._close_provider(self._openrouter_provider)
        self._yandex_provider = None
        self._openrouter_provider = None
        self._started = False
        if self._owns_recognizer:
            self._recognizer = None

    @staticmethod
    async def _close_provider(provider: Any | None) -> None:
        if provider is None:
            return
        close = getattr(provider, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def recognize_image(
        self,
        image_bytes: bytes,
        notify_wait: NotifyWait | None = None,
    ) -> PassportResult:
        """Recognize an image without writing a database record."""
        result, _, _ = await self._recognize(image_bytes, notify_wait)
        return result

    async def process_image(
        self,
        incoming: IncomingImage,
        notify_wait: NotifyWait | None = None,
    ) -> PassportResult:
        """Recognize, persist, and return a platform-neutral result."""
        result, hybrid_result, passport_data = await self._recognize(
            incoming.content,
            notify_wait,
        )

        if not result.success:
            return replace(result, source=incoming.source)

        is_telegram = incoming.source is MessengerSource.TELEGRAM
        async with self._repository_context() as repository:
            record = await repository.create(
                tg_user_id=self._safe_int(incoming.external_user_id)
                if is_telegram
                else None,
                tg_username=incoming.external_username if is_telegram else None,
                source_type=incoming.source_type,
                source_file_id=incoming.source_file_id,
                source_message_id=self._safe_int(incoming.external_message_id)
                if is_telegram
                else None,
                source_page_index=incoming.source_page_index,
                passport_number=passport_data.passport_number,
                expiry_date=passport_data.expiry_date,
                surname=passport_data.surname,
                name=passport_data.name,
                middle_name=passport_data.middle_name,
                gender=passport_data.gender,
                birth_date=passport_data.birth_date,
                birth_place=passport_data.birth_place,
                raw_payload={
                    "modules_used": list(hybrid_result.modules_attempted),
                    "field_providers": dict(hybrid_result.field_providers),
                },
                quality_score=result.quality_score,
                source=incoming.source.value,
                external_user_id=incoming.external_user_id,
                external_chat_id=incoming.external_chat_id,
                external_message_id=incoming.external_message_id,
                external_username=incoming.external_username,
            )

        return replace(
            result,
            record_id=str(record.id),
            source=incoming.source,
        )

    async def _recognize(
        self,
        image_bytes: bytes,
        notify_wait: NotifyWait | None,
    ) -> tuple[PassportResult, Any, PassportData]:
        await self.start()
        priority = self._settings.get_module_priority()
        if "openrouter" in priority and self._openrouter_provider is not None:
            await self._limiter.acquire(notify_wait=notify_wait)

        normalized_bytes, mime_type = self._image_processor.normalize_image(image_bytes)
        hybrid_result = await self._recognizer.recognize(normalized_bytes, mime_type)
        passport_data = hybrid_result.passport_data
        if not passport_data.gender:
            inferred = infer_gender(passport_data.middle_name, passport_data.surname)
            if inferred:
                passport_data = passport_data.model_copy(update={"gender": inferred})
                hybrid_result.field_providers["gender"] = "inferred"

        quality_score = passport_data.count_filled_fields()
        modules_used = tuple(hybrid_result.modules_used)
        return (
            self._build_result(
                passport_data,
                hybrid_result,
                quality_score,
                modules_used,
            ),
            hybrid_result,
            passport_data,
        )

    def _build_result(
        self,
        passport_data: PassportData,
        hybrid_result: Any,
        quality_score: int,
        modules_used: tuple[str, ...],
    ) -> PassportResult:
        details = self._format_details(
            passport_data,
            hybrid_result.field_providers,
            hybrid_result.per_module_data,
        )
        structured_details = {
            "fields": passport_data.model_dump(mode="json"),
            "field_providers": dict(hybrid_result.field_providers),
            "modules_used": list(modules_used),
        }
        state = "recognized" if quality_score else "unrecognized"
        format1 = format_passport_type1(passport_data) if quality_score else ""
        format2 = format_passport_type2(passport_data) if quality_score else ""
        return PassportResult(
            format1=format1,
            format2=format2,
            details=details,
            structured_details=structured_details,
            recognition_state=state,
            quality_score=quality_score,
            success=quality_score > 0,
            modules_used=modules_used,
        )

    def _format_details(
        self,
        passport_data: PassportData,
        field_providers: dict[str, str],
        per_module_data: dict[str, PassportData],
    ) -> str:
        lines: list[str] = []
        for module_key, data in per_module_data.items():
            lines.append(f"[{self._PROVIDER_LABELS.get(module_key, module_key)}]")
            for field_name, field_label in self._FIELD_LABELS.items():
                value = getattr(data, field_name, None)
                lines.append(
                    f"  {'+' if value is not None and str(value).strip() else '-'} "
                    f"{field_label}: {value if value is not None and str(value).strip() else '---'}"
                )
            lines.append("")

        lines.append("[Result]")
        for field_name, field_label in self._FIELD_LABELS.items():
            value = getattr(passport_data, field_name, None)
            if value is None or not str(value).strip():
                lines.append(f"  {field_label}: ---")
                continue
            provider = self._PROVIDER_LABELS.get(
                field_providers.get(field_name, "?"),
                field_providers.get(field_name, "?"),
            )
            lines.append(f"  {field_label}: {value} ({provider})")
        return "\n".join(lines)

    @staticmethod
    def _safe_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @asynccontextmanager
    async def _repository_context(self) -> AsyncIterator[Any]:
        if self._repository_factory is not None:
            repository = self._repository_factory()
            if inspect.isawaitable(repository):
                repository = await repository
            if hasattr(repository, "__aenter__"):
                async with repository as entered:
                    yield entered
            else:
                yield repository
            return

        from db.database import async_session_maker

        if async_session_maker is None:
            raise RuntimeError("Database not initialized. Call runtime.start() first.")
        async with async_session_maker() as session:
            yield PassportRepository(session)
