"""Shared application startup and shutdown lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, Awaitable, Callable

from db.database import close_db, create_tables, init_db
from services.passport_processing import PassportProcessingService


DB_INIT_RETRIES = 5
DB_INIT_DELAY = 3.0


def validate_ocr_configuration(settings_obj: Any) -> list[str]:
    """Validate enabled OCR providers without exposing credential values."""
    priority = settings_obj.get_module_priority()
    if "yandex_ocr" in priority:
        if not settings_obj.yc_folder_id:
            raise ValueError("yandex_ocr requires YC_FOLDER_ID")
        if not (
            settings_obj.yc_api_key
            or settings_obj.yc_iam_token
            or settings_obj.yc_oauth_token
        ):
            raise ValueError(
                "yandex_ocr requires YC_API_KEY, YC_IAM_TOKEN, or YC_OAUTH_TOKEN"
            )
    if "openrouter" in priority and not settings_obj.openrouter_api_key:
        raise ValueError("openrouter requires OPENROUTER_API_KEY")
    return priority


class ApplicationRuntime:
    """Own shared DB/OCR startup and deterministic resource closure."""

    def __init__(
        self,
        service: PassportProcessingService | None = None,
        service_factory: Callable[[], PassportProcessingService] = PassportProcessingService,
        settings_obj: Any | None = None,
        db_retries: int = DB_INIT_RETRIES,
        db_retry_delay: float = DB_INIT_DELAY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        from config import settings

        self.settings = settings_obj or settings
        self.service = service
        self._service_factory = service_factory
        self._db_retries = db_retries
        self._db_retry_delay = db_retry_delay
        self._sleep = sleep
        self._iam_refresh_task: asyncio.Task[Any] | None = None
        self._started = False

    async def start(self) -> PassportProcessingService:
        """Initialize DB, validate OCR configuration, and start the service."""
        if self._started:
            if self.service is None:
                raise RuntimeError("Runtime started without a processing service")
            return self.service

        await self._init_database()
        try:
            priority = validate_ocr_configuration(self.settings)
            if "yandex_ocr" in priority and not self.settings.yc_api_key:
                from utils.iam_refresher import refresh_iam_token, start_iam_refresh_loop

                if self.settings.yc_oauth_token:
                    token = await refresh_iam_token()
                    if token:
                        self.settings.yc_iam_token = token
                    else:
                        raise ValueError(
                            "YC_OAUTH_TOKEN exchange did not return an IAM token"
                        )
                self._iam_refresh_task = asyncio.create_task(
                    start_iam_refresh_loop()
                )

            if self.service is None:
                self.service = self._service_factory()
            await self.service.start()
            self._started = True
            return self.service
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Cancel refresh tasks and close OCR providers and the DB."""
        if self._iam_refresh_task is not None:
            self._iam_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._iam_refresh_task
            self._iam_refresh_task = None

        if self.service is not None:
            await self.service.close()
        await close_db()
        self._started = False

    async def _init_database(self) -> None:
        for attempt in range(1, self._db_retries + 1):
            try:
                init_db()
                await create_tables()
                return
            except Exception:
                await close_db()
                if attempt == self._db_retries:
                    raise
                await self._sleep(self._db_retry_delay * attempt)


Runtime = ApplicationRuntime
