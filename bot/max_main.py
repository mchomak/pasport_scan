"""Independent MAX bot entry point."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import suppress
from typing import Any

from bot.max_adapter import register_handlers


def _get_logger() -> Any:
    try:
        from utils.logger import get_logger

        return get_logger(__name__)
    except Exception:
        return logging.getLogger(__name__)


logger = _get_logger()

_SAFE_OCR_MODULES = frozenset({"openrouter", "yandex_ocr", "rupasportread"})


def _initialize_logger(settings_obj: Any | None) -> None:
    """Configure structured logging with the MAX source before startup checks."""

    from utils.logger import setup_logger

    log_level = _setting(settings_obj, "log_level", "LOG_LEVEL", "INFO")
    setup_logger(log_level, messenger_source="max")

    # Reacquire the logger after structlog configuration in case the module
    # started with the standard-library fallback.
    global logger
    logger = _get_logger()


def _ocr_modules(settings_obj: Any | None) -> list[str]:
    """Return configured OCR provider names without reading/logging credentials."""

    value: Any = None
    get_priority = getattr(settings_obj, "get_module_priority", None)
    if callable(get_priority):
        try:
            value = get_priority()
        except Exception:
            value = None
    if value is None:
        value = _setting(
            settings_obj,
            "ocr_module_priority",
            "OCR_MODULE_PRIORITY",
            "openrouter,rupasportread",
        )

    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = ()

    return [
        module
        for item in values
        if (module := str(item).strip().lower()) in _SAFE_OCR_MODULES
    ]


def _startup_diagnostics(settings_obj: Any | None) -> dict[str, Any]:
    modules = _ocr_modules(settings_obj)
    api_key = _setting(settings_obj, "openrouter_api_key", "OPENROUTER_API_KEY", "")
    return {
        "ocr_modules": modules,
        "openrouter_enabled": "openrouter" in modules and bool(str(api_key or "").strip()),
    }


def _log_error(message: str, exc: BaseException) -> None:
    error_type = type(exc).__name__
    try:
        logger.error(message, error_type=error_type)
    except TypeError:
        logger.error("%s (error_type=%s)", message, error_type)


def _load_settings() -> Any | None:
    try:
        from config import settings

        return settings
    except Exception:
        return None


def _setting(settings_obj: Any | None, attribute: str, env_name: str, default: Any) -> Any:
    if settings_obj is not None and hasattr(settings_obj, attribute):
        return getattr(settings_obj, attribute)
    return os.getenv(env_name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def max_enabled(settings_obj: Any | None = None) -> bool:
    """Return whether the MAX process is enabled by configuration."""

    return _as_bool(
        _setting(settings_obj, "max_bot_enabled", "MAX_BOT_ENABLED", False)
    )


def max_token(settings_obj: Any | None = None) -> str:
    """Read the MAX token without ever logging it."""

    value = _setting(settings_obj, "max_bot_token", "MAX_BOT_TOKEN", "")
    return value.strip() if isinstance(value, str) else str(value or "")


async def _close_object(obj: Any, method_name: str) -> None:
    method = getattr(obj, method_name, None)
    if callable(method):
        result = method()
        if inspect.isawaitable(result):
            await result


async def main(
    *,
    settings_obj: Any | None = None,
    bot: Any | None = None,
    dispatcher: Any | None = None,
    runtime: Any | None = None,
    service: Any | None = None,
) -> None:
    """Start MAX polling and close MAX/shared resources independently."""

    if settings_obj is None:
        settings_obj = _load_settings()
    _initialize_logger(settings_obj)
    logger.info("MAX bot initialization", **_startup_diagnostics(settings_obj))

    if not max_enabled(settings_obj):
        logger.info("MAX bot disabled")
        return

    token = max_token(settings_obj)
    if not token:
        raise RuntimeError("MAX_BOT_TOKEN is required when MAX_BOT_ENABLED is enabled")

    if bot is None or dispatcher is None:
        from maxapi import Bot, Dispatcher

        bot = bot or Bot(token)
        dispatcher = dispatcher or Dispatcher()

    if runtime is None:
        from services.runtime import ApplicationRuntime

        runtime = ApplicationRuntime(settings_obj=settings_obj)
    runtime_service = service
    try:
        if runtime_service is None:
            runtime_service = await runtime.start()
        register_handlers(
            dispatcher,
            runtime_service,
            bot,
            settings_obj=settings_obj,
        )
        await dispatcher.start_polling(bot)
    except asyncio.CancelledError:
        logger.info("MAX polling cancelled")
    except KeyboardInterrupt:
        logger.info("MAX polling stopped")
    except Exception as exc:
        _log_error("MAX polling failed", exc)
        raise
    finally:
        with suppress(Exception):
            await _close_object(dispatcher, "stop_polling")
        with suppress(Exception):
            await _close_object(bot, "close_session")
        with suppress(Exception):
            await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["main", "max_enabled", "max_token"]
