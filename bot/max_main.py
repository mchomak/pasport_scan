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
