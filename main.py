"""Telegram bot entry point."""

import asyncio
import os
import traceback

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent
import uvicorn

from config import settings
from utils.logger import setup_logger, get_logger

# Configure logging before importing adapters; they create loggers at import time.
setup_logger(settings.log_level)
logger = get_logger(__name__)

from bot.handlers import router, set_processing_service as set_telegram_processing_service
from services.runtime import ApplicationRuntime
from web.app import (
    app as web_app,
    set_processing_service as set_web_processing_service,
)


async def main():
    """Start Telegram, the shared processing runtime, and optional web UI."""
    logger.info("Starting Passport OCR Bot")

    runtime = ApplicationRuntime()
    bot: Bot | None = None
    web_server: uvicorn.Server | None = None
    web_task: asyncio.Task | None = None

    try:
        processing_service = await runtime.start()
        set_telegram_processing_service(processing_service)
        set_web_processing_service(processing_service)

        bot = Bot(
            token=settings.bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        dp = Dispatcher()

        @dp.errors()
        async def on_error(event: ErrorEvent):
            """Keep polling alive after an unexpected handler exception."""
            logger.error(
                "Unhandled error in handler",
                error=str(event.exception),
                traceback=traceback.format_exception(event.exception),
            )

        dp.include_router(router)
        logger.info("Bot configured, starting polling")

        web_enabled = getattr(settings, "web_enabled", None)
        if web_enabled is None:
            web_enabled = os.getenv("WEB_ENABLED", "true").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }

        if web_enabled:
            web_config = uvicorn.Config(
                web_app,
                host="0.0.0.0",
                port=int(settings.web_port),
                log_level="info",
            )
            web_server = uvicorn.Server(web_config)
            web_task = asyncio.create_task(web_server.serve())
            logger.info("Web server starting", port=settings.web_port)

        await dp.start_polling(bot)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Bot polling error", error=str(e))
    finally:
        if web_server is not None and web_task is not None:
            web_server.should_exit = True
            try:
                await web_task
            except Exception as e:
                logger.error("Web server shutdown failed", error=str(e))
        if bot is not None:
            await bot.session.close()
        await runtime.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.error("Fatal error", error=str(e), traceback=traceback.format_exc())
