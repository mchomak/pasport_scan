"""Main application entry point."""
import asyncio
import sys
import traceback

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent
import uvicorn

from config import settings
from utils.logger import setup_logger, get_logger

# Setup logging BEFORE importing handlers (they create objects that log at import time)
setup_logger(settings.log_level)
logger = get_logger(__name__)

from db.database import init_db, create_tables, close_db
from bot.handlers import router
from web.app import app as web_app

DB_INIT_RETRIES = 5
DB_INIT_DELAY = 3  # seconds


async def _init_database():
    """Initialize database with retries (useful when waiting for PostgreSQL in Docker)."""
    for attempt in range(1, DB_INIT_RETRIES + 1):
        try:
            init_db()
            await create_tables()
            logger.info("Database ready", attempt=attempt)
            return
        except Exception as e:
            logger.warning(
                "Database init failed, retrying...",
                attempt=attempt,
                max_retries=DB_INIT_RETRIES,
                error=str(e),
            )
            if attempt == DB_INIT_RETRIES:
                logger.error("Database init failed after all retries")
                raise
            await asyncio.sleep(DB_INIT_DELAY * attempt)


async def main():
    """Main application function."""
    logger.info("Starting Passport OCR Bot")

    # Initialize database (with retries for Docker startup order)
    await _init_database()

    # Check OCR providers based on module priority
    priority = settings.get_module_priority()
    logger.info("OCR module priority", modules=priority)

    if "yandex_ocr" in priority:
        if not settings.yc_folder_id:
            logger.error("yandex_ocr is in priority but YC_FOLDER_ID is not set")
            sys.exit(1)

        if settings.yc_api_key:
            # Api-Key auth — no refresh needed, key never expires
            logger.info(
                "Yandex OCR: using Api-Key auth (permanent)",
                key_preview=settings.yc_api_key[:8] + "***",
            )
        else:
            # IAM token auth — needs refresh
            from utils.iam_refresher import refresh_iam_token, start_iam_refresh_loop

            token = await refresh_iam_token()
            if token:
                settings.yc_iam_token = token
            elif settings.yc_oauth_token:
                logger.error(
                    "YC_OAUTH_TOKEN is set but IAM token exchange failed. "
                    "Check the OAuth token value and network connectivity. "
                    "Consider using YC_API_KEY instead (does not expire)."
                )
                sys.exit(1)

            if not settings.yc_iam_token:
                logger.error(
                    "yandex_ocr needs auth credentials. Set one of:\n"
                    "  YC_API_KEY     — permanent API key (recommended)\n"
                    "  YC_OAUTH_TOKEN — auto-refresh via OAuth\n"
                    "  YC_IAM_TOKEN   — manual IAM token (expires in 12h)"
                )
                sys.exit(1)

            logger.info(
                "Yandex OCR: using IAM Bearer token",
                token_preview=settings.yc_iam_token[:8] + "***",
            )
            asyncio.create_task(start_iam_refresh_loop())

        try:
            from ocr.provider import get_ocr_provider
            get_ocr_provider()
            logger.info("Yandex OCR provider initialized")
        except Exception as e:
            logger.error("Failed to initialize Yandex OCR provider", error=str(e))
            sys.exit(1)

    if "openrouter" in priority:
        if not settings.openrouter_api_key:
            logger.error("OpenRouter is in priority but OPENROUTER_API_KEY is not set")
            sys.exit(1)
        logger.info(
            "OpenRouter configured",
            model=settings.openrouter_model,
            rpm_limit=settings.openrouter_rpm or "unlimited",
        )

    # Initialize bot and dispatcher
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    # Global error handler — catches any unhandled exception in handlers
    @dp.errors()
    async def on_error(event: ErrorEvent):
        logger.error(
            "Unhandled error in handler",
            error=str(event.exception),
            traceback=traceback.format_exception(event.exception),
        )
        # Don't re-raise — the bot stays alive

    # Register router
    dp.include_router(router)

    logger.info("Bot configured, starting polling")

    # Start web server in background
    web_port = int(settings.web_port) if hasattr(settings, 'web_port') else 8080
    web_config = uvicorn.Config(
        web_app,
        host="0.0.0.0",
        port=web_port,
        log_level="info",
    )
    web_server = uvicorn.Server(web_config)
    web_task = asyncio.create_task(web_server.serve())
    logger.info("Web server starting", port=web_port)

    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Bot polling error", error=str(e))
    finally:
        web_server.should_exit = True
        await web_task
        await bot.session.close()
        await close_db()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.error("Fatal error", error=str(e), traceback=traceback.format_exc())