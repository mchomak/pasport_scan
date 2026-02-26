"""Main application entry point."""
import asyncio
import sys
import traceback

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent

from config import settings
from db.database import init_db, create_tables, close_db
from bot.handlers import router
from utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(settings.log_level)
logger = get_logger(__name__)

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

    # Check OCR provider
    try:
        from ocr.provider import get_ocr_provider
        ocr_provider = get_ocr_provider()
        logger.info("OCR provider initialized", provider=settings.ocr_provider_model)
    except Exception as e:
        logger.error("Failed to initialize OCR provider", error=str(e))
        sys.exit(1)

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

    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Bot polling error", error=str(e))
    finally:
        await bot.session.close()
        await close_db()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.error("Fatal error", error=str(e), traceback=traceback.format_exc())