"""Main application entry point."""
import asyncio
import sys
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from config import settings
from db.database import init_db, close_db
from bot.handlers import router
from utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(settings.log_level)
logger = get_logger(__name__)


async def main():
    """Main application function."""
    logger.info("Starting Passport OCR Bot")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        sys.exit(1)

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

    # Register router
    dp.include_router(router)

    logger.info("Bot configured, starting polling")

    try:
        # Start polling
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Bot error", error=str(e))
    finally:
        # Cleanup
        await bot.session.close()
        await close_db()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
