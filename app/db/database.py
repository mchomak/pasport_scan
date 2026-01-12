"""Database connection and session management."""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global engine and session maker
engine: AsyncEngine | None = None
async_session_maker: async_sessionmaker[AsyncSession] | None = None


def init_db() -> None:
    """Initialize database engine and session maker."""
    global engine, async_session_maker

    logger.info("Initializing database connection", database_url=settings.database_url)

    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info("Database initialized successfully")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_db() -> None:
    """Close database connection."""
    global engine

    if engine:
        logger.info("Closing database connection")
        await engine.dispose()
        engine = None
