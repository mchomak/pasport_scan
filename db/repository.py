"""Database repository for passport records."""
import uuid
from datetime import date
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import PassportRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class PassportRepository:
    """Repository for passport records operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tg_user_id: Optional[int],
        tg_username: Optional[str],
        source_type: str,
        source_file_id: Optional[str],
        source_message_id: Optional[int],
        source_page_index: Optional[int],
        passport_number: Optional[str],
        expiry_date: Optional[date],
        surname: Optional[str],
        name: Optional[str],
        middle_name: Optional[str],
        gender: Optional[str],
        birth_date: Optional[date],
        birth_place: Optional[str],
        raw_payload: dict,
        quality_score: int,
        *,
        source: str = "telegram",
        external_user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
        external_message_id: Optional[str] = None,
        external_username: Optional[str] = None,
    ) -> PassportRecord:
        """Create a new passport record."""
        source_value = getattr(source, "value", source)
        external_user_id = (
            str(external_user_id) if external_user_id is not None else None
        )
        external_chat_id = (
            str(external_chat_id) if external_chat_id is not None else None
        )
        external_message_id = (
            str(external_message_id) if external_message_id is not None else None
        )
        external_username = (
            str(external_username) if external_username is not None else None
        )
        if source_value == "telegram":
            external_user_id = (
                external_user_id
                if external_user_id is not None
                else str(tg_user_id) if tg_user_id is not None else None
            )
            external_message_id = (
                external_message_id
                if external_message_id is not None
                else str(source_message_id) if source_message_id is not None else None
            )
            external_username = (
                external_username if external_username is not None else tg_username
            )
        else:
            # Numeric Telegram columns must never receive MAX identifiers.
            tg_user_id = None
            tg_username = None
            source_message_id = None

        record = PassportRecord(
            id=uuid.uuid4(),
            tg_user_id=tg_user_id,
            tg_username=tg_username,
            source=source_value,
            external_user_id=external_user_id,
            external_chat_id=external_chat_id,
            external_message_id=external_message_id,
            external_username=external_username,
            source_type=source_type,
            source_file_id=source_file_id,
            source_message_id=source_message_id,
            source_page_index=source_page_index,
            passport_number=passport_number,
            expiry_date=expiry_date,
            surname=surname,
            name=name,
            middle_name=middle_name,
            gender=gender,
            birth_date=birth_date,
            birth_place=birth_place,
            raw_payload=raw_payload,
            quality_score=quality_score,
        )

        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)

        logger.info(
            "Created passport record",
            record_id=str(record.id),
            source=source_value,
            quality_score=quality_score
        )

        return record

    async def get_all(self) -> list[PassportRecord]:
        """Get all passport records ordered by creation date descending."""
        result = await self.session.execute(
            select(PassportRecord).order_by(PassportRecord.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_user(self, tg_user_id: int) -> list[PassportRecord]:
        """Get all records for a specific user."""
        result = await self.session.execute(
            select(PassportRecord)
            .where(PassportRecord.tg_user_id == tg_user_id)
            .order_by(PassportRecord.created_at.desc())
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Count total records."""
        result = await self.session.execute(
            select(PassportRecord)
        )
        return len(list(result.scalars().all()))
