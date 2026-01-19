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
        tg_user_id: int,
        tg_username: Optional[str],
        source_type: str,
        source_file_id: Optional[str],
        source_message_id: Optional[int],
        source_page_index: Optional[int],
        passport_number: Optional[str],
        issued_by: Optional[str],
        issue_date: Optional[date],
        subdivision_code: Optional[str],
        surname: Optional[str],
        name: Optional[str],
        middle_name: Optional[str],
        gender: Optional[str],
        birth_date: Optional[date],
        birth_place: Optional[str],
        raw_payload: dict,
        quality_score: int,
    ) -> PassportRecord:
        """Create a new passport record."""
        record = PassportRecord(
            id=uuid.uuid4(),
            tg_user_id=tg_user_id,
            tg_username=tg_username,
            source_type=source_type,
            source_file_id=source_file_id,
            source_message_id=source_message_id,
            source_page_index=source_page_index,
            passport_number=passport_number,
            issued_by=issued_by,
            issue_date=issue_date,
            subdivision_code=subdivision_code,
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
            user_id=tg_user_id,
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
