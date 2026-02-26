"""Database models."""
import uuid
from datetime import datetime, date
from typing import Optional
from sqlalchemy import String, BigInteger, Integer, Date, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class PassportRecord(Base):
    """Passport recognition record."""

    __tablename__ = "passport_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False
    )

    # Telegram info
    tg_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    tg_username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Source info
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="photo|pdf_page|image_document"
    )
    source_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    source_page_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Passport data
    passport_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    expiry_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Personal data
    surname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    middle_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    birth_place: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Raw OCR data
    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Quality metrics
    quality_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of filled fields"
    )

    def __repr__(self) -> str:
        return f"<PassportRecord {self.id} - {self.passport_number or 'N/A'}>"