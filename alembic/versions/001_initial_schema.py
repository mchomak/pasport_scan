"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-12

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create passport_records table."""
    op.create_table(
        'passport_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('tg_user_id', sa.BigInteger(), nullable=False),
        sa.Column('tg_username', sa.String(length=255), nullable=True),
        sa.Column('source_type', sa.String(length=50), nullable=False, comment='photo|pdf_page|image_document'),
        sa.Column('source_file_id', sa.String(length=255), nullable=True),
        sa.Column('source_message_id', sa.BigInteger(), nullable=True),
        sa.Column('source_page_index', sa.Integer(), nullable=True),
        sa.Column('passport_number', sa.String(length=50), nullable=True),
        sa.Column('issued_by', sa.Text(), nullable=True),
        sa.Column('issue_date', sa.Date(), nullable=True),
        sa.Column('subdivision_code', sa.String(length=20), nullable=True),
        sa.Column('surname', sa.String(length=255), nullable=True),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('middle_name', sa.String(length=255), nullable=True),
        sa.Column('gender', sa.String(length=10), nullable=True),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.Column('birth_place', sa.Text(), nullable=True),
        sa.Column('raw_payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('quality_score', sa.Integer(), nullable=False, server_default='0', comment='Number of filled fields'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_passport_records_tg_user_id'), 'passport_records', ['tg_user_id'], unique=False)


def downgrade() -> None:
    """Drop passport_records table."""
    op.drop_index(op.f('ix_passport_records_tg_user_id'), table_name='passport_records')
    op.drop_table('passport_records')
