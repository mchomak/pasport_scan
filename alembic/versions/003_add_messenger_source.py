"""Add messenger-neutral source metadata.

Revision ID: 003
Revises: 002
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add source metadata without removing or rewriting existing records."""
    op.add_column(
        "passport_records",
        sa.Column(
            "source",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'telegram'"),
        ),
    )
    for name in (
        "external_user_id",
        "external_chat_id",
        "external_message_id",
        "external_username",
    ):
        op.add_column(
            "passport_records",
            sa.Column(name, sa.String(length=255), nullable=True),
        )
    op.alter_column("passport_records", "tg_user_id", nullable=True)


def downgrade() -> None:
    """Reject downgrade if MAX rows cannot fit the legacy Telegram schema."""
    bind = op.get_bind()
    has_max_rows = bind.execute(
        sa.text("SELECT 1 FROM passport_records WHERE tg_user_id IS NULL LIMIT 1")
    ).first()
    if has_max_rows:
        raise RuntimeError(
            "Cannot downgrade messenger source migration while MAX rows exist"
        )

    op.alter_column("passport_records", "tg_user_id", nullable=False)
    for name in (
        "external_username",
        "external_message_id",
        "external_chat_id",
        "external_user_id",
        "source",
    ):
        op.drop_column("passport_records", name)
