"""Replace issue_date, issued_by, subdivision_code with expiry_date

Revision ID: 002
Revises: 001
Create Date: 2026-02-26

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Replace issue_date/issued_by/subdivision_code with expiry_date."""
    op.add_column('passport_records', sa.Column('expiry_date', sa.Date(), nullable=True))
    op.drop_column('passport_records', 'issue_date')
    op.drop_column('passport_records', 'issued_by')
    op.drop_column('passport_records', 'subdivision_code')


def downgrade() -> None:
    """Restore issue_date/issued_by/subdivision_code, drop expiry_date."""
    op.add_column('passport_records', sa.Column('subdivision_code', sa.String(length=20), nullable=True))
    op.add_column('passport_records', sa.Column('issued_by', sa.Text(), nullable=True))
    op.add_column('passport_records', sa.Column('issue_date', sa.Date(), nullable=True))
    op.drop_column('passport_records', 'expiry_date')
