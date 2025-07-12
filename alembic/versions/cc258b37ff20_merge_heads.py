"""Merge heads

Revision ID: cc258b37ff20
Revises: 20250623_add_sl_hit_tp2_hit, 5ef9d5fdb9dc
Create Date: 2025-06-23 17:18:52.373469

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cc258b37ff20'
down_revision: Union[str, None] = ('20250623_add_sl_hit_tp2_hit', '5ef9d5fdb9dc')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
