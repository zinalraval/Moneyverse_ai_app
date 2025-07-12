"""update_signal_status_to_enum

Revision ID: 8abf42b68224
Revises: 86a696f414e9
Create Date: 2025-06-18 06:31:10.690137

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8abf42b68224'
down_revision: Union[str, None] = '86a696f414e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create the enum type
    signalstatus = postgresql.ENUM('ACTIVE', 'TP1_HIT', 'COMPLETED', 'CANCELLED', 'WAITING_FOR_NEXT', name='signalstatus')
    signalstatus.create(op.get_bind())
    
    # Convert the status column to use the enum
    op.alter_column('signals', 'status',
                    type_=signalstatus,
                    postgresql_using="status::signalstatus")


def downgrade() -> None:
    """Downgrade schema."""
    # Convert back to string
    op.alter_column('signals', 'status',
                    type_=sa.String(length=20),
                    postgresql_using="status::text")
    
    # Drop the enum type
    signalstatus = postgresql.ENUM(name='signalstatus')
    signalstatus.drop(op.get_bind())
