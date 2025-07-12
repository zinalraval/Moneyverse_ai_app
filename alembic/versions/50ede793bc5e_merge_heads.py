"""merge heads

Revision ID: 50ede793bc5e
Revises: 20240701_add_preferences_to_user, cc258b37ff20
Create Date: 2025-06-30 11:50:37.690600

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '50ede793bc5e'
down_revision: Union[str, None] = ('20240701_add_preferences_to_user', 'cc258b37ff20')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
