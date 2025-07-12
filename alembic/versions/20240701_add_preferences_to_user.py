"""
Add preferences JSON column to users table
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240701_add_preferences_to_user'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('users', sa.Column('preferences', postgresql.JSON(astext_type=sa.Text()), nullable=True))

def downgrade():
    op.drop_column('users', 'preferences') 