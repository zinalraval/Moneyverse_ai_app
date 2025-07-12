"""
Add SL_HIT and TP2_HIT to signalstatus enum
"""

revision = '20250623_add_sl_hit_tp2_hit'
down_revision = 'fc30a0fea708'
branch_labels = None
depends_on = None

from alembic import op

def upgrade():
    op.execute("ALTER TYPE signalstatus ADD VALUE IF NOT EXISTS 'SL_HIT'")
    op.execute("ALTER TYPE signalstatus ADD VALUE IF NOT EXISTS 'TP2_HIT'")

def downgrade():
    # Enum value removal is not supported in Postgres without recreating the type
    pass 