import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from app.models.user import User
from app.models.signal import Signal
import uuid

@pytest.mark.asyncio
async def test_db_connection(db_session: AsyncSession):
    """Test database connection and session creation."""
    assert db_session is not None
    # Try a simple query
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1

@pytest.mark.asyncio
async def test_user_model(db_session: AsyncSession):
    """Test User model operations."""
    # Create a test user with a unique email
    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    test_user = User(
        email=unique_email,
        hashed_password="hashed_password",
        full_name="Test User",
        is_active=True
    )
    db_session.add(test_user)
    await db_session.commit()
    
    # Query the user using ORM
    result = await db_session.execute(select(User).where(User.email == unique_email))
    user = result.scalars().first()
    assert user is not None
    assert user.email == unique_email 
