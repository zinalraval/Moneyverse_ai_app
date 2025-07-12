import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.postgres import init_db, get_db
try:
    from app.db.postgres import async_init_db
except ImportError:
    async_init_db = None
from app.models.user import User
from app.utils.security import get_password_hash
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def create_admin_user(session: AsyncSession):
    """Create admin user if it doesn't exist."""
    # Check if admin user already exists
    result = await session.execute(select(User).where(User.email == "admin@moneyverse.ai"))
    existing_user = result.scalars().first()
    
    if existing_user:
        print("Admin user already exists")
        return existing_user
    
    # Create admin user
    hashed_password = get_password_hash("admin123")
    admin_user = User(
        email="admin@moneyverse.ai",
        hashed_password=hashed_password,
        full_name="Admin User",
        is_active=True
    )
    
    session.add(admin_user)
    await session.commit()
    await session.refresh(admin_user)
    
    print(f"Admin user created: {admin_user.email}")
    return admin_user

async def main():
    """Create admin user."""
    try:
        # Initialize database
        if async_init_db:
            await async_init_db()
        else:
            init_db()
        print("Database initialized successfully")

        # Create admin user
        async for session in get_db():
            user = await create_admin_user(session)
            break

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 