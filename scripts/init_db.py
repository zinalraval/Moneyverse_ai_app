import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.postgres import init_db, get_db
from app.models.license import License
from sqlalchemy.ext.asyncio import AsyncSession

async def create_test_license(session: AsyncSession):
    """Create a test license."""
    license = License(
        code="TEST-LICENSE-123",
        is_active=True
    )
    session.add(license)
    await session.commit()
    return license

async def main():
    """Initialize database and create test license."""
    try:
        # Initialize database
        await init_db()
        print("Database initialized successfully")

        # Create test license
        async for session in get_db():
            license = await create_test_license(session)
            print(f"Test license created: {license.code}")
            break

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 