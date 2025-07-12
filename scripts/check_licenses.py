import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.postgres import get_db_session
from app.models.license import License
from sqlalchemy import text

async def check_licenses():
    """Check available licenses in the database."""
    async with get_db_session() as db:
        print("=== LICENSE CHECK ===\n")
        
        # Check all licenses
        result = await db.execute(text("""
            SELECT id, code, is_active, created_at, expires_at, user_id
            FROM licenses 
            ORDER BY created_at DESC 
            LIMIT 10;
        """))
        licenses = result.fetchall()
        
        print(f"Found {len(licenses)} licenses:")
        for license in licenses:
            print(f"  - ID: {license[0]}, Code: {license[1]}, Active: {license[2]}, Created: {license[3]}, Expires: {license[4]}, User ID: {license[5]}")
        
        # Check active licenses
        result = await db.execute(text("""
            SELECT code, is_active, expires_at
            FROM licenses 
            WHERE is_active = true
            ORDER BY created_at DESC;
        """))
        active_licenses = result.fetchall()
        
        print(f"\nActive licenses ({len(active_licenses)}):")
        for license in active_licenses:
            print(f"  - Code: {license[0]}, Active: {license[1]}, Expires: {license[2]}")

if __name__ == "__main__":
    asyncio.run(check_licenses()) 