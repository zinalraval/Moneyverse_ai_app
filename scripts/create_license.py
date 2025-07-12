import asyncio
import argparse
from datetime import datetime
from app.db.postgres import get_db
from app.models.license import License
from app.models.user import User
from sqlalchemy import select

async def create_license(code, user_email, expires=None):
    async for session in get_db():
        # Look up user by email
        result = await session.execute(select(User).where(User.email == user_email))
        user = result.scalars().first()
        if not user:
            print(f"User with email {user_email} not found.")
            return
        # Parse expires date if provided
        expires_at = None
        if expires:
            try:
                expires_at = datetime.strptime(expires, "%Y-%m-%d")
            except Exception as e:
                print(f"Invalid expires date format. Use YYYY-MM-DD. Error: {e}")
                return
        # Create license
        license = License(
            code=code,
            user_id=user.id,
            is_active=True,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        session.add(license)
        await session.commit()
        print(f'License created: {license.code} for user {user_email}')
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a license for a user.")
    parser.add_argument('--code', required=True, help='License code')
    parser.add_argument('--user', required=True, help='User email')
    parser.add_argument('--expires', required=False, help='Expiration date (YYYY-MM-DD)')
    args = parser.parse_args()
    asyncio.run(create_license(args.code, args.user, args.expires)) 