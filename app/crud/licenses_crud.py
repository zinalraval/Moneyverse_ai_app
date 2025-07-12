from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.license import License
from app.schemas.licenses_schema import LicenseCreate, LicenseUpdate
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class LicenseCRUD:
    async def get_by_code(self, db: Session, code: str) -> License:
        """Get license by code"""
        result = await db.execute(select(License).where(License.code == code))
        return result.scalar_one_or_none()

    async def create(self, db: Session, *, obj_in: LicenseCreate) -> License:
        """Create new license"""
        db_obj = License(
            code=obj_in.code,
            is_active=obj_in.is_active,
            expires_at=obj_in.expires_at.replace(tzinfo=None) if obj_in.expires_at is not None else None,
            user_id=obj_in.user_id
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(self, db: Session, license_id: int, license: LicenseUpdate) -> License:
        """Update license"""
        db_license = await db.get(License, license_id)
        if not db_license:
            return None
        
        for field, value in license.dict(exclude_unset=True).items():
            setattr(db_license, field, value)
        
        await db.commit()
        await db.refresh(db_license)
        return db_license

    async def validate(self, db: Session, code: str) -> bool:
        """Validate license code"""
        license = await self.get_by_code(db, code)
        if not license:
            return False
        
        if not license.is_active:
            return False
        
        if license.expires_at and license.expires_at < datetime.utcnow():
            return False
        
        return True

    async def deactivate(self, db: Session, code: str) -> License:
        """Deactivate license and return the License object"""
        license = await self.get_by_code(db, code)
        if not license:
            return None
        license.is_active = False
        await db.commit()
        await db.refresh(license)
        return license

    async def extend(self, db: Session, code: str, days: int) -> License:
        """Extend license expiration date"""
        license = await self.get_by_code(db, code)
        if not license:
            return None
        
        if license.expires_at:
            license.expires_at += timedelta(days=days)
        else:
            license.expires_at = datetime.utcnow() + timedelta(days=days)
        
        await db.commit()
        await db.refresh(license)
        return license

    async def get_active_by_user_id(self, db: Session, user_id: int) -> list[License]:
        """Get all active licenses for a user"""
        result = await db.execute(
            select(License).where(
                License.user_id == user_id,
                License.is_active == True,
                (License.expires_at.is_(None) | (License.expires_at > datetime.utcnow()))
            )
        )
        return result.scalars().all()

# Create a singleton instance
license = LicenseCRUD() 