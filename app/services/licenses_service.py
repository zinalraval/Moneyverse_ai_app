from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from app.crud.licenses_crud import license
from app.schemas.licenses_schema import LicenseCreate
from app.models.user import User
from sqlalchemy import select
import uuid
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

async def create_license(
    db: Session, 
    user_email: str, 
    expiry_days: int, 
    features: list = None
) -> dict:
    """
    Create a new license for a user.
    
    Args:
        db: Database session
        user_email: Email of the user to create license for
        expiry_days: Number of days until license expires
        features: List of features (optional, not stored in current schema)
        
    Returns:
        dict: Created license data
    """
    try:
        # Find user by email
        result = await db.execute(select(User).where(User.email == user_email))
        user = result.scalar_one_or_none()
        
        if not user:
            raise ValueError(f"User with email {user_email} not found")
        
        # Generate unique license code
        license_code = f"LIC-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate expiration date
        expires_at = datetime.utcnow() + timedelta(days=expiry_days)
        
        # Create license object
        license_data = LicenseCreate(
            code=license_code,
            is_active=True,
            expires_at=expires_at,
            user_id=user.id
        )
        
        # Create license in database
        license_obj = await license.create(db, obj_in=license_data)
        
        logger.info(f"Created license {license_code} for user {user_email}")
        
        return {
            "code": license_obj.code,
            "user_email": user_email,
            "expires_at": license_obj.expires_at.isoformat() if license_obj.expires_at else None,
            "is_active": license_obj.is_active,
            "features": features or []
        }
        
    except Exception as e:
        logger.error(f"Error creating license: {str(e)}", exc_info=True)
        raise

async def verify_license(license_code: str, db: Session) -> bool:
    """
    Verify if a license is valid.
    
    Args:
        license_code: The license code to verify
        db: The database session to use
        
    Returns:
        bool: True if license is valid, False otherwise
    """
    logger.info(f"[DEBUG] Verifying license code: {license_code}")
    try:
        license_obj = await license.get_by_code(db, license_code)
        logger.info(f"[DEBUG] License object retrieved: {license_obj}")
        
        if not license_obj:
            logger.warning(f"[DEBUG] License not found: {license_code}")
            return False
            
        if not license_obj.is_active:
            logger.warning(f"[DEBUG] License is inactive: {license_code}")
            return False
            
        if license_obj.expires_at and license_obj.expires_at < datetime.utcnow():
            logger.warning(f"[DEBUG] License has expired: {license_code}")
            return False
            
        logger.info(f"[DEBUG] License is valid: {license_code}")
        return True
    except Exception as e:
        logger.error(f"[DEBUG] Error verifying license: {str(e)}", exc_info=True)
        return False 