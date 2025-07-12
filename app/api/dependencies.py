from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError
from app.utils.security import decode_access_token, decode_refresh_token, decode_token
from app.db.postgres import get_db, get_db_session_factory, get_db_session
from app.models.user import User
from sqlalchemy import select
import logging
from typing import Optional, AsyncGenerator
from app.config import settings
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

# Configure OAuth2 scheme for access tokens
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=True,
    scheme_name="JWT"
)

# Configure OAuth2 scheme for refresh tokens
refresh_token_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/refresh",
    auto_error=True,
    scheme_name="JWT-Refresh"
)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from the JWT token.
    
    Args:
        token: JWT token from the Authorization header
        db: Database session
        
    Returns:
        User: The authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    payload = decode_access_token(token)
    if payload is None:
        logger.warning("get_current_user: Invalid or expired token detected, raising 401")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email: str = payload.get("sub")
    if not email:
        logger.warning("get_current_user: Token payload missing email, raising 401")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()

    if not user:
        logger.warning(f"get_current_user: No user found for email {email}, raising 401")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user

async def get_db() -> AsyncGenerator[Session, None]:
    """
    Get a database session.
    Yields:
        Session: Database session
    """
    logger.debug("[get_db] Attempting to get database session...")
    async with get_db_session() as session:
            logger.debug("[get_db] Database session obtained.")
            yield session
            logger.debug("[get_db] Database session closed.")

async def get_market_data():
    """
    Get the market data service instance.
    
    Returns:
        MarketDataService: The market data service instance
    """
    return market_data_service
