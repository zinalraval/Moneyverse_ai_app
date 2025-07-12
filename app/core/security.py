from app.utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    decode_access_token,
    decode_refresh_token
)
from fastapi import WebSocket, status
from app.models.user import User
from app.db.postgres import get_db_session_factory
from sqlalchemy import select
import logging
from jose import jwt, JWTError, ExpiredSignatureError
from app.config import settings

logger = logging.getLogger(__name__)

__all__ = [
    'verify_password',
    'get_password_hash',
    'create_access_token',
    'create_refresh_token',
    'decode_token',
    'decode_access_token',
    'decode_refresh_token',
    'get_current_user_ws'
]

async def get_current_user_ws(websocket: WebSocket) -> User:
    """
    Get the current authenticated user from the WebSocket connection.
    
    Args:
        websocket: WebSocket connection
        
    Returns:
        User: The authenticated user
        
    Raises:
        WebSocketDisconnect: If authentication fails
    """
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        # Decode token
        payload = decode_access_token(token)
        if not payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        email: str = payload.get("sub")
        if not email:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        # Get user from database
        async with get_db_session_factory() as db:
            result = await db.execute(select(User).where(User.email == email))
            user = result.scalars().first()

            if not user:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None

            return user

    except Exception as e:
        logger.error(f"WebSocket authentication error: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return None 

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None 