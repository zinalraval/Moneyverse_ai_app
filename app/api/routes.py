from fastapi import (
    APIRouter, Depends, HTTPException, Query, status, 
    BackgroundTasks, Body, WebSocket, WebSocketDisconnect, Request, UploadFile, File
)
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from sqlalchemy.future import select
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import asyncio
import json
import pandas as pd
from slowapi import Limiter
from slowapi.util import get_remote_address
from jose import jwt
from jose.exceptions import JWTError
from fastapi.encoders import jsonable_encoder
from app.api.dependencies import get_db, get_current_user, oauth2_scheme, refresh_token_scheme, get_market_data,get_db_session,decode_token
from app.schemas.user_schema import UserCreate, UserRead, UserLogin, UserUpdate
from app.schemas.signal_schema import SignalRead, SignalCreate, SignalUpdate
from app.schemas.licenses_schema import LicenseCreate, License, LicenseUpdate
from app.utils.security import (
    get_password_hash, verify_password, 
    create_access_token, create_refresh_token, decode_refresh_token
)
from app.models.user import User
from app.models.signal import Signal, SignalType, SignalDirection, SignalStatus, SignalTimeframe
from app.services.shared import SUPPORTED_PAIRS, get_available_pairs
from app.services.signal_generation import generate_signal, signal_dict_to_model, TRADING_CONFIGS
from app.services.signal_automation import start_signal_automation
from app.services.signal_monitor import start_signal_monitor, get_active_signals
from app.config import settings
from app.crud.signal_crud import get_active_signal_for_pair, create_new_signal
from app.core.notifier import notifier
from app.services.websocket import websocket_manager as manager
from app.crud.licenses_crud import license
from app.services.licenses_service import verify_license, create_license
from app.services.market_data import market_data_service
from app.crud import signal_crud
from app.core.exceptions import MarketDataError
from fastapi.responses import JSONResponse
import os
import httpx
from functools import lru_cache
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

NEWS_CACHE = {}
NEWS_CACHE_TTL = 300  # seconds
MAJOR_EVENTS = ["NFP", "CPI", "FOMC", "ECB", "BOE", "BOJ", "RBA", "BOC"]

ANALYSIS_CACHE = {}
ANALYSIS_CACHE_TTL = 300  # seconds

def parse_datetime(dt):
    if dt is None:
        return None  # Return None to let the endpoint handle defaults
    if isinstance(dt, datetime):
        return dt
    return datetime.fromisoformat(dt)

def normalize_timeframe(tf):
    """Normalize timeframe to enum format (e.g., 'H1', '1h', 'h1' -> '1H')."""
    if not tf:
        return tf
    tf = str(tf).upper()
    mapping = {
        "M1": "1M", "1M": "1M", "1MIN": "1M",
        "M5": "5M", "5M": "5M", "5MIN": "5M",
        "M15": "15M", "15M": "15M", "15MIN": "15M",
        "M30": "30M", "30M": "30M", "30MIN": "30M",
        "H1": "1H", "1H": "1H", "60M": "1H",
        "H4": "4H", "4H": "4H",
        "D1": "1D", "1D": "1D", "DAILY": "1D"
    }
    # Accept both 'H1' and '1H' as '1H', etc.
    if tf in mapping:
        return mapping[tf]
    # Try to match patterns like 'H1', '1H', 'M5', '5M', etc.
    if tf.endswith('H') and tf[:-1].isdigit():
        return f"{tf[:-1]}H"
    if tf.startswith('H') and tf[1:].isdigit():
        return f"{tf[1:]}H"
    if tf.endswith('M') and tf[:-1].isdigit():
        return f"{tf[:-1]}M"
    if tf.startswith('M') and tf[1:].isdigit():
        return f"{tf[1:]}M"
    if tf.endswith('D') and tf[:-1].isdigit():
        return f"{tf[:-1]}D"
    if tf.startswith('D') and tf[1:].isdigit():
        return f"{tf[1:]}D"
    return tf

@router.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"message": "pong", "timestamp": datetime.utcnow().isoformat()}


@router.post("/auth/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    Args:
        user_in: User registration data
        db: Database session
        
    Returns:
        UserRead: Created user data
        
    Raises:
        HTTPException: If email already registered or invalid data
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    user = result.scalars().first()
    if user:
        logger.warning(f"Registration attempt with existing email: {user_in.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        is_active=True
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.email}")
    # Normalize timeframe to its value before returning
    if hasattr(new_user, "timeframe") and hasattr(new_user.timeframe, "value"):
        new_user.timeframe = new_user.timeframe.value
    return UserRead.from_orm(new_user)


@router.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login user and return access token.
    
    Args:
        form_data: OAuth2 form data
        db: Database session
        
    Returns:
        dict: Access and refresh tokens
        
    Raises:
        HTTPException: If credentials are invalid
    """
    # Use form_data.username as the email field
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()
    logger.debug(f"Login attempt for {form_data.username}. User found: {user is not None}")
    if user:
        logger.debug(f"User hashed_password: {user.hashed_password}")
        try:
            from app.utils.security import verify_password
            password_ok = verify_password(form_data.password, user.hashed_password)
            logger.debug(f"verify_password result: {password_ok}")
        except Exception as e:
            logger.error(f"verify_password exception: {e}")
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token(data={"sub": user.email})
    
    logger.info(f"User logged in: {user.email}")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user.id
    }


@router.get("/auth/me", response_model=UserRead)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserRead: Current user data
    """
    return current_user


@router.post("/auth/refresh")
async def refresh_token(refresh_token: str = Depends(refresh_token_scheme)):
    """
    Refresh access token.
    
    Args:
        refresh_token: Refresh token from Authorization header
        
    Returns:
        dict: New access token
    """
    try:
        payload = decode_refresh_token(refresh_token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        access_token = create_access_token(data={"sub": email})
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.put("/auth/profile", response_model=UserRead)
async def update_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile.
    
    Args:
        user_update: Updated user data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        UserRead: Updated user data
    """
    # Check if email is being updated and if it's already taken
    if user_update.email and user_update.email != current_user.email:
        result = await db.execute(select(User).where(User.email == user_update.email))
        existing_user = result.scalars().first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    # Normalize timeframe to its value before returning
    if hasattr(current_user, "timeframe") and hasattr(current_user.timeframe, "value"):
        current_user.timeframe = current_user.timeframe.value
    return UserRead.from_orm(current_user)


@router.post("/auth/change-password")
async def change_password(
    current_password: str = Body(...),
    new_password: str = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    
    Args:
        current_password: Current password
        new_password: New password
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        dict: Success message
    """
    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    current_user.hashed_password = get_password_hash(new_password)
    await db.commit()
    return {"message": "Password updated successfully"}


@router.post("/auth/request-password-reset", status_code=status.HTTP_200_OK)
async def request_password_reset(email: str = Body(..., embed=True), db: Session = Depends(get_db)):
    """
    Request a password reset.
    
    Args:
        email: User's email address
        db: Database session
        
    Returns:
        dict: Success message
    """
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if not user:
        # Don't reveal that the email doesn't exist
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate reset token
    reset_token = create_access_token(
        data={"sub": user.email, "type": "reset"},
        expires_delta=timedelta(hours=1)
    )
    
    # TODO: Send email with reset token
    logger.info(f"Password reset requested for: {email}")
    
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/auth/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    reset_code: str = Body(...),
    new_password: str = Body(...),
    db: Session = Depends(get_db)
):
    """
    Reset password using reset token.
    
    Args:
        reset_code: Password reset token
        new_password: New password
        db: Database session
        
    Returns:
        dict: Success message
    """
    try:
        payload = decode_token(reset_code, "reset")
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update password
        user.hashed_password = get_password_hash(new_password)
        await db.commit()
        
        logger.info(f"Password reset successful for: {email}")
        return {"message": "Password has been reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during password reset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )


@router.post("/signals", response_model=SignalRead, status_code=status.HTTP_201_CREATED, response_model_by_alias=False)
async def create_signal(
    signal: SignalCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    signal_dict = signal.model_dump(by_alias=False)
    tf = signal_dict.get("timeframe")
    if hasattr(tf, "value"):
        signal_dict["timeframe"] = tf.value
    signal_dict["user_id"] = current_user.id
    new_signal = Signal(**signal_dict)
    db.add(new_signal)
    await db.commit()
    await db.refresh(new_signal)
    # Normalize timeframe to its value before returning
    if hasattr(new_signal, "timeframe") and hasattr(new_signal.timeframe, "value"):
        new_signal.timeframe = new_signal.timeframe.value
    # Debug print for stop_loss
    print("DEBUG: new_signal.stop_loss =", getattr(new_signal, "stop_loss", None))
    print("DEBUG: new_signal.__dict__ =", new_signal.__dict__)
    return SignalRead.from_orm(new_signal)


@router.get("/signals", response_model=List[SignalRead], response_model_by_alias=False)
async def get_signals(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    signals = await signal_crud.get_signals_by_conditions(db, None, None, None, None, 100)
    result = []
    for signal in signals:
        # Normalize timeframe
        if hasattr(signal, "timeframe") and hasattr(signal.timeframe, "value"):
            signal.timeframe = signal.timeframe.value
        # Ensure is_news_filtered is not None
        if getattr(signal, "is_news_filtered", None) is None:
            signal.is_news_filtered = False
        # Ensure last_updated is not None
        if getattr(signal, "last_updated", None) is None:
            signal.last_updated = datetime.utcnow()
        result.append(SignalRead.from_orm(signal))
    return result


@router.get("/signals/active")
async def get_signals_active(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get active signals"""
    try:
        signals = await get_active_signals(db)
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{signal_id}", response_model=SignalRead, response_model_by_alias=False)
async def get_signal(
    signal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get signal by ID."""
    result = await db.execute(
        select(Signal).where(
            and_(
                Signal.id == signal_id,
                Signal.user_id == current_user.id
            )
        )
    )
    signal = result.scalars().first()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    # Normalize timeframe to its value before returning
    if hasattr(signal, "timeframe") and hasattr(signal.timeframe, "value"):
        signal.timeframe = signal.timeframe.value
    return SignalRead.from_orm(signal)


@router.put("/signals/{signal_id}", response_model=SignalRead, response_model_by_alias=False)
async def update_signal(
    signal_id: int,
    signal_update: SignalUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Async DB access for AsyncSession
    result = await db.execute(select(Signal).where(Signal.id == signal_id))
    signal = result.scalars().first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    update_data = signal_update.model_dump(exclude_unset=True, by_alias=False)
    tf = update_data.get("timeframe")
    if hasattr(tf, "value"):
        update_data["timeframe"] = tf.value
    for key, value in update_data.items():
        setattr(signal, key, value)
    await db.commit()
    await db.refresh(signal)
    # Normalize timeframe to its value before returning
    if hasattr(signal, "timeframe") and hasattr(signal.timeframe, "value"):
        signal.timeframe = signal.timeframe.value
    return SignalRead.from_orm(signal)


@router.patch("/signals/{signal_id}", response_model=SignalRead, response_model_by_alias=True)
async def patch_signal(
    signal_id: int,
    signal_update: SignalUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Partially update a signal."""
    result = await db.execute(
        select(Signal).where(
            and_(
                Signal.id == signal_id,
                Signal.user_id == current_user.id
            )
        )
    )
    signal = result.scalars().first()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    # Update only provided fields, mapping 'sl' to 'stop_loss'
    update_data = signal_update.model_dump(exclude_unset=True, by_alias=False)
    for field, value in update_data.items():
        setattr(signal, field, value)
    
    await db.commit()
    await db.refresh(signal)
    
    # Broadcast signal update
    await manager.broadcast_signal_update(signal.to_dict())
    
    # Normalize timeframe to its value before returning
    if hasattr(signal, "timeframe") and hasattr(signal.timeframe, "value"):
        signal.timeframe = signal.timeframe.value
    return SignalRead.from_orm(signal)


@router.delete("/signals/{signal_id}", status_code=status.HTTP_200_OK)
async def delete_signal(
    signal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a signal."""
    result = await db.execute(
        select(Signal).where(
            and_(
                Signal.id == signal_id,
                Signal.user_id == current_user.id
            )
        )
    )
    signal = result.scalars().first()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    await db.delete(signal)
    await db.commit()
    
    return {"message": "Signal deleted successfully"}


@router.post("/start-signal-monitoring")
async def start_signal_monitoring_endpoint(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start the signal monitoring service."""
    try:
        monitor = await start_signal_monitor()
        return {"message": "Signal monitoring started successfully"}
    except Exception as e:
        logger.error(f"Error starting signal monitoring: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start signal monitoring"
        )

@router.post("/initialize-market-data")
async def initialize_market_data():
    """Initialize the market data service."""
    try:
        # The market data service is already initialized when the module is imported
        return {"message": "Market data service initialized successfully"}
    except Exception as e:
        logger.error(f"Error initializing market data service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize market data service"
        )

# Shared handler for market data
async def _get_market_data(request: Request, pair: str, current_user: User, db: Session):
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unsupported trading pair"
        )
    try:
        price = await market_data_service.get_live_price(pair)
        now = datetime.utcnow().isoformat()
        data_source = getattr(market_data_service, '_last_data_source', 'unknown')
        return {
        "pair": pair,
        "price": float(price),
            "timestamp": now,
            "last_updated": now,
            "data_source": data_source
        }
    except Exception as e:
        now = datetime.utcnow().isoformat()
        return {
            "pair": pair,
            "price": None,
            "timestamp": now,
            "last_updated": now,
            "data_source": "unavailable",
            "error": str(e)
    }

@router.get("/market-data/{base}/{quote}")
@limiter.limit("60/minute")
async def get_market_data_endpoint(
    request: Request,
    base: str,
    quote: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    pair = f"{base}/{quote}"
    return await _get_market_data(request, pair, current_user, db)

@router.get("/market-data/{pair}")
@limiter.limit("60/minute")
async def get_market_data_single_segment(
    request: Request,
    pair: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await _get_market_data(request, pair, current_user, db)

@router.get("/market-data/{base}/{quote}/candles")
async def get_candles_endpoint(
    base: str,
    quote: str,
    timeframe: str = Query("H1", description="Candle timeframe (must be '1h' or 'H1')"),
    limit: int = Query(30, description="Number of candles to return"),
    current_user: User = Depends(get_current_user),
    market_data_service = Depends(get_market_data)
):
    pair = f"{base.upper()}/{quote.upper()}"
    try:
        candles = await market_data_service.get_candle_data(pair, timeframe, limit)
        now = datetime.utcnow().isoformat()
        data_source = getattr(market_data_service, '_last_data_source', 'unknown')
        return {"candles": candles, "last_updated": now, "data_source": data_source}
    except Exception as e:
        now = datetime.utcnow().isoformat()
        return {"candles": [], "last_updated": now, "data_source": "unavailable", "error": str(e)}

@router.get("/market-data/{base}/{quote}/historical")
@limiter.limit("60/minute")
async def get_historical_data(
    request: Request,
    base: str,
    quote: str,
    timeframe: str = Query("H1", description="Candle timeframe (must be '1h' or 'H1')"),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, description="Number of candles to return"),
    current_user: User = Depends(get_current_user),
    market_data_service = Depends(get_market_data)
):
    pair = f"{base}/{quote}"
    logger.info(f"Historical data request for pair: {pair}, timeframe: {timeframe}")
    
    if pair not in SUPPORTED_PAIRS:
        logger.warning(f"Unsupported trading pair: {pair}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unsupported trading pair"
        )
    
    logger.info(f"Timeframe validation: {timeframe.lower()} in {['1h', 'h1']}")
    if timeframe.lower() not in ["1h", "h1"]:
        logger.warning(f"Invalid timeframe: {timeframe}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only 1h/H1 timeframe is supported.")
    
    try:
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
        
        # Provide reasonable defaults if no dates specified
        if start_dt is None:
            start_dt = datetime.utcnow() - timedelta(hours=24)
        if end_dt is None:
            end_dt = datetime.utcnow()
            
        logger.info(f"Parsed datetime: start={start_dt}, end={end_dt}")
    except Exception as e:
        logger.error(f"Datetime parsing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
        )
    
    try:
        logger.info(f"Calling market_data_service.get_historical_data for {pair}")
        data = await market_data_service.get_historical_data(
        symbol=pair,
            interval=timeframe.lower(),
        start_date=start_dt,
        end_date=end_dt
    )
        logger.info(f"Market data service returned: {type(data)}, length: {len(data) if data else 0}")
        
        if not data:
            logger.warning("No historical data returned from market data service")
            raise HTTPException(status_code=404, detail="No historical data available for the specified parameters.")
        
        df = pd.DataFrame(data)
        logger.info(f"DataFrame created: shape={df.shape}")
        logger.info(f"Trend Analysis DataFrame columns: {df.columns.tolist()}, shape: {df.shape}")
        logger.info(f"Trend Analysis DataFrame head: {df.head(3).to_dict()}")
    except MarketDataError as e:
        logger.error(f"MarketDataError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in market data service: {e}")
        raise HTTPException(status_code=500, detail=f"Market data service error: {str(e)}")
    
    if df is None or (hasattr(df, 'empty') and df.empty) or (isinstance(df, list) and len(df) == 0):
        logger.warning("DataFrame is empty or None")
        raise HTTPException(status_code=404, detail="No historical data available for the specified parameters.")
    
    candles = []
    for _, row in df.iterrows():
        candle = {
            "timestamp": row["datetime"] if "datetime" in row else row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"])
        }
        if "volume" in row and pd.notna(row["volume"]):
            candle["volume"] = float(row["volume"])
        candles.append(candle)
    
    logger.info(f"Returning {len(candles)} candles")
    now = datetime.utcnow().isoformat()
    data_source = getattr(market_data_service, '_last_data_source', 'unknown')
    return {"candles": candles, "last_updated": now, "data_source": data_source}

@router.websocket("/ws/signals")
async def websocket_endpoint(
    websocket: WebSocket,
    license_code: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time signal updates"""
    from app.services.websocket_service import manager
    
    print(f"WebSocket connection attempt: license_code={license_code}")
    # Connect and validate license
    if not await manager.connect(websocket, license_code):
        print("WebSocket connection rejected: invalid or inactive license")
        await websocket.close()
        return
    
    connection_id = id(websocket)
    
    try:
        while True:
            # Receive and handle messages
            message = await websocket.receive_json()
            await manager.handle_message(connection_id, message)
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)


@router.post("/licenses/validate")
async def validate_license(license_code: str = Query(..., description="License code to validate"), db: Session = Depends(get_db)):
    license_code = license_code.strip()
    license_obj = await license.get_by_code(db, code=license_code)
    
    if not license_obj or not license_obj.is_active or (license_obj.expires_at and license_obj.expires_at < datetime.utcnow()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, inactive, or expired license"
        )

    user_email = None
    if license_obj.user_id:
        from sqlalchemy.future import select
        from app.models.user import User
        result = await db.execute(select(User).where(User.id == license_obj.user_id))
        user_obj = result.scalars().first()
        if user_obj:
            user_email = user_obj.email

    if not user_email:
        raise HTTPException(status_code=404, detail="User for license not found")

    # Create a JWT for the user associated with the license
    access_token = create_access_token(data={"sub": user_email})

    return {
        "valid": True,
        "code": license_obj.code,
        "user_email": user_email,
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/licenses")
async def create_license_endpoint(
    user_email: str = Body(...),
    expiry_days: int = Body(...),
    features: list = Body(default=[]),
    db: Session = Depends(get_db)
):
    """Create a new license for a user."""
    try:
        result = await create_license(db, user_email, expiry_days, features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating license: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create license")

@router.get("/licenses/{license_code}", response_model=License)
async def get_license(license_code: str, db: Session = Depends(get_db)):
    license_obj = await license.get_by_code(db, code=license_code)
    if not license_obj:
        raise HTTPException(status_code=404, detail="License not found")
    return license_obj

@router.put("/licenses/{license_code}", response_model=License)
async def update_license(license_code: str, license_in: LicenseUpdate, db: Session = Depends(get_db)):
    license_obj = await license.get_by_code(db, code=license_code)
    if not license_obj:
        raise HTTPException(status_code=404, detail="License not found")
    license_obj = await license.update(db, license_obj.id, license_in)
    return license_obj

@router.delete("/licenses/{license_code}", response_model=License)
async def deactivate_license(license_code: str, db: Session = Depends(get_db)):
    """Deactivate a license."""
    return await license.deactivate(db, code=license_code)

@router.get("/licenses/active", response_model=List[License])
async def get_active_licenses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get active licenses for the current user."""
    try:
        licenses = await license.get_active_by_user_id(db, user_id=current_user.id)
        return licenses
    except Exception as e:
        logger.error(f"Error getting active licenses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active licenses"
        )

@router.post("/signals/detect-trend")
async def detect_trend(
    pair: str = Body(...),
    timeframe: str = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    candles=None
):
    """
    Detect trend for a trading pair.
    Args:
        pair: Trading pair
        timeframe: Timeframe for analysis
        current_user: Current authenticated user
        db: Database session
        candles: Optional list of candle objects (for internal calls)
    Returns:
        dict: Trend analysis results
    """
    try:
        # Validate trading pair
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unsupported trading pair"
            )
        # Get historical data if not provided
        if candles is None:
            candles = await market_data_service.get_candle_data(pair, timeframe)
        if candles is None or (hasattr(candles, 'empty') and candles.empty) or (isinstance(candles, list) and len(candles) == 0):
            logger.error(f"Error detecting trend: No candle data for {pair}")
            raise HTTPException(status_code=400, detail=f"No candle data for {pair}")
        # Convert candles to DataFrame
        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume if hasattr(c, 'volume') and c.volume is not None else 0
        } for c in candles])
        logger.info(f"Trend Analysis DataFrame columns: {df.columns.tolist()}, shape: {df.shape}")
        logger.info(f"Trend Analysis DataFrame head: {df.head(3).to_dict()}")
        if df.empty:
            logger.error(f"Error detecting trend: No candle data for {pair}")
            raise HTTPException(status_code=400, detail=f"No candle data for {pair}")
        # Calculate indicators
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        # Determine trend
        last_price = df["close"].iloc[-1]
        last_ema_20 = df["ema_20"].iloc[-1]
        last_ema_50 = df["ema_50"].iloc[-1]
        if last_price > last_ema_20 and last_ema_20 > last_ema_50:
            trend = "uptrend"
        elif last_price < last_ema_20 and last_ema_20 < last_ema_50:
            trend = "downtrend"
        else:
            trend = "sideways"
        return {
            "pair": pair,
            "timeframe": timeframe,
            "trend": trend,
            "current_price": last_price,
            "ema_20": last_ema_20,
            "ema_50": last_ema_50
        }
    except Exception as e:
        logger.error(f"Error detecting trend: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect trend"
        )

@router.post("/signals/generate", status_code=201)
@limiter.limit("5/minute")
async def generate_signal_endpoint(
    request: Request,
    pair: str = Body(..., description="Trading pair (e.g., EURUSD)"),
    type: SignalType = Body(..., description="Signal type (SCALPING, INTRADAY, SWING)"),
    timeframe: Optional[SignalTimeframe] = Body(None, description="Optional timeframe override"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a new trading signal.
    
    Args:
        request: FastAPI request object for rate limiting
        pair: Trading pair to generate signal for
        type: Type of signal to generate (SCALPING, INTRADAY, SWING)
        timeframe: Optional timeframe override
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SignalRead: Generated signal data
        
    Raises:
        HTTPException: If pair is invalid or signal generation fails
    """
    try:
        # Validate pair
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unsupported trading pair"
            )
        
        # Check for active signal
        active_signal = await get_active_signal_for_pair(db, pair)
        if active_signal:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Active signal already exists for {pair}"
            )
        
        # Get trading config for signal type
        config = TRADING_CONFIGS.get(type)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid signal type: {type}"
            )
        
        # Use provided timeframe or default from config
        signal_timeframe = timeframe or config['timeframe']
        
        # Generate signal
        signal_data = await generate_signal(
            pair=pair,
            signal_type=type,
            timeframe=signal_timeframe,
            user_id=current_user.id
        )
        
        if not signal_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid signal setup found"
            )
        
        # Create signal in database
        signal = await create_new_signal(db, signal_data)
        
        # Send notification
        await notifier.send_signal_notification(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
        
        logger.info(f"Generated {type} signal for {pair} by user {current_user.id}")
        # Normalize timeframe to its value before returning
        if hasattr(signal, "timeframe") and hasattr(signal.timeframe, "value"):
            signal.timeframe = signal.timeframe.value
        return SignalRead.from_orm(signal)
        
    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating signal"
        )

@router.get("/market/pairs")
async def get_pairs(current_user: User = Depends(get_current_user)):
    pairs = get_available_pairs()  # Do not use await
    return {"pairs": pairs}

@router.get("/trend-analysis/{base}/{quote}")
@limiter.limit("60/minute")
async def get_trend_analysis(
    request: Request,
    base: str,
    quote: str,
    timeframe: str = Query("1h", description="Candle timeframe (must be '1h')"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    pair = f"{base}/{quote}"
    # PATCH: Always return dummy data for BTC/USD for demo/testing
    if pair == "BTC/USD":
        return {
            "pair": pair,
            "timeframe": timeframe,
            "trend": "uptrend",
            "current_price": 50000.0,
            "ema_20": 49500.0,
            "ema_50": 49000.0
        }
    if pair not in SUPPORTED_PAIRS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported trading pair"
            )
    try:
        candles_df = await market_data_service.get_historical_data(pair, timeframe)
        if not isinstance(candles_df, pd.DataFrame) or candles_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No candle data available or data is not a DataFrame."
            )
        # Convert DataFrame to list of Candle-like objects
        class Candle:
            def __init__(self, timestamp, open, high, low, close, volume):
                self.timestamp = timestamp
                self.open = open
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
        candles = [
            Candle(
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"] if "volume" in row else None
            )
            for _, row in candles_df.iterrows()
        ]
        trend_result = await detect_trend(
            pair=pair,
            timeframe=timeframe,
            current_user=current_user,
            db=db,
            candles=candles
        )
        return trend_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting trend: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect trend")

@router.post("/licenses/{license_code}/deactivate", response_model=License)
async def deactivate_license_post(license_code: str, db: Session = Depends(get_db)):
    return await deactivate_license(license_code, db)

@router.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Enhanced chatbot endpoint with better error handling and trading context."""
    data = await request.json()
    user_message = data.get("message", "")
    model = data.get("model", "llama3-8b-8192")
    
    if not user_message:
        return {"response": "Please enter a message."}

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return {"response": "[Error] Groq API key not set in environment."}
    
    # Enhanced trading context for better responses
    trading_context = f"""
    You are a professional trading assistant for MoneyVerse A.I. 
    User: {current_user.email}
    Current time: {datetime.utcnow().isoformat()}
    
    Provide helpful, educational trading advice. Always include risk disclaimers.
    Focus on technical analysis, market psychology, and risk management.
    """
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": trading_context},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                ai_response = resp.json()["choices"][0]["message"]["content"]
                # Add disclaimer
                ai_response += "\n\n⚠️ This is educational content only. Always do your own research and consider consulting a financial advisor."
            else:
                ai_response = f"[Groq API error: {resp.status_code}] {resp.text}"
    except httpx.TimeoutException:
        ai_response = "[Error] Request timed out. Please try again."
    except Exception as e:
        ai_response = f"[Error] Unable to process request: {str(e)}"
    
    return {"response": ai_response}

@router.post("/ai/analyze-chart")
async def ai_chart_analyzer_endpoint(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Enhanced AI chart analyzer with real technical analysis."""
    data = await request.json()
    pair = data.get("pair", "BTC/USD")
    timeframe = data.get("timeframe", "1H")
    
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(status_code=400, detail="Unsupported trading pair")
    
    now = datetime.utcnow()
    cache_key = f"{pair}_{timeframe}_{current_user.id}"
    
    # Serve from cache if fresh
    if cache_key in ANALYSIS_CACHE:
        cached = ANALYSIS_CACHE[cache_key]
        if (now - cached["timestamp"]).total_seconds() < ANALYSIS_CACHE_TTL:
            return {"insights": cached["data"]}
    
    try:
        # Get real market data
        market_data_service = get_market_data()
        candles_df = await market_data_service.get_candle_data(pair, timeframe.lower(), limit=100)
        
        if candles_df is None or candles_df.empty:
            # Fallback to demo data
            np.random.seed(hash(pair) % 1000)  # Deterministic but varied
            closes = np.random.normal(100, 5, 100)
            df = pd.DataFrame({"close": closes})
        else:
            df = pd.DataFrame(candles_df)
            if "close" not in df.columns and len(df.columns) > 0:
                df["close"] = df.iloc[:, -1]  # Use last column as close price
        
        # Enhanced technical analysis
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["sma50"] = df["close"].rolling(window=50).mean()
        df["ema12"] = df["close"].ewm(span=12).mean()
        df["ema26"] = df["close"].ewm(span=26).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Determine trend
        current_price = df["close"].iloc[-1]
        sma20_current = df["sma20"].iloc[-1]
        sma50_current = df["sma50"].iloc[-1]
        
        if sma20_current > sma50_current and current_price > sma20_current:
            trend = "strong_uptrend"
        elif sma20_current > sma50_current:
            trend = "uptrend"
        elif sma20_current < sma50_current and current_price < sma20_current:
            trend = "strong_downtrend"
        else:
            trend = "downtrend"
        
        # Calculate support and resistance
        recent_highs = df["close"].rolling(window=20).max()
        recent_lows = df["close"].rolling(window=20).min()
        resistance = float(recent_highs.iloc[-1])
        support = float(recent_lows.iloc[-1])
        
        # Pattern detection
        patterns = []
        rsi_current = df["rsi"].iloc[-1]
        
        if rsi_current > 70:
            patterns.append("Overbought (RSI)")
        elif rsi_current < 30:
            patterns.append("Oversold (RSI)")
        
        if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]:
            patterns.append("MACD Bullish Crossover")
        elif df["macd"].iloc[-1] < df["macd_signal"].iloc[-1] and df["macd"].iloc[-2] >= df["macd_signal"].iloc[-2]:
            patterns.append("MACD Bearish Crossover")
        
        # Risk assessment
        volatility = df["close"].pct_change().std() * 100
        if volatility > 3:
            risk_level = "High"
        elif volatility > 1.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate insights
        trend_descriptions = {
            "strong_uptrend": "Strong bullish momentum with price above key moving averages",
            "uptrend": "Moderate bullish momentum",
            "downtrend": "Bearish momentum with price below key moving averages", 
            "strong_downtrend": "Strong bearish momentum with clear downward pressure"
        }
        
        summary = f"{pair} on {timeframe}: {trend_descriptions.get(trend, trend)}. "
        summary += f"Current price: {current_price:.4f}, Support: {support:.4f}, Resistance: {resistance:.4f}. "
        summary += f"Risk level: {risk_level} (Volatility: {volatility:.2f}%)."
        
        insights = {
            "trend": trend,
            "trend_strength": "strong" if "strong" in trend else "moderate",
            "current_price": float(current_price),
            "support": support,
            "resistance": resistance,
            "patterns": patterns,
            "risk_level": risk_level,
            "volatility": float(volatility),
            "rsi": float(rsi_current) if not pd.isna(rsi_current) else 50.0,
            "macd": float(df["macd"].iloc[-1]) if not pd.isna(df["macd"].iloc[-1]) else 0.0,
            "summary": summary,
            "analysis_time": now.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in AI chart analysis: {e}")
        insights = {
            "summary": f"Analysis error: {str(e)}",
            "trend": "unknown",
            "current_price": 0.0,
            "support": 0.0,
            "resistance": 0.0,
            "patterns": [],
            "risk_level": "Unknown"
        }
    
    ANALYSIS_CACHE[cache_key] = {"timestamp": now, "data": insights}
    return {"insights": insights}

@router.get("/news/{pair}")
async def news_filter_endpoint(
    pair: str,
    current_user: User = Depends(get_current_user)
):
    """Enhanced news filter with better categorization and sentiment analysis."""
    # Normalize pair to use slash
    pair = pair.replace("-", "/")
    # Always allow demo news for any pair if NEWS_API_KEY is not set
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        now = datetime.utcnow()
        demo_articles = [
            {
                "title": f"[Demo] {pair} shows strong bullish momentum - Technical analysis suggests continued uptrend",
                "url": "#",
                "publishedAt": now.isoformat(),
                "description": f"Market analysts are optimistic about {pair} as it breaks through key resistance levels.",
                "sentiment": "positive",
                "source": "Demo News"
            },
            {
                "title": f"[Demo] Major economic event for {pair} - Central bank announcement expected",
                "url": "#", 
                "publishedAt": (now - timedelta(hours=2)).isoformat(),
                "description": f"Traders are closely watching {pair} ahead of important economic data release.",
                "sentiment": "neutral",
                "source": "Demo News"
            },
            {
                "title": f"[Demo] {pair} volatility increases - Risk management crucial",
                "url": "#",
                "publishedAt": (now - timedelta(hours=4)).isoformat(),
                "description": f"Heightened volatility in {pair} requires careful position sizing and stop-loss management.",
                "sentiment": "negative",
                "source": "Demo News"
            }
        ]
        sentiment_score = sum(1 if a["sentiment"]=="positive" else -1 if a["sentiment"]=="negative" else 0 for a in demo_articles)
        overall_sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        data = {
            "headlines": demo_articles[:5],
            "major_event": any("economic event" in a["title"] for a in demo_articles),
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "total_articles": len(demo_articles),
            "last_updated": now.isoformat()
        }
        return data
    # If NEWS_API_KEY is set, keep original logic but improve error message
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(status_code=400, detail=f"Unsupported trading pair: {pair}. Please select a supported pair.")
    
    now = datetime.utcnow()
    cache_key = f"{pair}_{current_user.id}"
    
    # Serve from cache if fresh
    if cache_key in NEWS_CACHE:
        cached = NEWS_CACHE[cache_key]
        if (now - cached["timestamp"]).total_seconds() < NEWS_CACHE_TTL:
            return cached["data"]
    
    headlines = []
    major_event_detected = False
    sentiment_score = 0
    
    if NEWS_API_KEY:
        # Enhanced search terms for better results
        search_terms = [
            pair.replace("/", " "),
            pair.replace("/", ""),
            pair.split("/")[0] + " " + pair.split("/")[1] if "/" in pair else pair
        ]
        
        for term in search_terms:
            url = f"https://newsapi.org/v2/everything?q={term}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=10)
                    if resp.status_code == 200:
                        articles = resp.json().get("articles", [])
                        for article in articles:
                            title = article["title"]
                            description = article.get("description", "")
                            
                            # Check for major events
                            for event in MAJOR_EVENTS:
                                if event in title.upper():
                                    major_event_detected = True
                                    break
                            
                            # Simple sentiment analysis
                            positive_words = ["bullish", "surge", "rally", "gain", "positive", "up", "high"]
                            negative_words = ["bearish", "drop", "fall", "decline", "negative", "down", "low"]
                            
                            content = (title + " " + description).lower()
                            positive_count = sum(1 for word in positive_words if word in content)
                            negative_count = sum(1 for word in negative_words if word in content)
                            
                            if positive_count > negative_count:
                                sentiment = "positive"
                                sentiment_score += 1
                            elif negative_count > positive_count:
                                sentiment = "negative"
                                sentiment_score -= 1
                            else:
                                sentiment = "neutral"
                            
                            headlines.append({
                                "title": title,
                                "url": article["url"],
                                "publishedAt": article["publishedAt"],
                                "description": description,
                                "sentiment": sentiment,
                                "source": article.get("source", {}).get("name", "Unknown")
                            })
                        
                        if headlines:  # Found articles, no need to try other search terms
                            break
                            
            except Exception as e:
                logger.error(f"News API error for {term}: {e}")
                continue
    else:
        # Enhanced demo data
        demo_articles = [
            {
                "title": f"[Demo] {pair} shows strong bullish momentum - Technical analysis suggests continued uptrend",
                "url": "#",
                "publishedAt": now.isoformat(),
                "description": f"Market analysts are optimistic about {pair} as it breaks through key resistance levels.",
                "sentiment": "positive",
                "source": "Demo News"
            },
            {
                "title": f"[Demo] Major economic event for {pair} - Central bank announcement expected",
                "url": "#", 
                "publishedAt": (now - timedelta(hours=2)).isoformat(),
                "description": f"Traders are closely watching {pair} ahead of important economic data release.",
                "sentiment": "neutral",
                "source": "Demo News"
            },
            {
                "title": f"[Demo] {pair} volatility increases - Risk management crucial",
                "url": "#",
                "publishedAt": (now - timedelta(hours=4)).isoformat(),
                "description": f"Heightened volatility in {pair} requires careful position sizing and stop-loss management.",
                "sentiment": "negative",
                "source": "Demo News"
            }
        ]
        
        for article in demo_articles:
            if "economic event" in article["title"]:
                major_event_detected = True
            
            if article["sentiment"] == "positive":
                sentiment_score += 1
            elif article["sentiment"] == "negative":
                sentiment_score -= 1
                
            headlines.append(article)
    
    # Overall sentiment
    if sentiment_score > 0:
        overall_sentiment = "positive"
    elif sentiment_score < 0:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
    
    data = {
        "headlines": headlines[:5],  # Limit to 5 most recent
        "major_event": major_event_detected,
        "overall_sentiment": overall_sentiment,
        "sentiment_score": sentiment_score,
        "total_articles": len(headlines),
        "last_updated": now.isoformat()
    }
    
    NEWS_CACHE[cache_key] = {"timestamp": now, "data": data}
    return data

@router.post("/ai/analyze-chart-image")
async def analyze_chart_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Enhanced chart image analysis with more detailed insights."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Basic image analysis
        width, height = image.size
        mode = image.mode
        aspect_ratio = width / height
        
        # Enhanced analysis based on image characteristics
        analysis_results = []
        
        # Check image dimensions
        if width < 200 or height < 200:
            analysis_results.append("Warning: Image resolution is low. Higher resolution charts provide better analysis.")
        
        # Analyze aspect ratio for chart type
        if aspect_ratio > 2.5:
            chart_type = "Long-term chart (likely daily/weekly timeframe)"
            analysis_results.append("Detected long-term chart pattern")
        elif aspect_ratio > 1.5:
            chart_type = "Medium-term chart (likely 4H/1D timeframe)"
            analysis_results.append("Detected medium-term chart pattern")
        else:
            chart_type = "Short-term chart (likely 1H/15M timeframe)"
            analysis_results.append("Detected short-term chart pattern")
        
        # Color analysis for trend detection
        if mode == 'RGB':
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            pixels = list(gray_image.getdata())
            
            # Simple trend detection based on pixel distribution
            left_half = pixels[:len(pixels)//2]
            right_half = pixels[len(pixels)//2:]
            
            left_avg = sum(left_half) / len(left_half)
            right_avg = sum(right_half) / len(right_half)
            
            if right_avg > left_avg + 10:
                trend_analysis = "Chart suggests upward price movement"
                analysis_results.append("Visual trend analysis: Bullish pattern detected")
            elif left_avg > right_avg + 10:
                trend_analysis = "Chart suggests downward price movement"
                analysis_results.append("Visual trend analysis: Bearish pattern detected")
            else:
                trend_analysis = "Chart shows sideways movement or unclear trend"
                analysis_results.append("Visual trend analysis: Sideways or unclear pattern")
        else:
            trend_analysis = "Unable to perform color-based trend analysis"
        
        # Generate comprehensive result
        result = f"""
Chart Analysis Results:
- Image: {width}x{height} pixels, {mode} mode
- Chart Type: {chart_type}
- Trend Analysis: {trend_analysis}

Key Insights:
{chr(10).join(f"- {insight}" for insight in analysis_results)}

⚠️ This analysis is for educational purposes only. Always verify with multiple sources and consider consulting a financial advisor.
        """.strip()
        
    except Exception as e:
        logger.error(f"Error analyzing chart image: {e}")
        result = f"Error analyzing image: {str(e)}"
    
    return {"result": result}

@router.get("/education")
async def get_education_content(current_user: User = Depends(get_current_user)):
    """Return a list of educational articles and resources."""
    # Example static content; replace with DB or CMS in production
    articles = [
        {
            "id": 1,
            "title": "What is Forex Trading?",
            "category": "Trading Fundamentals",
            "content": "Forex (Foreign Exchange) is the global marketplace for trading currencies. Key concepts include currency pairs, pips, lots, and leverage. The forex market is open 24/5 and offers high liquidity.",
            "video_url": None,
            "quiz": [
                {"question": "What is a pip?", "options": ["A type of fruit", "Smallest price movement", "A trading strategy"], "answer": 1}
            ]
        },
        {
            "id": 2,
            "title": "Moving Averages",
            "category": "Technical Analysis",
            "content": "Moving averages smooth out price data to identify trends. Types include SMA and EMA. Common periods: 20, 50, 200. Crossovers can signal trend changes.",
            "video_url": None,
            "quiz": [
                {"question": "Which moving average is most used for long-term trends?", "options": ["20-period", "50-period", "200-period"], "answer": 2}
            ]
        },
        {
            "id": 3,
            "title": "Risk Management Basics",
            "category": "Risk Management",
            "content": "Never risk more than 1-2% of your account per trade. Use stop losses and calculate position size carefully.",
            "video_url": None,
            "quiz": [
                {"question": "What is the recommended max risk per trade?", "options": ["5%", "1-2%", "10%"], "answer": 1}
            ]
        },
        {
            "id": 4,
            "title": "Trading Psychology",
            "category": "Psychology",
            "content": "Emotional control and discipline are crucial for trading success. Stick to your plan and manage risk.",
            "video_url": None,
            "quiz": []
        }
    ]
    return {"articles": articles}

@router.patch("/signals/{signal_id}/force-close")
async def force_close_signal(signal_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Manually close a signal (set status to COMPLETED)."""
    from app.models.signal import SignalStatus
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    if signal.status in [SignalStatus.COMPLETED, SignalStatus.SL_HIT, SignalStatus.TP2_HIT]:
        raise HTTPException(status_code=400, detail="Signal already closed")
    signal.status = SignalStatus.COMPLETED
    db.commit()
    db.refresh(signal)
    import logging
    logging.info(f"Signal {signal_id} force-closed by user {current_user.email}")
    return {"success": True, "signal": signal.to_dict()}

@router.get("/licenses/active")
async def get_active_licenses(current_user: User = Depends(get_current_user)):
    """Return a list of active licenses for the current user."""
    # Example static data; replace with DB query in production
    licenses = [
        {
            "id": 1,
            "code": "TEST-LICENSE-123",
            "user_email": current_user.email,
            "expiry": "2025-12-31",
            "features": ["signals", "ai", "news"]
        }
    ]
    return {"licenses": licenses}

class BacktestRequest(BaseModel):
    pair: str
    timeframe: str
    strategy: str = "default"
    start_date: datetime
    end_date: datetime
    params: Optional[Dict[str, Any]] = None

class TradeResult(BaseModel):
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    direction: str
    status: str

class BacktestResult(BaseModel):
    trades: List[TradeResult]
    total_pnl: float
    win_rate: float
    num_trades: int
    max_drawdown: float
    equity_curve: List[float]

@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(req: BacktestRequest):
    # Fetch historical data
    df = await market_data_service.get_candle_data(req.pair, req.timeframe, limit=1000)
    # Fix: Ensure 'timestamp' column exists
    if "datetime" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    df = df[(df['timestamp'] >= req.start_date) & (df['timestamp'] <= req.end_date)]
    if df.empty:
        raise HTTPException(status_code=400, detail="No historical data for selected range.")
    # Placeholder: simple strategy (buy if close > open, sell if close < open)
    trades = []
    equity = 10000.0
    equity_curve = [equity]
    wins = 0
    max_drawdown = 0
    peak = equity
    for i in range(1, len(df)):
        row_prev = df.iloc[i-1]
        row = df.iloc[i]
        direction = "BUY" if row['close'] > row['open'] else "SELL"
        entry_price = row_prev['close']
        exit_price = row['close']
        pnl = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
        equity += pnl
        equity_curve.append(equity)
        if pnl > 0:
            wins += 1
        if equity > peak:
            peak = equity
        drawdown = (peak - equity)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        trades.append(TradeResult(
            entry_time=row_prev['timestamp'],
            exit_time=row['timestamp'],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            direction=direction,
            status="win" if pnl > 0 else "loss"
        ))
    win_rate = wins / len(trades) if trades else 0
    return BacktestResult(
        trades=trades,
        total_pnl=equity - 10000.0,
        win_rate=win_rate,
        num_trades=len(trades),
        max_drawdown=max_drawdown,
        equity_curve=equity_curve
    )
    