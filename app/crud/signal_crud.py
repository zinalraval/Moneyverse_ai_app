from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_
from app.models.signal import Signal, SignalStatus, SignalType, SignalSetup
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

async def get_active_signal_for_pair(db: Session, pair: str) -> Optional[Signal]:
    """
    Retrieves the active signal for a given trading pair.
    
    Args:
        db: Database session
        pair: Trading pair (e.g., "BTC/USD")
        
    Returns:
        Optional[Signal]: Active signal if found, None otherwise
    """
    result = await db.execute(
        select(Signal).where(
            and_(
                Signal.pair == pair,
                Signal.status == SignalStatus.ACTIVE
            )
        )
    )
    return result.scalars().first()

async def create_new_signal(db: Session, signal_data: Dict[str, Any]) -> Signal:
    """
    Creates a new signal in the database.
    Args:
        db: Database session
        signal_data: Signal data dictionary or Signal object
    Returns:
        Signal: Created signal
    """
    try:
        if isinstance(signal_data, dict):
            # Ensure setup_conditions is a list
            if 'setup_conditions' in signal_data and not isinstance(signal_data['setup_conditions'], list):
                signal_data['setup_conditions'] = [signal_data['setup_conditions']]
            new_signal = Signal(**signal_data)
            db.add(new_signal)
            await db.commit()
            await db.refresh(new_signal)
            logger.info(f"Created new signal: {new_signal.id} for {new_signal.pair}")
            return new_signal
        else:
            # Assume it's already a Signal object
            db.add(signal_data)
            await db.commit()
            await db.refresh(signal_data)
            logger.info(f"Created signal (ORM): {signal_data.id}")
            return signal_data
    except Exception as e:
        logger.error(f"Error creating signal: {str(e)}")
        await db.rollback()
        raise

async def get_signals_by_conditions(
    db: Session,
    pair: Optional[str] = None,
    signal_type: Optional[SignalType] = None,
    status: Optional[SignalStatus] = None,
    setup_conditions: Optional[List[SignalSetup]] = None,
    limit: int = 10
) -> List[Signal]:
    """
    Get signals filtered by various conditions.
    
    Args:
        db: Database session
        pair: Trading pair filter
        signal_type: Signal type filter
        status: Signal status filter
        setup_conditions: List of setup conditions to filter by
        limit: Maximum number of signals to return
        
    Returns:
        List[Signal]: List of matching signals
    """
    query = select(Signal)
    conditions = []
    
    if pair:
        conditions.append(Signal.pair == pair)
    if signal_type:
        conditions.append(Signal.type == signal_type)
    if status:
        conditions.append(Signal.status == status)
    if setup_conditions:
        # Filter signals that have all specified setup conditions
        for condition in setup_conditions:
            conditions.append(Signal.setup_conditions.contains([condition]))
            
    if conditions:
        query = query.where(and_(*conditions))
        
    query = query.order_by(Signal.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

async def update_signal_status(
    db: Session,
    signal_id: int,
    status: SignalStatus,
    exit_price: Optional[float] = None
) -> Optional[Signal]:
    """
    Update signal status and exit price.
    
    Args:
        db: Database session
        signal_id: Signal ID
        status: New status
        exit_price: Optional exit price
        
    Returns:
        Optional[Signal]: Updated signal if found, None otherwise
    """
    try:
        result = await db.execute(select(Signal).where(Signal.id == signal_id))
        signal = result.scalars().first()
        
        if not signal:
            return None
            
        signal.status = status
        if exit_price is not None:
            signal.exit_price = exit_price
            signal.exit_time = datetime.utcnow()
            # Calculate PnL
            if signal.direction == "BUY":
                signal.pnl = (exit_price - signal.entry_price) / signal.entry_price * 100
            else:
                signal.pnl = (signal.entry_price - exit_price) / signal.entry_price * 100
                
        await db.commit()
        await db.refresh(signal)
        logger.info(f"Updated signal {signal_id} status to {status}")
        return signal
    except Exception as e:
        logger.error(f"Error updating signal {signal_id}: {str(e)}")
        await db.rollback()
        raise

async def get_expired_signals(db: Session, max_age: timedelta) -> List[Signal]:
    """
    Get signals that have expired (not updated within max_age).
    
    Args:
        db: Database session
        max_age: Maximum age for active signals
        
    Returns:
        List[Signal]: List of expired signals
    """
    expiry_time = datetime.utcnow() - max_age
    result = await db.execute(
        select(Signal).where(
            and_(
                Signal.status == SignalStatus.ACTIVE,
                Signal.last_updated < expiry_time
            )
        )
    )
    return result.scalars().all() 