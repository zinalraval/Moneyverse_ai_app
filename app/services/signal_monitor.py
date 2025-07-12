from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from app.models.signal import Signal, SignalStatus, SignalType, SignalDirection
import logging
from datetime import datetime
import asyncio
from typing import Dict, Set, List, Optional, Any
from app.core.notifier import notifier
from app.services.websocket import websocket_manager
from datetime import timedelta
from app.db.postgres import get_db_session_factory
from app.crud.signal_crud import get_signals_by_conditions, update_signal_status
from app.schemas.signal_schema import SignalUpdate
from app.config import settings
from fastapi import HTTPException
from app.core.callbacks import callback_manager

logger = logging.getLogger(__name__)

# Store active signal IDs to prevent duplicate processing
active_signal_ids: Set[int] = set()
active_monitor_tasks: Dict[int, asyncio.Task] = {} # To keep track of active monitoring tasks

class SignalMonitor:
    def __init__(self):
        self._active_signals_by_pair: Dict[str, Dict[int, Signal]] = {}
        self._monitoring_task = None
        self._db_session_factory = get_db_session_factory()

    async def start(self):
        """Load active signals and register for price updates."""
        if self._monitoring_task:
            return

        # Register the price update handler
        callback_manager.register('price_update', self.on_price_update)
        
        async with self._db_session_factory() as db:
            try:
                result = await db.execute(
                    select(Signal).where(Signal.status.in_([SignalStatus.ACTIVE, SignalStatus.TP1_HIT]))
                )
                active_signals = result.scalars().all()
                for signal in active_signals:
                    self.add_signal(signal)
                logger.info(f"Loaded {len(active_signals)} active signals for monitoring.")
            except Exception as e:
                logger.error(f"Failed to load active signals: {e}")

        self._monitoring_task = asyncio.create_task(self._dummy_task())

    async def _dummy_task(self):
        """A dummy task to keep the monitor alive. Price checks are now event-driven."""
        while True:
            await asyncio.sleep(3600) # Sleep for an hour, actual checks are event-driven

    def add_signal(self, signal: Signal):
        """Add a signal to the monitoring dictionary."""
        if signal.pair not in self._active_signals_by_pair:
            self._active_signals_by_pair[signal.pair] = {}
        self._active_signals_by_pair[signal.pair][signal.id] = signal
        logger.info(f"Added signal {signal.id} for pair {signal.pair} to monitor.")

    def remove_signal(self, signal_id: int, pair: str):
        """Remove a signal from monitoring."""
        if pair in self._active_signals_by_pair and signal_id in self._active_signals_by_pair[pair]:
            del self._active_signals_by_pair[pair][signal_id]
            if not self._active_signals_by_pair[pair]:
                del self._active_signals_by_pair[pair]
            logger.info(f"Removed signal {signal_id} for pair {pair} from monitor.")

    async def on_price_update(self, pair: str, price: float):
        """This is the entry point for real-time price updates."""
        if pair not in self._active_signals_by_pair:
            return

        for signal_id, signal in list(self._active_signals_by_pair[pair].items()):
            # Using a copy to avoid issues with modification during iteration
            await self._check_signal(signal, price)
            
    async def _check_signal(self, signal: Signal, current_price: float):
        """Check a single signal against the current price."""
        try:
                    # Check for TP1 hit
            if signal.status == SignalStatus.ACTIVE and (
                (signal.direction == "BUY" and current_price >= signal.tp1) or
                (signal.direction == "SELL" and current_price <= signal.tp1)
            ):
                            await self._handle_tp1_hit(signal)

                    # Check for TP2 hit
            elif signal.status == SignalStatus.TP1_HIT and (
                (signal.direction == "BUY" and current_price >= signal.tp2) or
                (signal.direction == "SELL" and current_price <= signal.tp2)
            ):
                            await self._handle_tp2_hit(signal)
                    
            # Check for SL hit (for both ACTIVE and TP1_HIT signals)
            if signal.status in [SignalStatus.ACTIVE, SignalStatus.TP1_HIT] and (
                (signal.direction == "BUY" and current_price <= signal.stop_loss) or
                (signal.direction == "SELL" and current_price >= signal.stop_loss)
            ):
                            await self._handle_sl_hit(signal)
                
        except Exception as e:
            logger.error(f"Error checking signal {signal.id}: {e}")

    async def _update_db_and_notify(self, signal: Signal, updates: Dict):
        """Update signal in the database and notify clients."""
        async with self._db_session_factory() as db:
            try:
                # Get the signal from the database to ensure it's fresh
                result = await db.execute(
                    select(Signal).where(Signal.id == signal.id)
                )
                db_signal = result.scalar_one_or_none()
                
                if not db_signal:
                    logger.error(f"Signal {signal.id} not found in database")
                    return
                
                # Apply updates
                for key, value in updates.items():
                    setattr(db_signal, key, value)
                
                db_signal.last_updated = datetime.utcnow()
                db.add(db_signal)
                await db.commit()
                
                # After committing, the signal object is updated
                await notifier.send_signal_update(db_signal.to_dict())
                logger.info(f"Signal {db_signal.id} updated to {db_signal.status.value}")
                
            except Exception as e:
                logger.error(f"Failed to update signal {signal.id} in DB: {e}")
                await db.rollback()

    async def _handle_tp1_hit(self, signal: Signal):
        """Handle TP1 hit event."""
        updates = {
            "status": SignalStatus.TP1_HIT,
            "label": "Move SL to Breakeven",
            "sl": signal.entry_price
        }
        await self._update_db_and_notify(signal, updates)

    async def _handle_tp2_hit(self, signal: Signal):
        """Handle TP2 hit event."""
        updates = {
            "status": SignalStatus.COMPLETED,
            "label": "Trade Completed. Wait for Next Signal."
        }
        await self._update_db_and_notify(signal, updates)
        
        # Release memory lock to allow new signals
        from app.services.signal_generation import release_memory_lock
        await release_memory_lock(self._db_session_factory(), signal.pair, signal.type)
        
        self.remove_signal(signal.id, signal.pair)

    async def _handle_sl_hit(self, signal: Signal):
        """Handle SL hit event."""
        updates = {
            "status": SignalStatus.CANCELLED,
            "label": "Trade Completed. Wait for Next Signal."
        }
        await self._update_db_and_notify(signal, updates)
        
        # Release memory lock to allow new signals
        from app.services.signal_generation import release_memory_lock
        await release_memory_lock(self._db_session_factory(), signal.pair, signal.type)
        
        self.remove_signal(signal.id, signal.pair)

# Create singleton instance
signal_monitor = SignalMonitor()

async def start_signal_monitor():
    await signal_monitor.start()

async def stop_signal_monitor():
    await signal_monitor.stop()

async def get_active_signals(db: Session) -> List[Dict[str, Any]]:
    """Get all active trading signals"""
    try:
        query = select(Signal).where(
            Signal.status == SignalStatus.ACTIVE
        )
        result = await db.execute(query)
        signals = result.scalars().all()
        return [
            {
                "id": signal.id,
                "pair": signal.pair,
                "signal_type": signal.type.value if hasattr(signal.type, 'value') else signal.type,
                "direction": signal.direction.value if hasattr(signal.direction, 'value') else signal.direction,
                "entry": signal.entry_price,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "stop_loss": signal.stop_loss,
                "confidence": float(signal.confidence),
                "status": signal.status.value if hasattr(signal.status, 'value') else signal.status,
                "generated_time": int(signal.generated_time.timestamp()) if signal.generated_time else None,
                "created_at": signal.created_at.isoformat() if signal.created_at else None,
                "last_updated": signal.last_updated.isoformat() if signal.last_updated else None,
                "reason": signal.reason,
                "label": signal.label,
                "logic_note": signal.logic_note,
                # Add more fields as needed
            }
            for signal in signals
        ]
    except Exception as e:
        logger.error(f"Error fetching active signals: {e}")
        raise

# This service checks the status of active trading signals and updates them based on current market prices.
# It fetches the current price for each signal's trading pair and updates the signal status
# accordingly if the price hits the take profit (TP) or stop loss (SL) levels.
# It assumes that the Signal model has fields for direction, TP1, TP2, SL, status, and last_updated.
# Ensure you have the necessary fields in your Signal model:
# - direction: "Buy" or "Sell"
# - tp1: Take Profit level 1
# - tp2: Take Profit level 2
# - sl: Stop Loss level
# - status: Current status of the signal (e.g., "Active", "TP1 Hit", "TP2 Hit", "SL Hit")
# - last_updated: Timestamp of the last update (optional, but useful for tracking)
# This service should be called periodically (e.g., via a background task or cron job)                              
# to ensure that the signal statuses are kept up-to-date with the latest market conditions.
# Ensure you have the necessary imports and configurations in your main application file
# to run this service periodically, such as using a task scheduler or background worker.
# Ensure you have the necessary imports and configurations in your main application file
# to run this service periodically, such as using a task scheduler or background worker.
# Ensure you have the necessary imports and configurations in your main application file