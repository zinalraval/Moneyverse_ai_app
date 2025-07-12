import json
import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from app.crud.licenses_crud import license as license_crud
from app.db.postgres import get_db
from app.services.market_data import market_data_service
from app.services.signal_generation import generate_signal
from app.models.signal import SignalType, SignalTimeframe
from app.services.licenses_service import verify_license

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.license_connections: Dict[str, str] = {}  # license_code -> connection_id
        self.connection_monitoring: Dict[str, Dict] = {}  # connection_id -> {pair, timeframe}
        self.connection_heartbeats: Dict[str, datetime] = {}  # connection_id -> last_heartbeat
        self.HEARTBEAT_INTERVAL = 30  # seconds
        self.HEARTBEAT_TIMEOUT = 60  # seconds

    async def connect(self, websocket: WebSocket, license_code: str) -> bool:
        """Connect a new WebSocket client"""
        try:
            # Validate license
            async for db in get_db():
                license_obj = await license_crud.get_by_code(db, code=license_code)
                if not license_obj or not license_obj.is_active:
                    logger.warning(f"Invalid or inactive license: {license_code}")
                    return False
                if license_obj.expires_at and license_obj.expires_at < datetime.utcnow():
                    logger.warning(f"Expired license: {license_code}")
                    return False

            connection_id = id(websocket)
            self.active_connections[connection_id] = websocket
            self.license_connections[license_code] = connection_id
            self.connection_heartbeats[connection_id] = datetime.utcnow()
            
            # Start heartbeat monitoring
            asyncio.create_task(self.monitor_heartbeat(connection_id))
            
            logger.info(f"WebSocket connected: {connection_id} for license: {license_code}")
            return True
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            return False

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            
            # Clean up connection data
            del self.active_connections[connection_id]
            if connection_id in self.connection_heartbeats:
                del self.connection_heartbeats[connection_id]
            
            # Remove from license connections
            for license_code, conn_id in list(self.license_connections.items()):
                if conn_id == connection_id:
                    del self.license_connections[license_code]
                    break
            
            # Remove from monitoring
            if connection_id in self.connection_monitoring:
                del self.connection_monitoring[connection_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")

    async def monitor_heartbeat(self, connection_id: str):
        """Monitor connection heartbeat"""
        while connection_id in self.active_connections:
            try:
                last_heartbeat = self.connection_heartbeats.get(connection_id)
                if last_heartbeat and (datetime.utcnow() - last_heartbeat).total_seconds() > self.HEARTBEAT_TIMEOUT:
                    logger.warning(f"Heartbeat timeout for connection: {connection_id}")
                    await self.disconnect(connection_id)
                    break
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring: {e}")
                break

    async def send_heartbeat(self, connection_id: str):
        """Send heartbeat to client"""
        if connection_id in self.active_connections:
            try:
                await self.send_message(connection_id, {"type": "heartbeat"})
                self.connection_heartbeats[connection_id] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await self.disconnect(connection_id)

    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await self.disconnect(connection_id)

    async def broadcast_message(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all connected clients"""
        for connection_id in list(self.active_connections.keys()):
            if connection_id != exclude:
                await self.send_message(connection_id, message)

    async def start_monitoring(self, connection_id: str, pair: str, timeframe: str):
        """Start monitoring a trading pair for a connection"""
        if connection_id in self.active_connections:
            self.connection_monitoring[connection_id] = {
                "pair": pair,
                "timeframe": timeframe,
                "last_update": datetime.utcnow()
            }
            
            # Send initial market data
            live_price = await market_data_service.get_live_price(pair)
            if live_price:
                await self.send_price_update(connection_id, pair, live_price)
            
            # Send initial trend analysis
            trend_data = await self.analyze_trend(pair, timeframe)
            if trend_data:
                await self.send_trend_update(connection_id, trend_data)
            
            logger.info(f"Started monitoring {pair} on {timeframe} for connection: {connection_id}")

            # --- NEW: Send all active signals for this pair/timeframe ---
            from app.models.signal import Signal, SignalStatus
            from app.db.postgres import get_db_session
            async with get_db_session() as db:
                result = await db.execute(
                    Signal.__table__.select().where(
                        (Signal.pair == pair) &
                        (Signal.timeframe == timeframe) &
                        (Signal.status == SignalStatus.ACTIVE)
                    )
                )
                signals = result.fetchall()
                signals_list = [dict(row._mapping) for row in signals]
                await self.send_signal_update(connection_id, signals_list)
                logger.info(f"Sent {len(signals_list)} active signals to connection {connection_id} for {pair} {timeframe}")

    async def stop_monitoring(self, connection_id: str):
        """Stop monitoring for a connection"""
        if connection_id in self.connection_monitoring:
            del self.connection_monitoring[connection_id]
            logger.info(f"Stopped monitoring for connection: {connection_id}")

    async def send_signal_update(self, connection_id: str, signals: List[dict]):
        """Send signal update to a specific connection"""
        await self.send_message(connection_id, {
            "type": "signal_update",
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def send_price_update(self, connection_id: str, pair: str, price: float):
        """Send price update to a specific connection"""
        await self.send_message(connection_id, {
            "type": "price_update",
            "pair": pair,
            "price": price,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def send_market_data(self, connection_id: str, market_data: dict):
        """Send market data to a specific connection"""
        await self.send_message(connection_id, {
            "type": "market_data",
            **market_data,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def send_trend_update(self, connection_id: str, trend_data: dict):
        """Send trend update to a specific connection"""
        await self.send_message(connection_id, {
            "type": "trend_update",
            **trend_data,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def broadcast_signal_update(self, signal: dict):
        """Broadcast signal update to all connected clients"""
        message = {
            "type": "signal_update",
            "signal": signal,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_message(message)

    async def analyze_trend(self, pair: str, timeframe: str) -> Optional[dict]:
        """Analyze trend for a trading pair"""
        try:
            # Get candle data for analysis
            candles_df = await market_data_service.get_candle_data(pair, timeframe, limit=100)
            if candles_df.empty:
                return None

            # Calculate EMAs
            prices = candles_df['close'].tolist()
            ema_20 = sum(prices[-20:]) / min(20, len(prices))
            ema_50 = sum(prices[-50:]) / min(50, len(prices))
            
            # Determine trend
            current_price = prices[-1]
            if current_price > ema_20 and ema_20 > ema_50:
                trend = "uptrend"
            elif current_price < ema_20 and ema_20 < ema_50:
                trend = "downtrend"
            else:
                trend = "sideways"

            return {
                "pair": pair,
                "timeframe": timeframe,
                "trend": trend,
                "current_price": current_price,
                "ema_20": ema_20,
                "ema_50": ema_50
            }
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return None

    async def handle_message(self, connection_id: str, message: dict):
        """Handle incoming WebSocket messages"""
        logger.info(f"[WebSocket] Received message from connection {connection_id}: {message}")
        try:
            if message["type"] == "start_monitoring":
                logger.info(f"Received start_monitoring for connection {connection_id}: pair={message['pair']}, timeframe={message['timeframe']}")
                await self.start_monitoring(
                    connection_id,
                    message["pair"],
                    message["timeframe"]
                )
            elif message["type"] == "stop_monitoring":
                await self.stop_monitoring(connection_id)
            elif message["type"] == "heartbeat":
                self.connection_heartbeats[connection_id] = datetime.utcnow()
            else:
                logger.warning(f"Unknown message type: {message['type']}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

# Global WebSocket manager instance
manager = WebSocketManager() 