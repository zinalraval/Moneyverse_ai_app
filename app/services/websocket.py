import json
import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from app.crud.licenses_crud import license
from app.db.postgres import get_db, get_db_session
from app.services.market_data_utils import get_market_data, SUPPORTED_PAIRS
from app.models.signal import SignalType, SignalTimeframe
from app.config import settings
from app.services.licenses_service import verify_license

logger = logging.getLogger(__name__)

class WebSocketManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.license_connections: Dict[str, str] = {}
        self.connection_monitoring: Dict[str, Dict] = {}
        self.connection_heartbeats: Dict[str, datetime] = {}
        self.HEARTBEAT_INTERVAL = 30
        self.HEARTBEAT_TIMEOUT = 60

    async def connect(self, websocket: WebSocket, license_code: str) -> bool:
        try:
            # Validate license
            async with get_db_session() as db:
                license = await license.get_by_code(db, code=license_code)
                if not license or not license.is_active:
                    logger.warning(f"Invalid or inactive license: {license_code}")
                    return False
                if license.expires_at and license.expires_at < datetime.utcnow():
                    logger.warning(f"Expired license: {license_code}")
                    return False

            await websocket.accept()
            connection_id = str(id(websocket))
            self.active_connections[connection_id] = websocket
            self.license_connections[license_code] = connection_id
            self.connection_heartbeats[connection_id] = datetime.utcnow()
            
            # Start heartbeat monitoring
            asyncio.create_task(self.monitor_heartbeat(connection_id))
            
            logger.info(f"WebSocket connected: {connection_id} for license: {license_code}")
            return True
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}", exc_info=True)
            return False

    async def disconnect(self, connection_id: str):
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

    async def handle_heartbeat(self, connection_id: str):
        if connection_id in self.active_connections:
            self.connection_heartbeats[connection_id] = datetime.utcnow()
            try:
                await self.send_message(connection_id, {"type": "heartbeat"})
            except Exception as e:
                logger.error(f"Error handling heartbeat: {e}")
                await self.disconnect(connection_id)

    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await self.disconnect(connection_id)

    async def broadcast_message(self, message: Dict[str, Any], exclude: Optional[str] = None):
        for connection_id in list(self.active_connections.keys()):
            if connection_id != exclude:
                await self.send_message(connection_id, message)

    async def start_monitoring(self, connection_id: str, pair: str, timeframe: str):
        if connection_id in self.active_connections:
            self.connection_monitoring[connection_id] = {
                "pair": pair,
                "timeframe": timeframe,
                "last_update": datetime.utcnow()
            }
            
            # Send initial market data
            market_data = await get_market_data(pair)
            if market_data:
                await self.send_market_data(connection_id, market_data)
            
            logger.info(f"Started monitoring {pair} on {timeframe} for connection: {connection_id}")

    async def stop_monitoring(self, connection_id: str):
        if connection_id in self.connection_monitoring:
            del self.connection_monitoring[connection_id]
            logger.info(f"Stopped monitoring for connection: {connection_id}")

    async def send_market_data(self, connection_id: str, market_data: dict):
        await self.send_message(connection_id, {
            "type": "market_data",
            **market_data,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def send_signal_update(self, connection_id: str, signals: List[dict]):
        await self.send_message(connection_id, {
            "type": "signal_update",
            "signals": signals,
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

# Create singleton instance
websocket_manager = WebSocketManager() 