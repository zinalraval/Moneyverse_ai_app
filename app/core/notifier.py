from typing import List, Dict
import asyncio
import logging
from fastapi import WebSocket
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}  # Map license_code to WebSocket
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent broadcasts
        self.last_heartbeat: Dict[str, datetime] = {}  # Track last heartbeat per connection

    async def connect(self, websocket: WebSocket, license_code: str):
        """Connect a new WebSocket client with its license code."""
        try:
            self.connections[license_code] = websocket
            self.last_heartbeat[license_code] = datetime.utcnow()
            logger.info(f"WebSocket connected for license: {license_code}")
        except Exception as e:
            logger.error(f"Error connecting WebSocket for license {license_code}: {e}")
            raise

    def remove(self, websocket: WebSocket, license_code: str = None):
        """Remove a WebSocket connection."""
        try:
            if license_code and license_code in self.connections:
                del self.connections[license_code]
                if license_code in self.last_heartbeat:
                    del self.last_heartbeat[license_code]
                logger.info(f"WebSocket removed for license: {license_code}")
            else:
                # Find and remove by WebSocket instance
                for code, ws in list(self.connections.items()):
                    if ws == websocket:
                        del self.connections[code]
                        if code in self.last_heartbeat:
                            del self.last_heartbeat[code]
                        logger.info(f"WebSocket removed for license: {code}")
                        break
        except Exception as e:
            logger.error(f"Error removing WebSocket: {e}")

    async def send_signal_update(self, message: str):
        """Send a signal update to all connected clients."""
        if not self.connections:
            logger.debug("[Notifier] No active WebSocket connections. Skipping broadcast.")
            return
        disconnected_connections = []
        logger.debug("[Notifier] Entering send_signal_update")
        async with self.semaphore:
            for license_code, connection in list(self.connections.items()):
                try:
                    logger.debug(f"[Notifier] About to send message to {license_code}")
                    await connection.send_text(message)
                    logger.debug(f"[Notifier] Signal update sent to license: {license_code}")
                except Exception as e:
                    logger.error(f"Could not send message to websocket for license {license_code}: {e}")
                    disconnected_connections.append((connection, license_code))

        # Remove disconnected connections
        for connection, license_code in disconnected_connections:
            self.remove(connection, license_code)
        logger.debug("[Notifier] Exiting send_signal_update")

    async def send_signal_notification(self, signal):
        """Send a signal notification to all connected clients."""
        try:
            logger.debug("[Notifier] Entering send_signal_notification")
            # Convert signal to JSON string
            signal_data = signal.to_dict() if hasattr(signal, 'to_dict') else signal
            message = json.dumps({
                "type": "signal_notification",
                "signal": signal_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.debug("[Notifier] About to call send_signal_update")
            await self.send_signal_update(message)
            logger.info(f"Signal notification sent for signal {signal.get('id', 'unknown') if isinstance(signal, dict) else getattr(signal, 'id', 'unknown')}")
            logger.debug("[Notifier] Exiting send_signal_notification")
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")

    async def send_heartbeat(self, license_code: str):
        """Send a heartbeat to a specific client."""
        if license_code in self.connections:
            try:
                await self.connections[license_code].send_text(
                    '{"type": "heartbeat", "timestamp": "' + datetime.utcnow().isoformat() + '"}'
                )
                self.last_heartbeat[license_code] = datetime.utcnow()
                logger.debug(f"Heartbeat sent to license: {license_code}")
            except Exception as e:
                logger.error(f"Error sending heartbeat to license {license_code}: {e}")
                self.remove(self.connections[license_code], license_code)

    def get_connection_count(self) -> int:
        """Get the current number of active connections."""
        return len(self.connections)

notifier = Notifier() 