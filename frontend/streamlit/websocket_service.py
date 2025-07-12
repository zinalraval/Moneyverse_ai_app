import asyncio
import json
import websockets 
from typing import Dict, Optional
import streamlit as st
import threading
from queue import Queue, Empty
import logging
import os

logger = logging.getLogger(__name__)

# It's better to have the WebSocketService managed within Streamlit's session state
# so we don't need a global variable.

class WebSocketService:
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.message_queue = Queue()
        self.listener_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.uri = None
        self.license_code = None
        self.loop = None

    def connect(self, uri: str, license_code: str):
        with self._lock:
            if self.is_connected or (self.listener_thread and self.listener_thread.is_alive()):
                logger.info("WebSocket already connected or connection in progress.")
                return
            
            self.uri = uri
            self.license_code = license_code
            
            self.stop_event.clear()
            self.listener_thread = threading.Thread(target=self._run_listener, name="WebSocketListener")
            self.listener_thread.daemon = True
            self.listener_thread.start()

    def _run_listener(self):
        """The main loop for the listener thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._listen_for_messages())
        finally:
            self.loop.close()

    async def _listen_for_messages(self):
        """Connects and listens for messages, putting them in the queue."""
        full_uri = f"{self.uri}?license_code={self.license_code}"
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(full_uri) as ws:
                    self.ws = ws
                    self.is_connected = True
                    logger.info(f"WebSocket connected to {full_uri}")
                    
                    while not self.stop_event.is_set():
                        try:
                            message_str = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                            message = json.loads(message_str)
                            self.message_queue.put(message)
                            logger.debug(f"Received message: {message}")
                        except asyncio.TimeoutError:
                            continue
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode message: {message_str}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Connection closed during receive.")
                            break # Break inner loop to reconnect

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}. Retrying in 5 seconds...")
            
            finally:
                self.is_connected = False
                self.ws = None

            if not self.stop_event.is_set():
                await asyncio.sleep(5)

    def send_message(self, message: dict):
        if not self.is_connected or not self.ws or not self.loop:
            logger.warning("Cannot send message, WebSocket is not connected or loop is not running.")
            return

        async def _send():
            try:
                await self.ws.send(json.dumps(message))
                logger.debug(f"Sent message: {message}")
            except websockets.exceptions.ConnectionClosed:
                logger.error("Failed to send message: connection closed.")
        
        if self.listener_thread and self.listener_thread.is_alive():
             future = asyncio.run_coroutine_threadsafe(_send(), self.loop)
             try:
                 future.result(timeout=2)
             except Exception as e:
                 logger.error(f"Error sending message via threadsafe call: {e}")


    def disconnect(self):
        with self._lock:
            if not self.listener_thread:
                return

            self.stop_event.set()
            if self.listener_thread.is_alive():
                self.listener_thread.join(timeout=5)
            
            self.is_connected = False
            self.ws = None
            self.listener_thread = None
            self.loop = None
            logger.info("Disconnected from WebSocket server")

    def get_message(self) -> Optional[Dict]:
        try:
            return self.message_queue.get_nowait()
        except Empty:
            return None

def get_websocket_service() -> WebSocketService:
    """
    Get the WebSocketService instance from Streamlit's session state,
    creating it if it doesn't exist.
    """
    if 'ws_service' not in st.session_state:
        st.session_state.ws_service = WebSocketService()
    return st.session_state.ws_service 