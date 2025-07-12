from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException, status
from typing import List, Optional, Dict, Any
from app.services.websocket_service import manager as websocket_manager
from app.services.shared import SUPPORTED_PAIRS
from app.models.signal import SignalStatus, SignalType, SignalTimeframe, SignalSetup
from app.models.user import User
from app.api.dependencies import get_current_user, get_db
from app.services.licenses_service import verify_license
from app.db.postgres import get_db_session, get_db_session_factory
import logging
from datetime import datetime
from sqlalchemy.orm import Session
import json
from pydantic import BaseModel, ValidationError
import asyncio
from app.config import settings
from app.services.signal_service import SignalService
from app.models.license import License
# from app.core.security import get_current_user_ws

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize signal service
signal_service = SignalService()

class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""
    type: str
    data: Dict[str, Any]

class SignalSubscriptionMessage(WebSocketMessage):
    """Message model for signal subscriptions."""
    type: str = "subscribe_signal"
    data: Dict[str, Any] = {
        "signal_id": int,
        "pairs": List[str]
    }

class PriceSubscriptionMessage(WebSocketMessage):
    """Message model for price subscriptions."""
    type: str = "subscribe_price"
    data: Dict[str, Any] = {
        "pairs": List[str]
    }

class SetupSubscriptionMessage(WebSocketMessage):
    """Message model for setup condition subscriptions."""
    type: str = "subscribe_setup"
    data: Dict[str, Any] = {
        "setup": str  # SignalSetup enum value
    }

async def handle_websocket_message(websocket: WebSocket, message: str, connection_id: str) -> None:
    """Handle incoming WebSocket messages."""
    try:
        # Parse message
        data = json.loads(message)
        msg = WebSocketMessage(**data)
        
        # Handle different message types
        if msg.type == "heartbeat":
            await websocket_manager.handle_heartbeat(connection_id)
            
        elif msg.type == "monitor":
            pair = msg.data.get("pair")
            timeframe = msg.data.get("timeframe")
            if pair and timeframe:
                await websocket_manager.start_monitoring(connection_id, pair, timeframe)
                
        elif msg.type == "stop_monitor":
            await websocket_manager.stop_monitoring(connection_id)
            
        elif msg.type == "subscribe_signal":
            signal_msg = SignalSubscriptionMessage(**data)
            signal_id = signal_msg.data.get("signal_id")
            if signal_id:
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "data": {"signal_id": signal_id}
                })
                
        elif msg.type == "subscribe_price":
            price_msg = PriceSubscriptionMessage(**data)
            pairs = price_msg.data.get("pairs", [])
            if pairs:
                # Validate pairs
                if not all(pair in SUPPORTED_PAIRS for pair in pairs):
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Invalid trading pair"}
                    })
                    return
                
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "data": {"pairs": pairs}
                })
                
        elif msg.type == "subscribe_setup":
            setup_msg = SetupSubscriptionMessage(**data)
            setup = setup_msg.data.get("setup")
            if setup:
                try:
                    # Convert string to SignalSetup enum
                    setup_enum = SignalSetup[setup]
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "data": {"setup": setup}
                    })
                except KeyError:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Invalid setup condition: {setup}"}
                    })
                
        else:
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Unknown message type: {msg.type}"}
            })
            
    except ValidationError as e:
        await websocket.send_json({
            "type": "error",
            "data": {"message": f"Invalid message format: {str(e)}"}
        })
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Invalid JSON format"}
        })
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Internal server error"}
        })

@router.websocket("/ws/signals")
async def websocket_endpoint(
    websocket: WebSocket,
    license_code: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time signal updates."""
    if license_code:
        license_code = license_code.strip()
    logger.debug(f"[WebSocket] Received connection request. License code: {license_code}")
    try:
        # Validate license if provided
        if license_code:
            logger.debug(f"[WebSocket] Verifying license: {license_code}")
            is_valid = await verify_license(license_code, db)
            if not is_valid:
                logger.warning(f"[WebSocket] Invalid license code: {license_code}. Closing connection.")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            logger.debug(f"[WebSocket] License {license_code} is valid.")
        else:
            logger.debug("[WebSocket] No license code provided.")
        
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Connect to WebSocket manager
        connection_id = str(id(websocket))
        logger.debug(f"[WebSocket] Attempting to connect websocket_manager for connection_id: {connection_id}")
        if not await websocket_manager.connect(websocket, license_code or "anonymous"):
            logger.warning(f"[WebSocket] Failed to connect to websocket_manager for connection_id: {connection_id}. Closing connection.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        logger.debug(f"[WebSocket] Successfully connected to websocket_manager for connection_id: {connection_id}")
        
        try:
            while True:
                # Receive and process messages
                data = await websocket.receive_text()
                print(f"RAW WS MESSAGE: {data}")
                await handle_websocket_message(websocket, data, connection_id)
                
        except WebSocketDisconnect:
            logger.info(f"[WebSocket] Disconnected: {connection_id}")
        finally:
            # Clean up
            logger.debug(f"[WebSocket] Cleaning up connection: {connection_id}")
            await websocket_manager.disconnect(connection_id)
            await websocket_manager.stop_monitoring(connection_id)
            logger.debug(f"[WebSocket] Clean up complete for connection: {connection_id}")
            
    except Exception as e:
        logger.error(f"[WebSocket] Fatal error in websocket_endpoint for connection_id {connection_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_e:
            logger.error(f"[WebSocket] Error during WebSocket close after fatal error: {close_e}")

@router.websocket("/ws/user")
async def user_websocket_endpoint(
    websocket: WebSocket,
    current_user: User = Depends(get_current_user),
    pairs: Optional[List[str]] = Query(None)
):
    """WebSocket endpoint for authenticated user updates."""
    try:
        connection_id = str(id(websocket))
        if not await websocket_manager.connect(websocket, f"user_{current_user.id}"):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        try:
            while True:
                data = await websocket.receive_text()
                print(f"RAW WS MESSAGE: {data}")
                await handle_websocket_message(websocket, data, connection_id)
                
        except WebSocketDisconnect:
            logger.info(f"User WebSocket disconnected: {connection_id}")
        finally:
            await websocket_manager.disconnect(connection_id)
            await websocket_manager.stop_monitoring(connection_id)
            
    except Exception as e:
        logger.error(f"User WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass

@router.websocket("/ws/prices")
async def price_websocket_endpoint(
    websocket: WebSocket,
    pairs: List[str] = Query(...),
    license_code: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time price updates."""
    logger.info(f"[WS] Incoming connection: path={websocket.url.path}, query={websocket.url.query}, license_code={license_code}")
    try:
        # Verify license if provided
        if license_code:
            logger.info(f"WebSocket connection attempt with license code: {license_code}")
            is_license_valid = await verify_license(license_code, db)
            logger.info(f"License verification result for {license_code}: {is_license_valid}")
            if not is_license_valid:
                await websocket.close(code=1008, reason="Invalid license")
                return

        # Validate pairs
        if not all(pair in SUPPORTED_PAIRS for pair in pairs):
            await websocket.close(code=1008, reason="Invalid trading pair")
            return

        # Accept connection
        await websocket.accept()
        logger.info(f"New WebSocket connection established for price updates: {pairs}")
        
        # Connect to manager with pairs
        client_id = await websocket_manager.connect(websocket, pairs)
        
        try:
            while True:
                # Handle incoming messages
                data = await websocket.receive_text()
                print(f"RAW WS MESSAGE: {data}")
                await handle_websocket_message(websocket, data, client_id)
                
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed normally")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            await websocket_manager.disconnect(client_id)
            
    except Exception as e:
        logger.error(f"Error in price websocket endpoint: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@router.websocket("/ws/setups")
async def setup_websocket_endpoint(
    websocket: WebSocket,
    setups: List[str] = Query(...),
    license_code: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time setup condition updates."""
    logger.info(f"[WS] Incoming connection: path={websocket.url.path}, query={websocket.url.query}, license_code={license_code}")
    try:
        # Verify license if provided
        if license_code:
            logger.info(f"WebSocket connection attempt with license code: {license_code}")
            is_license_valid = await verify_license(license_code, db)
            logger.info(f"License verification result for {license_code}: {is_license_valid}")
            if not is_license_valid:
                await websocket.close(code=1008, reason="Invalid license")
                return

        # Validate setups
        try:
            setup_enums = [SignalSetup[setup] for setup in setups]
        except KeyError as e:
            await websocket.close(code=1008, reason=f"Invalid setup condition: {str(e)}")
            return

        # Accept connection
        await websocket.accept()
        logger.info(f"New WebSocket connection established for setup updates: {setups}")
        
        # Connect to manager
        client_id = await websocket_manager.connect(websocket)
        
        # Subscribe to setups
        for setup in setup_enums:
            await websocket_manager.subscribe_to_setup(client_id, setup)
        
        try:
            while True:
                # Handle incoming messages
                data = await websocket.receive_text()
                print(f"RAW WS MESSAGE: {data}")
                await handle_websocket_message(websocket, data, client_id)
                
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed normally")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            await websocket_manager.disconnect(client_id)
            
    except Exception as e:
        logger.error(f"Error in setup websocket endpoint: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@router.get("/ws/stats")
async def get_websocket_stats(current_user: User = Depends(get_current_user)):
    """Get WebSocket connection statistics."""
    return {
        "active_connections": len(websocket_manager.active_connections),
        "monitored_pairs": len(websocket_manager.connection_monitoring),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/ws/health")
async def get_websocket_health():
    """Get WebSocket service health status."""
    return {
        "status": "healthy",
        "active_connections": len(websocket_manager.active_connections),
        "timestamp": datetime.utcnow().isoformat()
    } 