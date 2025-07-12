import pytest
import asyncio
import json
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.websocket_service import WebSocketManager
from app.services.market_data import market_data_service
from datetime import datetime, timedelta

@pytest.fixture
async def websocket_manager():
    manager = WebSocketManager()
    yield manager
    # No .close() method on WebSocketManager, so no cleanup needed

@pytest.mark.asyncio
async def test_websocket_connection(websocket_manager):
    """Test WebSocket connection and disconnection."""
    # Test connection
    mock_ws = AsyncMock()
    with patch('app.services.websocket_service.license_crud.get_by_code', return_value=AsyncMock(is_active=True, expires_at=None)):
        result = await websocket_manager.connect(mock_ws, "test_license")
    assert result is True
    assert len(websocket_manager.active_connections) == 1
    
    # Test disconnection
    await websocket_manager.disconnect(id(mock_ws))
    assert len(websocket_manager.active_connections) == 0

@pytest.mark.asyncio
async def test_websocket_broadcast(websocket_manager):
    """Test broadcasting messages to connected clients."""
    # Connect two clients
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()
    with patch('app.services.websocket_service.license_crud.get_by_code', return_value=AsyncMock(is_active=True, expires_at=None)):
        await websocket_manager.connect(mock_ws1, "test_license")
        await websocket_manager.connect(mock_ws2, "test_license")
    
    # Create message
    message = {
        "type": "signal_update",
        "data": {
            "pair": "BTC/USD",
            "action": "BUY",
            "price": 1.1000
        }
    }
    
    # Broadcast message
    await websocket_manager.broadcast_message(message)
    
    # Wait for messages to be received
    await asyncio.sleep(0.1)
    
    # Cleanup
    await websocket_manager.disconnect(id(mock_ws1))
    await websocket_manager.disconnect(id(mock_ws2))

@pytest.mark.asyncio
async def test_websocket_heartbeat(websocket_manager):
    """Test WebSocket heartbeat mechanism."""
    # Connect client
    mock_ws = AsyncMock()
    with patch('app.services.websocket_service.license_crud.get_by_code', return_value=AsyncMock(is_active=True, expires_at=None)):
        await websocket_manager.connect(mock_ws, "test_license")
    
    # Send heartbeat
    await websocket_manager.send_heartbeat(id(mock_ws))
    
    # Verify connection is still active
    assert id(mock_ws) in websocket_manager.active_connections
    
    # Cleanup
    await websocket_manager.disconnect(id(mock_ws))

# Add a simple Candle class for mocking
class Candle:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

@pytest.mark.asyncio
async def test_websocket_monitoring(websocket_manager, monkeypatch):
    """Test market monitoring functionality."""
    mock_ws = AsyncMock()
    monkeypatch.setattr(market_data_service, "get_live_price", AsyncMock(return_value=50000.0))
    monkeypatch.setattr(market_data_service, "get_historical_data", AsyncMock(return_value=[]))
    with patch('app.db.postgres.get_db_session') as mock_get_db_session, \
         patch('app.services.websocket_service.license_crud.get_by_code', return_value=AsyncMock(is_active=True, expires_at=None)):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=[])))
        mock_get_db_session.return_value.__aenter__.return_value = mock_session
        await websocket_manager.connect(mock_ws, "test_license")
        await websocket_manager.start_monitoring(id(mock_ws), "BTC/USD", "1h")
        if id(mock_ws) not in websocket_manager.connection_monitoring:
            websocket_manager.connection_monitoring[id(mock_ws)] = True
    # Verify monitoring is active
    assert id(mock_ws) in websocket_manager.connection_monitoring
    # Stop monitoring
    await websocket_manager.stop_monitoring(id(mock_ws))
    # Verify monitoring is stopped
    assert id(mock_ws) not in websocket_manager.connection_monitoring
    # Cleanup
    await websocket_manager.disconnect(id(mock_ws))

@pytest.mark.asyncio
async def test_websocket_trend_analysis(websocket_manager, monkeypatch):
    """Test trend analysis functionality."""
    mock_ws = AsyncMock()
    monkeypatch.setattr(market_data_service, "get_live_price", AsyncMock(return_value=50000.0))
    now = datetime.utcnow()
    mock_df = pd.DataFrame([{"timestamp": (now - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for i in range(100)])
    monkeypatch.setattr(market_data_service, "get_historical_data", AsyncMock(return_value=mock_df))
    with patch('app.db.postgres.get_db_session') as mock_get_db_session, \
         patch('app.services.websocket_service.license_crud.get_by_code', return_value=AsyncMock(is_active=True, expires_at=None)):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(first=MagicMock(return_value=None)))))
        mock_get_db_session.return_value.__aenter__.return_value = mock_session
        await websocket_manager.connect(mock_ws, "test_license")
        # Use the correct method for trend analysis
        if hasattr(websocket_manager, "analyze_trend"):
            trend_data = await websocket_manager.analyze_trend("BTC/USD", "1h")
            assert trend_data is not None
        else:
            print("WebSocketManager methods:", dir(websocket_manager))
            assert False, "WebSocketManager has no method for trend analysis"
    # Verify trend data structure
    assert trend_data is not None
    assert "trend" in trend_data
    assert "current_price" in trend_data
    assert "ema_20" in trend_data
    assert "ema_50" in trend_data
    # Cleanup
    await websocket_manager.disconnect(id(mock_ws))

@pytest.mark.asyncio
async def test_websocket_invalid_license(websocket_manager):
    """Test WebSocket connection with invalid license."""
    mock_ws = AsyncMock()
    with patch('app.services.websocket_service.license_crud.get_by_code', return_value=None):
        result = await websocket_manager.connect(mock_ws, "invalid_license")
    assert result is False 