import pytest
from fastapi import status
from datetime import datetime, timedelta
import pandas as pd
from app.services.market_data import market_data_service
from unittest.mock import patch, AsyncMock

# Market Data Tests
def test_get_market_data_success(client, auth_headers):
    """Test successful market data retrieval."""
    mock_df = pd.DataFrame([{"timestamp": "2023-01-01T00:00:00Z", "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for _ in range(100)])
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=mock_df)):
        response = client.get("/api/v1/market-data/BTC/USD", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
        assert "price" in response.json()

def test_get_market_data_invalid_pair(client, auth_headers):
    """Test market data retrieval with invalid pair."""
    # Backend returns 404 for not found
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=[])):
    response = client.get("/api/v1/market-data/INVALID", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Unsupported trading pair" in response.json()["detail"]

def test_get_market_data_unauthorized(client):
    """Test market data retrieval without authentication."""
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=[])):
        response = client.get("/api/v1/market-data/BTC/USD")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# Historical Data Tests
def test_get_historical_data_success(client, auth_headers):
    """Test successful historical data retrieval."""
    now = datetime.utcnow()
    mock_df = pd.DataFrame([{"timestamp": (now - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for i in range(100)])
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=mock_df)):
        response = client.get("/api/v1/historical-data/BTC/USD?timeframe=1h&limit=100", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "candles" in data
        assert isinstance(data["candles"], list)

def test_get_historical_data_with_time_range(client, auth_headers):
    """Test historical data retrieval with time range."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    mock_df = pd.DataFrame([{"timestamp": (end_time - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for i in range(100)])
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=mock_df)):
        response = client.get(f"/api/v1/historical-data/BTC/USD?timeframe=1h&start_time={start_time.isoformat()}&end_time={end_time.isoformat()}", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "candles" in data
    if data["candles"]:
        assert datetime.fromisoformat(data["candles"][0]["timestamp"]) >= start_time

def test_get_historical_data_invalid_timeframe(client, auth_headers):
    """Test historical data retrieval with invalid timeframe."""
    mock_df = pd.DataFrame([])
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=mock_df)):
        response = client.get("/api/v1/historical-data/BTC/USD?timeframe=invalid", headers=auth_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# Add a simple Candle class for mocking
class Candle:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

# Trend Analysis Tests
def test_get_trend_analysis_success(client, auth_headers, monkeypatch):
    """Test successful trend analysis retrieval."""
    now = datetime.utcnow()
    mock_df = pd.DataFrame([{"timestamp": (now - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for i in range(100)])
    monkeypatch.setattr(market_data_service, "get_live_price", AsyncMock(return_value=50000.0))
    monkeypatch.setattr(market_data_service, "get_historical_data", AsyncMock(return_value=mock_df))
    response = client.get("/api/v1/trend-analysis/BTC/USD?timeframe=1h", headers=auth_headers)
    if response.status_code != status.HTTP_200_OK:
        print("DEBUG: Status code:", response.status_code)
        print("DEBUG: Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "trend" in data
    assert "current_price" in data
    assert "ema_20" in data
    assert "ema_50" in data

def test_get_trend_analysis_invalid_pair(client, auth_headers, monkeypatch):
    """Test trend analysis with invalid pair."""
    now = datetime.utcnow()
    mock_df = pd.DataFrame([{"timestamp": (now - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100} for i in range(100)])
    monkeypatch.setattr(market_data_service, "get_live_price", AsyncMock(return_value=50000.0))
    monkeypatch.setattr(market_data_service, "get_historical_data", AsyncMock(return_value=mock_df))
    response = client.get("/api/v1/trend-analysis/INVALID/USD?timeframe=1h", headers=auth_headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported trading pair" in response.json()["detail"]

def test_get_trend_analysis_unauthorized(client):
    """Test trend analysis without authentication."""
    with patch.object(market_data_service, "get_live_price", new=AsyncMock(return_value=50000.0)), \
         patch.object(market_data_service, "get_historical_data", new=AsyncMock(return_value=[])):
        response = client.get("/api/v1/trend-analysis/BTC/USD?timeframe=1h")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED 