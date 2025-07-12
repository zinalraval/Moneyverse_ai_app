import pytest
from fastapi.testclient import TestClient
from fastapi import status
from fastapi import WebSocketDisconnect
from app.main import app
from app.models.user import User
from app.models.signal import Signal, SignalType, SignalStatus, SignalDirection, SignalTimeframe, SignalSetup
from app.models.license import License
from app.utils.security import get_password_hash
from app.api import dependencies
import uuid
from datetime import datetime, timedelta
import json
from typing import Dict
from app.config import settings
from app.core.security import create_access_token
from app.db.postgres import get_db_session_factory,init_db
import asyncio
import websockets
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from app.services.market_data import market_data_service
import time

# Fixtures
@pytest.fixture
async def override_get_db(db_session):
    async def _override_get_db():
        yield db_session
    app.dependency_overrides[dependencies.get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(dependencies.get_db, None)

@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for test user."""
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    response = TestClient(app).post("/api/v1/auth/login", data=login_data)
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}

@pytest.fixture
async def override_get_current_user(test_user):
    """Override the get_current_user dependency for testing."""
    async def _override_get_current_user():
        return test_user
    app.dependency_overrides[dependencies.get_current_user] = _override_get_current_user
    yield
    app.dependency_overrides.pop(dependencies.get_current_user, None)

@pytest.fixture
async def override_get_market_data(mock_market_data):
    """Override the get_market_data dependency for testing."""
    async def _override_get_market_data():
        return mock_market_data
    app.dependency_overrides[dependencies.get_market_data] = _override_get_market_data
    yield
    app.dependency_overrides.pop(dependencies.get_market_data, None)

@pytest.fixture
def client(override_get_db, override_get_current_user, mock_market_data):
    """Create a test client with overridden dependencies."""
    from app.api import dependencies
    async def _async_override_get_market_data():
        return mock_market_data
    app.dependency_overrides[dependencies.get_market_data] = _async_override_get_market_data
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(dependencies.get_market_data, None)

@pytest.fixture
async def test_user(db_session):
    """Create a test user for API tests."""
    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    user = User(
        email=unique_email,
        hashed_password=get_password_hash("testpassword"),
        full_name="Test User",
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest.fixture
async def test_license(db_session, test_user):
    """Create a test license for API tests."""
    license = License(
        code=f"TEST-{uuid.uuid4().hex[:8]}",
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(days=30),
        user_id=test_user.id
    )
    db_session.add(license)
    await db_session.commit()
    await db_session.refresh(license)
    return license

@pytest.fixture
async def test_signal(db_session, test_user):
    """Create a test signal for API tests."""
    signal = Signal(
        pair="BTC/USD",
        type=SignalType.SCALPING,
        direction=SignalDirection.BUY,
        status=SignalStatus.ACTIVE,
        timeframe=SignalTimeframe.H1,
        entry_price=1.1000,
        stop_loss=1.0950,
        tp1=1.1050,
        tp2=1.1100,
        user_id=test_user.id,
        confidence=0.8,
        reason="Test signal",
        logic_note="Test logic",
        setup_conditions=[SignalSetup.MACD_CROSSOVER],
        is_news_filtered=False
    )
    db_session.add(signal)
    await db_session.commit()
    await db_session.refresh(signal)
    return signal

# Mock market data service
@pytest.fixture
def mock_market_data():
    with patch('app.services.market_data.market_data_service') as mock:
        mock.get_live_price = AsyncMock(return_value=1.1000)
        mock.get_candle_data = AsyncMock(return_value=pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'open': [1.1000],
            'high': [1.1100],
            'low': [1.0900],
            'close': [1.1050],
            'volume': [1000]
        }))
        mock.get_historical_data = AsyncMock(return_value=pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'open': [1.1000],
            'high': [1.1100],
            'low': [1.0900],
            'close': [1.1050],
            'volume': [1000]
        }))
        mock.initialize = AsyncMock()
        mock.close = AsyncMock()
        mock.get_market_data = AsyncMock(return_value={"price": 1.1000})
        yield mock

# Authentication Tests
def test_register_success(client, test_user):
    """Test successful user registration."""
    user_data = {
        "email": f"newuser_{uuid.uuid4().hex[:8]}@example.com",
        "password": "newpassword",
        "full_name": "New User",
        "user_id": test_user.id
    }
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["email"] == user_data["email"]

def test_register_duplicate_email(client, test_user):
    """Test registration with existing email."""
    user_data = {
        "email": test_user.email,
        "password": "newpassword",
        "full_name": "New User",
        "user_id": test_user.id
    }
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "already registered" in response.json()["detail"]

def test_login_success(client, test_user):
    """Test successful login."""
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "testtoken",
            "refresh_token": "testrefreshtoken",
            "token_type": "bearer"
        }
        mock_post.return_value = mock_response
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()

def test_login_invalid_credentials(client):
    """Test login with invalid credentials."""
    login_data = {
        "username": "nonexistent@example.com",
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_refresh_token_success(client, test_user):
    """Test successful token refresh."""
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "testtoken",
            "refresh_token": "testrefreshtoken",
            "token_type": "bearer"
        }
        mock_post.return_value = mock_response
    login_response = client.post("/api/v1/auth/login", data=login_data)
    refresh_token = login_response.json()["refresh_token"]
    response = client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {refresh_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()

def test_refresh_token_invalid(client):
    """Test refresh with invalid token."""
    response = client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_access_token_expired(test_user):
    """Test access token expiration and refresh flow."""
    expired_token = create_access_token(
        data={"sub": test_user.email},
        expires_delta=timedelta(microseconds=1)
    )
    time.sleep(1)  # Ensure token is expired
    headers = {"Authorization": f"Bearer {expired_token}"}
    # Use a fresh client with no dependency overrides
    client = TestClient(app)
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_protected_endpoint_with_valid_token(client, test_user):
    """Test accessing protected endpoint with valid token."""
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    with patch("requests.post") as mock_post, patch("httpx.post") as mock_httpx_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "testtoken",
            "refresh_token": "testrefreshtoken",
            "token_type": "bearer"
        }
        mock_post.return_value = mock_response
        mock_httpx_post.return_value = mock_response
        login_response = client.post("/api/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["email"] == test_user.email

# Signal Tests
def test_create_signal_success(client, test_user, auth_headers):
    """Test successful signal creation."""
    signal_data = {
        "pair": "BTC/USD",
        "type": "SCALPING",
        "direction": "BUY",
        "timeframe": "1H",
        "entry": 1.1000,
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
        "tp1": 1.1050,
        "tp2": 1.1100,
        "confidence": 0.8,
        "reason": "Test signal",
        "logic_note": "Test logic",
        "setup_conditions": ["MACD_CROSSOVER"],
        "is_news_filtered": False
    }
    response = client.post("/api/v1/signals", json=signal_data, headers=auth_headers)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["pair"] == signal_data["pair"]
    assert data["type"] == signal_data["type"]
    assert data["direction"] == signal_data["direction"]
    assert data["timeframe"] == signal_data["timeframe"]
    assert data["entry"] == signal_data["entry"]
    assert data["entry_price"] == signal_data["entry_price"]
    assert data["stop_loss"] == signal_data["stop_loss"]
    assert data["tp1"] == signal_data["tp1"]
    assert data["tp2"] == signal_data["tp2"]
    assert data["confidence"] == signal_data["confidence"]
    assert data["reason"] == signal_data["reason"]
    assert data["logic_note"] == signal_data["logic_note"]
    assert data["setup_conditions"] == signal_data["setup_conditions"]
    assert data["is_news_filtered"] == signal_data["is_news_filtered"]

def test_create_signal_invalid_data(client, auth_headers):
    """Test signal creation with invalid data."""
    signal_data = {
        "pair": "INVALID/PAIR",
        "type": "INVALID_TYPE",
        "direction": "INVALID_DIRECTION"
    }
    response = client.post("/api/v1/signals", json=signal_data, headers=auth_headers)
    if response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_get_signals_list(client, test_user, auth_headers):
    """Test getting signals list with filters."""
    response = client.get("/api/v1/signals", headers=auth_headers)
    data = response.json()
    assert isinstance(data, list)
    # Optionally, check for expected signals
    # assert len(data) > 0 or == 0 depending on setup

def test_get_signal_detail(client, test_user, test_signal, auth_headers):
    """Test getting signal details."""
    response = client.get(
        f"/api/v1/signals/{test_signal.id}",
        headers=auth_headers
    )
    if response.status_code != status.HTTP_200_OK:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == test_signal.id
    assert "stop_loss" in data
    assert "confidence" in data
    assert "reason" in data
    assert "logic_note" in data
    assert "setup_conditions" in data

def test_update_signal(client, test_user, test_signal, auth_headers):
    """Test updating a signal."""
    update_data = {
        "status": "COMPLETED",
        "exit_price": 1.1050,
        "stop_loss": 1.0960
    }
    response = client.put(
        f"/api/v1/signals/{test_signal.id}",
        json=update_data,
        headers=auth_headers
    )
    if response.status_code != status.HTTP_200_OK:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == update_data["status"]
    assert data["exit_price"] == update_data["exit_price"]
    assert data["stop_loss"] == update_data["stop_loss"]

def test_delete_signal(client, auth_headers, test_signal):
    """Test deleting signal."""
    response = client.delete(f"/api/v1/signals/{test_signal.id}", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

# Market Data Tests
@pytest.mark.asyncio
async def test_get_market_data(client, auth_headers, mock_market_data):
    """Test getting market data."""
    response = client.get(
        "/api/v1/market-data/BTC/USD",
        headers=auth_headers
    )
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.asyncio
async def test_get_candles(client, auth_headers, mock_market_data):
    """Test getting candle data."""
    response = client.get(
        "/api/v1/market-data/BTC/USD/candles?timeframe=1H&limit=30",
        headers=auth_headers
    )
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.asyncio
async def test_get_historical_data(client, auth_headers, mock_market_data):
    """Test getting historical data."""
    start_time = (datetime.utcnow() - timedelta(days=1)).isoformat()
    end_time = datetime.utcnow().isoformat()
    response = client.get(
        f"/api/v1/market-data/BTC/USD/historical?start_time={start_time}&end_time={end_time}&timeframe=1H",
        headers=auth_headers
    )
    if response.status_code != status.HTTP_200_OK:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "candles" in data

def test_get_market_data_invalid_pair(client, auth_headers):
    """Test market data retrieval with invalid pair."""
    response = client.get("/api/v1/market-data/INVALID/USD?timeframe=1H", headers=auth_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Unsupported trading pair" in response.json()["detail"]

# License Tests
def test_validate_license(client, test_user, test_license, auth_headers):
    """Test license validation."""
    response = client.post(f"/api/v1/licenses/validate?license_code={test_license.code}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["code"] == test_license.code

def test_create_license(client, test_user, auth_headers):
    """Test license creation."""
    license_data = {
        "user_email": test_user.email,
        "expiry_days": 30,
        "features": []
    }
    response = client.post("/api/v1/licenses", json=license_data, headers=auth_headers)
    if response.status_code != status.HTTP_200_OK:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "code" in data
    assert data["is_active"] is True

def test_get_license(client, test_user, test_license, auth_headers):
    """Test getting license details."""
    response = client.get(f"/api/v1/licenses/{test_license.code}", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["code"] == test_license.code

def test_update_license(client, test_user, test_license, auth_headers):
    """Test license update."""
    update_data = {
        "code": test_license.code,
        "is_active": False,
        "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "user_id": test_user.id
    }
    response = client.put(f"/api/v1/licenses/{test_license.code}", json=update_data, headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["is_active"] == update_data["is_active"]

def test_deactivate_license(client, test_user, test_license, auth_headers):
    """Test deactivating license."""
    response = client.post(f"/api/v1/licenses/{test_license.code}/deactivate", headers=auth_headers)
    if response.status_code != status.HTTP_200_OK:
        print("Response JSON:", response.json())
    assert response.status_code == status.HTTP_200_OK

# System Tests
def test_ping(client):
    """Test ping endpoint."""
    response = client.get("/api/v1/ping")
    assert response.status_code == status.HTTP_200_OK
    assert "message" in response.json()
    assert "timestamp" in response.json()

def test_health(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"

def test_start_signal_monitoring(client, auth_headers):
    """Test starting signal monitoring."""
    response = client.post("/api/v1/start-signal-monitoring", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == "Signal monitoring started successfully"

def test_initialize_market_data(client, auth_headers):
    """Test initializing market data."""
    response = client.post("/api/v1/initialize-market-data", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

# WebSocket Tests
def test_websocket_connection(client, auth_headers):
    """Test WebSocket connection."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/signals?license_code=TEST-123") as websocket:
            data = websocket.receive_json()
            assert "type" in data
            assert "data" in data

def test_websocket_invalid_license(client, auth_headers):
    """Test WebSocket connection with invalid license."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/signals?license_code=invalid") as websocket:
            websocket.receive_json()

# Error Cases
def test_unauthorized_access(client):
    """Test accessing protected endpoints without authentication."""
    app.dependency_overrides.pop(dependencies.get_current_user, None)
    response = client.get("/api/v1/auth/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_invalid_token(client):
    """Test accessing protected endpoints with invalid token."""
    app.dependency_overrides.pop(dependencies.get_current_user, None)
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_rate_limiting(client, auth_headers):
    """Test rate limiting on sensitive endpoints."""
    # Try to generate signals multiple times
    for _ in range(6):
        response = client.post(
            "/api/v1/signals/generate",
            json={"pair": "BTCUSD", "type": "SCALPING"},
            headers=auth_headers
        )
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS  