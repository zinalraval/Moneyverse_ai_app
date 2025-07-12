import pytest
from fastapi import status
from app.main import app
from app.models.user import User
from app.utils.security import get_password_hash, verify_password
from app.core.security import create_access_token
from datetime import datetime, timedelta
import uuid
from app.api import dependencies
from app.db.postgres import get_db_session_factory,init_db
import asyncio
import time

@pytest.fixture
async def test_user(db_session):
    """Create a test user for auth tests."""
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

def test_login_success(client, db_session):
    """Test successful login."""
    print('DEBUG: client type:', type(client), 'repr:', repr(client))
    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    user = User(
        email=unique_email,
        hashed_password=get_password_hash("testpassword"),
        full_name="Test User",
        is_active=True
    )
    db_session.add(user)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(db_session.commit())
    loop.run_until_complete(db_session.refresh(user))
    login_data = {
        "username": unique_email,
        "password": "testpassword"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials(client):
    """Test login with invalid credentials."""
    login_data = {
        "username": "nonexistent@example.com",
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Incorrect email or password" in response.json()["detail"]

def test_refresh_token_success(client, test_user):
    """Test successful token refresh."""
    # First login to get refresh token
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    login_response = client.post("/api/v1/auth/login", data=login_data)
    refresh_token = login_response.json()["refresh_token"]
    
    # Use refresh token to get new access token
    headers = {"Authorization": f"Bearer {refresh_token}"}
    response = client.post("/api/v1/auth/refresh", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"

def test_refresh_token_invalid(client):
    """Test refresh token with invalid token."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post("/api/v1/auth/refresh", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid refresh token" in response.json()["detail"]

def test_refresh_token_missing(client):
    """Test refresh token with missing token."""
    response = client.post("/api/v1/auth/refresh")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_refresh_token_expired(client, test_user):
    """Test refresh token with expired token."""
    # Create an expired refresh token
    expired_token = create_access_token(
        data={"sub": test_user.email, "type": "refresh"},
        expires_delta=timedelta(microseconds=1)
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = client.post("/api/v1/auth/refresh", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid refresh token" in response.json()["detail"]

def test_access_token_expired(client, test_user):
    """Test access token expiration and refresh flow."""
    # Create an expired access token
    expired_token = create_access_token(
        data={"sub": test_user.email},
        expires_delta=timedelta(microseconds=1)
    )
    time.sleep(1)  # Ensure token is expired
    headers = {"Authorization": f"Bearer {expired_token}"}
    
    # Try to access protected endpoint with expired token
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_protected_endpoint_with_valid_token(client, test_user):
    """Test accessing protected endpoint with valid token."""
    # First login to get access token
    login_data = {
        "username": test_user.email,
        "password": "testpassword"
    }
    login_response = client.post("/api/v1/auth/login", data=login_data)
    access_token = login_response.json()["access_token"]
    
    # Access protected endpoint
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["email"] == test_user.email 

def test_password_hash_and_verify():
    password = "testpassword"
    hashed = get_password_hash(password)
    print(f"HASHED: {hashed}")
    assert verify_password(password, hashed), "Password verification failed" 