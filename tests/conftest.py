import os
if os.path.exists("test.db"):
    os.remove("test.db")
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["MOCK_MARKET_DATA"] = "true"
os.environ["TESTING"] = "true"
from app.config import settings
import pytest
import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from typing import AsyncGenerator
from app.main import app
from app.db.postgres import get_db, get_db_session_factory
from app.models.user import User
from app.models.license import License
from app.core.security import create_access_token, get_password_hash
from app.services.market_data import market_data_service
from datetime import datetime, timedelta
from sqlalchemy.pool import NullPool
from unittest.mock import AsyncMock
import pandas as pd
from sqlalchemy import text
from app.api import dependencies
from app.db import postgres

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def engine():
    engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture(scope="function")
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    async with async_session() as session:
        yield session
        await session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()

@pytest.fixture(scope="function")
async def test_user(db_session):
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

@pytest.fixture(scope="function")
def client(test_user, db_session, monkeypatch):
    async def _async_get_db():
        yield db_session
    async def _async_get_db_session():
        yield db_session
    monkeypatch.setattr(dependencies, "get_db", _async_get_db)
    monkeypatch.setattr(postgres, "get_db_session", _async_get_db_session)
    from starlette.testclient import TestClient
    with TestClient(app) as test_client:
        yield test_client
    monkeypatch.setattr(dependencies, "get_db", dependencies.get_db)
    monkeypatch.setattr(postgres, "get_db_session", postgres.get_db_session)

@pytest.fixture
async def test_license(db_session, test_user) -> License:
    """Create a test license."""
    license = License(
        code=f"TEST_{uuid.uuid4().hex[:8]}",
        user_id=test_user.id,
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db_session.add(license)
    await db_session.commit()
    await db_session.refresh(license)
    return license

@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for test user."""
    access_token = create_access_token(
        data={"sub": test_user.email, "type": "access"}
    )
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture(autouse=True)
async def clean_signals_table(db_session):
    from app.models.signal import Signal
    await db_session.execute(text("DELETE FROM signals"))
    await db_session.commit()
    yield

@pytest.fixture
async def test_signal(db_session, test_user):
    """Create a test signal."""
    from app.models.signal import Signal, SignalType, SignalDirection, SignalStatus, SignalTimeframe, SignalSetup
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

@pytest.fixture(autouse=True)
def patch_market_data(monkeypatch):
    now = datetime.utcnow()
    mock_df = pd.DataFrame([
        {"timestamp": (now - timedelta(hours=i)), "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "volume": 100}
        for i in range(100)
    ])
    monkeypatch.setattr(market_data_service, "get_live_price", AsyncMock(return_value=50000.0))
    monkeypatch.setattr(market_data_service, "get_historical_data", AsyncMock(return_value=mock_df))
    monkeypatch.setattr(market_data_service, "get_candle_data", AsyncMock(return_value=mock_df))

@pytest.fixture(autouse=True)
def override_db(monkeypatch, db_session):
    def _sync_get_db():
        yield db_session
    def _sync_get_db_session():
        yield db_session
    monkeypatch.setattr(dependencies, "get_db", _sync_get_db)
    monkeypatch.setattr(postgres, "get_db_session", _sync_get_db_session)
    yield
    monkeypatch.setattr(dependencies, "get_db", dependencies.get_db)
    monkeypatch.setattr(postgres, "get_db_session", postgres.get_db_session) 