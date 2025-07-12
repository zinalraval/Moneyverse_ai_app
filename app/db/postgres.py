import logging
from app.config import settings

logger = logging.getLogger(__name__)

def init_db():
    """No-op for sync/test mode. In async mode, use async_init_db."""
    pass

def close_db():
    """No-op for sync/test mode. In async mode, use async_close_db."""
    pass

if settings.DATABASE_URL.startswith("sqlite:///"):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    def get_db_session_factory():
        return SessionLocal
    def get_db_session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    def get_db():
        yield from get_db_session()
else:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import NullPool
    from typing import AsyncGenerator
    from contextlib import asynccontextmanager
    engine_kwargs = {
        "echo": True,
        "poolclass": NullPool
    } if settings.DEBUG else {
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_timeout": settings.DB_POOL_TIMEOUT,
        "pool_recycle": settings.DB_POOL_RECYCLE
    }
engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs
)
async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)
Base = declarative_base()
def get_db_engine():
    return engine
def get_db_session_factory():
    return async_session_factory
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
async def async_init_db() -> None:
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise
async def async_close_db() -> None:
    try:
        await engine.dispose()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {str(e)}", exc_info=True)
        raise
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
