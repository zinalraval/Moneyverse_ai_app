from typing import List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, validator
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    PRODUCTION: bool = False
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Moneyverse AI Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Security
    SECRET_KEY: str = "a46a94ee45cd31fb2a06cec30acaef15ef10463511902d4cf87964d15b55c610"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALLOWED_HOSTS: List[str] = ["*"]
    TRUSTED_PROXIES: List[str] = ["127.0.0.1", "::1"]
    
    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "moneyverse"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: str = "5432"
    POSTGRES_SCHEMA: str = "public"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL."""
        if self.TESTING:
            return "sqlite+aiosqlite:///./test.db"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # CORS
    CORS_ORIGINS: str = "*"
    CORS_METHODS: str = "*"
    CORS_HEADERS: str = "*"
    CORS_CREDENTIALS: bool = True
    
    # MongoDB
    # IMPORTANT: The hostname must match the service name in docker-compose.yml (mongo)
    MONGODB_URL: str = "mongodb://root:rootpassword@mongo:27017"
    MONGODB_DB: str = "moneyverse"
    
    # Market Data
    TWELVEDATA_API_KEY: str = "6551ab0a0d924caa87c0aba7bea23b99"
    TWELVEDATA_BASE_URL: str = "https://api.twelvedata.com"
    MARKET_DATA_CACHE_TTL: int = 300  # 5 minutes
    CANDLE_DATA_UPDATE_INTERVAL: int = 5  # seconds
    SIGNAL_MONITOR_INTERVAL: int = 5 # seconds
    SIGNAL_UPDATE_INTERVAL: int = 30 # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # App Settings
    APP_NAME: str = "MoneyVerse"
    
    # License validation
    LICENSE_CHECK_INTERVAL_HOURS: int = 24
    
    # Testing
    TESTING: bool = False
    MOCK_MARKET_DATA: bool = False
    
    # Monitoring and Metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Performance
    REQUEST_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 100
    
    # Security Headers
    SECURITY_HEADERS: bool = True
    HSTS_MAX_AGE: int = 31536000
    CSP_POLICY: str = "default-src 'self'"
    
    def get_cors_origins(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    def get_cors_methods(self) -> List[str]:
        if self.CORS_METHODS == "*":
            return ["*"]
        return [method.strip() for method in self.CORS_METHODS.split(",")]

    def get_cors_headers(self) -> List[str]:
        if self.CORS_HEADERS == "*":
            return ["*"]
        return [header.strip() for header in self.CORS_HEADERS.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

settings = get_settings()
