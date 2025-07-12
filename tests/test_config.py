from pydantic_settings import BaseSettings
from typing import List
import os

class TestSettings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Moneyverse AI Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "test_secret_key_123"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database Settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "moneyverse_test"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_SCHEMA: str = "public"
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/moneyverse_test"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # CORS Settings
    CORS_ORIGINS: str = "*"  # Changed to string
    CORS_METHODS: str = "*"  # Changed to string
    CORS_HEADERS: str = "*"  # Changed to string
    CORS_CREDENTIALS: bool = True

    # Testing
    TESTING: bool = True

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

    class Config:
        env_file = None  # Disable env file loading
        case_sensitive = True

# Create test settings instance
test_settings = TestSettings()

# Override any environment variables
os.environ["TESTING"] = "true"
os.environ["DEBUG"] = "true"
os.environ["POSTGRES_SCHEMA"] = "public"
os.environ["POSTGRES_URL"] = test_settings.POSTGRES_URL 