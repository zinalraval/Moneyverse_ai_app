import logging
import sys
from logging.handlers import RotatingFileHandler
from app.config import settings
import os

def setup_logging() -> None:
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR) # Changed from settings.LOG_LEVEL (INFO) to ERROR
    
    # For debugging, set root logger to DEBUG
    if settings.DEBUG: # Use settings.DEBUG to control overall verbosity
        root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if root_logger.handlers:
        return

    # Create formatters
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/moneyverse.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific log levels for different modules for more verbose debugging
    logging.getLogger("uvicorn").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    # Explicitly ensure app-level loggers capture all debug information relevant to the application logic
    logging.getLogger("app").setLevel(logging.ERROR) # General app logger
    logging.getLogger("app.api.routes").setLevel(logging.ERROR) # Specific for routes
    logging.getLogger("app.services.signal_monitor").setLevel(logging.ERROR) # For signal monitor
    logging.getLogger("app.services.market_data").setLevel(logging.ERROR) # For market data

    # Log startup message
    logging.info("Logging configured successfully")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name) 