import logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
from fastapi import FastAPI, Request, status, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from app.api.routes import router as api_router
from app.api.websocket import router as websocket_router
from app.config import settings
from app.utils.logging import setup_logging
from app.db.postgres import init_db, get_db_engine
try:
    from app.db.postgres import async_init_db
except ImportError:
    async_init_db = None
from app.services.market_data import initialize as market_data_initialize, market_data_service
from app.services.signal_automation import start_signal_automation
from app.services.signal_monitor import start_signal_monitor
from app.services.signal_service import SignalService
import logging
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Dict, Optional
import time
import traceback
import os
import sys
from datetime import datetime
import asyncio
import signal
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_openmetrics
# from app.db.mongodb import client as mongo_client
import sqlalchemy
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import warnings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Force all major loggers to ERROR level
import logging as _logging
_logging.getLogger("sqlalchemy.engine").setLevel(_logging.ERROR)
_logging.getLogger("uvicorn").setLevel(_logging.ERROR)
_logging.getLogger("uvicorn.access").setLevel(_logging.ERROR)
_logging.getLogger("app").setLevel(_logging.ERROR)
_logging.getLogger("app.api.routes").setLevel(_logging.ERROR)
_logging.getLogger("app.services.signal_monitor").setLevel(_logging.ERROR)
_logging.getLogger("app.services.market_data").setLevel(_logging.ERROR)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_CONNECTIONS = Counter('websocket_connections_active', 'Active WebSocket connections')

# Global variables for graceful shutdown
signal_service: Optional[SignalService] = None
shutdown_event = asyncio.Event()

# Initialize Sentry for error tracking
SENTRY_DSN = getattr(settings, 'SENTRY_DSN', None) or os.getenv('SENTRY_DSN')
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1,  # Adjust as needed
        environment=settings.ENVIRONMENT,
        release=settings.VERSION
    )
    logger.info("Sentry error tracking initialized.")
else:
    logger.info("Sentry DSN not set. Error tracking is disabled.")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global signal_service
    
    try:
        # Initialize services on startup
        logger.info("Starting application initialization...")
        start_time = time.time()
        
        # Initialize database if not in testing mode
        if not getattr(settings, 'TESTING', False):
            logger.info("Initializing database...")
            if async_init_db:
                await async_init_db()
            else:
                init_db()
            logger.info("Database initialized successfully")
        
        # Initialize market data service
        logger.info("Initializing market data service...")
        await market_data_initialize()
        logger.info("Market data service initialized successfully")
        
        # Start signal automation if not in testing mode
        if not getattr(settings, 'TESTING', False):
            logger.info("Starting signal automation...")
            await start_signal_automation()
            logger.info("Signal automation started successfully")
        
        # Start signal monitor if not in testing mode
        if not getattr(settings, 'TESTING', False):
            logger.info("Starting signal monitor...")
            await start_signal_monitor()
            logger.info("Signal monitor started successfully")
        
        # Initialize signal service
        signal_service = SignalService()
        logger.info("Starting signal service...")
        await signal_service.start()
        logger.info("Signal service started successfully")
        
        init_time = time.time() - start_time
        logger.info(f"Application initialization completed successfully in {init_time:.2f} seconds")
        logger.info("Application is ready to serve requests")
        
        yield
        
        logger.info("Shutdown phase initiated")
        
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Performing cleanup operations...")
        cleanup_start = time.time()
        
        try:
           if signal_service:
            await signal_service.stop()
            logger.info("Signal service stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping signal service: {str(e)}")

        cleanup_time = time.time() - cleanup_start
        logger.info(f"Application shutdown complete in {cleanup_time:.2f} seconds")

# Create FastAPI app with production settings
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-ready MoneyVerse AI Backend API",
    openapi_url="/openapi.json",
    docs_url="/docs" if not settings.PRODUCTION else None,
    redoc_url="/redoc" if not settings.PRODUCTION else None,
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Production middleware stack
if settings.PRODUCTION:
    # Force HTTPS in production
    app.add_middleware(HTTPSRedirectMiddleware)
    
    # Trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.get_cors_methods(),
    allow_headers=settings.get_cors_headers(),
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enhanced request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request details
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    logger.info(f"[REQUEST] {request.method} {request.url.path} - IP: {client_ip} - UA: {user_agent}")
    
    try:
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log response details
        logger.info(f"[RESPONSE] {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s")
        
        # Record metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        REQUEST_LATENCY.observe(duration)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[ERROR] {request.method} {request.url.path} - Error: {str(e)} - Duration: {duration:.3f}s")
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()
        raise

# Global exception handler with improved error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    error_id = f"err_{int(time.time())}_{hash(str(exc)) % 10000}"
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception [{error_id}]: {str(exc)}")
    logger.error(f"Traceback [{error_id}]: {error_traceback}")
    # Report to Sentry if enabled
    if SENTRY_DSN:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("error_id", error_id)
            scope.set_extra("request_url", str(request.url))
            scope.set_extra("method", request.method)
            scope.set_extra("traceback", error_traceback)
            sentry_sdk.capture_exception(exc)
    # Record error metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()
    # Return appropriate error response based on environment
    if settings.PRODUCTION:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    else:
        return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
                "error_id": error_id,
                "traceback": error_traceback.splitlines(),
                "timestamp": datetime.utcnow().isoformat()
        }
    )

# Health check endpoints
@app.get("/health")
async def health_check() -> Dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/health/detailed")
async def detailed_health_check() -> Dict:
    """Detailed health check endpoint with service status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "signal_service": "running" if signal_service else "stopped",
            "database": "unknown",
            "mongodb": "unknown",
            "market_data": "unknown"
        },
        "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
        "memory_usage": get_memory_usage()
    }
    # --- Postgres check ---
    try:
        engine = get_db_engine()
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        health_status["services"]["database"] = "connected"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    # --- MongoDB check ---
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: mongo_client.admin.command('ping'))
        health_status["services"]["mongodb"] = "connected"
    except Exception as e:
        health_status["services"]["mongodb"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    # --- Market Data Provider check ---
    try:
        # Try a simple price fetch for a major pair
        await market_data_service.get_live_price("BTC/USD")
        health_status["services"]["market_data"] = "connected"
    except Exception as e:
        health_status["services"]["market_data"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    # If any service is down, set status to degraded
    if any(v.startswith("error") or v == "stopped" for v in health_status["services"].values()):
        health_status["status"] = "degraded"
    return health_status

@app.get("/health/ready")
async def readiness_check() -> Dict:
    """Readiness check for Kubernetes."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/live")
async def liveness_check() -> Dict:
    """Liveness check for Kubernetes."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(websocket_router)

# Utility functions
def get_memory_usage() -> Dict:
    """Get memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}

@app.on_event("startup")
async def print_routes():
    """Log all registered routes on startup."""
    logger = logging.getLogger("uvicorn")
    logger.info("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ', '.join(route.methods) if route.methods else 'GET'
            logger.info(f"  {methods} {route.path}")
    
    # Store startup time for uptime calculation
    app.state.start_time = time.time()
    
    logger.info(f"Application started in {settings.ENVIRONMENT} mode")
    logger.info(f"API version: {settings.VERSION}")
    logger.info(f"Production mode: {settings.PRODUCTION}")

# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event_handler():
    """Handle graceful shutdown."""
    logger.info("Shutdown event received, waiting for active connections to close...")
    
    # Wait for shutdown signal or timeout
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timeout reached, forcing shutdown")
    
    logger.info("Shutdown complete")

# WebSocket connection tracking
@app.websocket("/ws/connection-tracker")
async def websocket_connection_tracker(websocket: WebSocket):
    """Track WebSocket connections for monitoring."""
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.on_event("startup")
def check_api_keys():
    required_keys = [
        ("TWELVEDATA_API_KEY", "Market Data (TwelveData)"),
        ("ALPHAVANTAGE_API_KEY", "Market Data (AlphaVantage)"),
        ("NEWS_API_KEY", "News Provider")
    ]
    for key, desc in required_keys:
        if not os.getenv(key):
            logging.warning(f"[Startup] {desc} API key ({key}) is missing. The backend will use demo/mock data for this service.")

if __name__ == "__main__":
    import uvicorn
    
    # Production server configuration
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=not settings.PRODUCTION,
        workers=settings.WORKERS if settings.PRODUCTION else 1,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*" if not settings.PRODUCTION else settings.TRUSTED_PROXIES
    )

# Suppress all Python warnings (including Pydantic) at startup to keep logs clean
warnings.filterwarnings("ignore")
