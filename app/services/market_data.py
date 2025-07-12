import os
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
import pandas as pd
import asyncio
import logging
from app.config import settings
from app.models.signal import SignalType, SignalTimeframe
from pydantic import BaseModel
import numpy as np
import aiohttp
from cachetools import TTLCache
from collections import defaultdict
import random
from app.core.exceptions import MarketDataError
import time
from prometheus_client import Counter, Histogram, Gauge
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from app.services.market_data_utils import SUPPORTED_PAIRS, update_market_data, update_price
from app.core.callbacks import callback_manager
import yfinance as yf
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
if TYPE_CHECKING:
    from app.services.websocket_service import WebSocketManager

logger = logging.getLogger(__name__)

# app/services/market_data.py
# Market data service using Twelve Data API
# Ensure you have the Twelve Data API key set in your environment variables

TWELVEDATA_API_KEY = settings.TWELVEDATA_API_KEY
BASE_URL = settings.TWELVEDATA_BASE_URL

# Cache configuration
price_cache = TTLCache(maxsize=100, ttl=settings.MARKET_DATA_CACHE_TTL)
candle_cache = TTLCache(maxsize=100, ttl=settings.MARKET_DATA_CACHE_TTL)

# Map signal types to Twelve Data intervals
SIGNAL_TYPE_TO_INTERVAL = {
    SignalType.SCALPING: "5m",
    SignalType.INTRADAY: "1h",
    SignalType.SWING: "4h"
}

# Map signal types to Alpha Vantage intervals
SIGNAL_TYPE_TO_INTERVAL_ALPHA = {
    SignalType.SCALPING: "1min",
    SignalType.INTRADAY: "5min",
    SignalType.SWING: "60min"
}

# Map SignalTimeframe to Twelve Data intervals
TIMEFRAME_TO_INTERVAL = {
    SignalTimeframe.M5: "5min",
    SignalTimeframe.M30: "30m",
    SignalTimeframe.H1: "1h",
    SignalTimeframe.H4: "4h",
    SignalTimeframe.D1: "1day"
}

# Metrics
API_REQUESTS = Counter('market_data_api_requests_total', 'Total API requests made', ['endpoint', 'status'])
API_LATENCY = Histogram('market_data_api_latency_seconds', 'API request latency in seconds', ['endpoint'])
CACHE_HITS = Counter('market_data_cache_hits_total', 'Total cache hits', ['type'])
CACHE_MISSES = Counter('market_data_cache_misses_total', 'Total cache misses', ['type'])
MOCK_DATA_USAGE = Counter('market_data_mock_usage_total', 'Total times mock data was used', ['endpoint'])
RATE_LIMIT_HITS = Counter('market_data_rate_limit_hits_total', 'Total rate limit hits')
ACTIVE_REQUESTS = Gauge('market_data_active_requests', 'Number of active API requests')

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class MarketDataService:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.api_key = settings.TWELVEDATA_API_KEY
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = datetime.now()
        self._min_request_interval = timedelta(seconds=30)  # 30 seconds between requests
        self._initialized = False
        self._use_mock_data = settings.MOCK_MARKET_DATA  # Use mock data if set in env
        self._rate_limit_reset_time = None  # Track when rate limits will reset
        self._price_cache = TTLCache(maxsize=100, ttl=60)  # Cache prices for 60 seconds
        self._candle_cache = TTLCache(maxsize=100, ttl=300)  # Cache candles for 5 minutes
        self._retry_count = 0
        self._max_retries = 3
        self._request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        self._ws_connections = {}
        self._fallback_polling = {}
        self._fallback_interval = 5  # seconds
        self._max_fallback_attempts = 3
        self._fallback_attempts = defaultdict(int)
        self._candle_update_task: Optional[asyncio.Task] = None # New attribute for periodic candle updates
        self._last_update = {}
        self._live_price_subscribers = set()
        self._candle_subscribers = set()
        self._update_interval = 30  # seconds, for periodic updates
        self._max_latency = 2  # seconds
        self.websocket_manager: "WebSocketManager" = None
        
        # Market hours configuration
        self._market_hours = {
            "forex": {
                "open": "00:00",
                "close": "23:59",
                "timezone": "UTC"
            },
            "crypto": {
                "open": "00:00",
                "close": "23:59",
                "timezone": "UTC"
            },
            "stocks": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York"
            }
        }
        
        # Volatility profiles for different timeframes
        self._volatility_profiles = {
            "1min": 0.0005,  # 0.05% per minute
            "5min": 0.001,   # 0.1% per 5 minutes
            "15min": 0.002,  # 0.2% per 15 minutes
            "30min": 0.003,  # 0.3% per 30 minutes
            "1h": 0.004,     # 0.4% per hour
            "4h": 0.008,     # 0.8% per 4 hours
            "1day": 0.02     # 2% per day
        }

    async def initialize(self):
        """Initialize the market data service."""
        if not self._initialized:
            self._session = aiohttp.ClientSession()
            self._initialized = True
            # Start periodic candle updates
            self._candle_update_task = asyncio.create_task(self._update_candles_periodically())
        logger.info("Market data service initialized")

    async def close(self):
        """Close the market data service."""
        if self._candle_update_task:
            self._candle_update_task.cancel()
            try:
                await self._candle_update_task
            except asyncio.CancelledError:
                pass
            self._candle_update_task = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._initialized = False
        logger.info("Market data service closed")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_market_open(self, symbol: str) -> bool:
        """Check if the market is currently open for the given symbol."""
        now = datetime.now()
        
        # Determine market type
        if symbol in ["BTC/USD", "ETH/USD"]:
            market_type = "crypto"
        elif symbol in ["EUR/USD", "GBP/USD", "USD/JPY"]:
            market_type = "forex"
        else:
            market_type = "stocks"
        
        market_config = self._market_hours[market_type]
        
        # For forex and crypto, market is always open
        if market_type in ["forex", "crypto"]:
            return True
        
        # For stocks, check market hours
        market_open = datetime.strptime(market_config["open"], "%H:%M").time()
        market_close = datetime.strptime(market_config["close"], "%H:%M").time()
        current_time = now.time()
        
        return market_open <= current_time <= market_close

    def _get_volatility_for_timeframe(self, interval: str) -> float:
        """Get the appropriate volatility for the given timeframe."""
        # Convert interval to standard format
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30m",
            "1h": "1h",
            "4h": "4h",
            "1day": "1day"
        }
        
        standard_interval = interval_map.get(interval, "1day")
        return self._volatility_profiles[standard_interval]

    async def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make a request to the Twelve Data API with rate limiting and error recovery."""
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        try:
            # Check if we should try real data again after rate limit reset
            if self._use_mock_data and self._rate_limit_reset_time:
                if datetime.now() >= self._rate_limit_reset_time:
                    logger.info("Rate limit reset time reached, attempting real data again")
                    self._use_mock_data = False
                    self._rate_limit_reset_time = None
                    self._retry_count = 0
                    MOCK_DATA_USAGE.labels(endpoint=endpoint).dec()

            # If we're using mock data due to rate limits, stay in mock mode
            if self._use_mock_data:
                MOCK_DATA_USAGE.labels(endpoint=endpoint).inc()
                logger.info(f"Using mock data for {endpoint} with params: {params}")
                return await self._get_mock_data(endpoint, params)

            # Check if market is open for the symbol
            symbol = params.get("symbol")
            if symbol and not self._is_market_open(symbol):
                logger.info(f"Market is closed for {symbol}, using mock data")
                return await self._get_mock_data(endpoint, params)

            # Ensure we respect rate limits
            now = datetime.now()
            time_since_last_request = now - self._last_request_time
            if time_since_last_request < self._min_request_interval:
                await asyncio.sleep((self._min_request_interval - time_since_last_request).total_seconds())

            async with self._request_semaphore:
                session = await self._get_session()
                params["apikey"] = self.api_key

                # Convert all datetime values in params to ISO strings
                for k, v in params.items():
                    if isinstance(v, datetime):
                        params[k] = v.isoformat()
                    
                try:
                    async with session.get(f"{self.base_url}/{endpoint}", params=params) as response:
                        if response.status == 429:  # Rate limit exceeded
                            logger.warning(f"Rate limit exceeded for {endpoint}, switching to mock data")
                            RATE_LIMIT_HITS.inc()
                            self._use_mock_data = True
                            self._rate_limit_reset_time = datetime.now() + timedelta(hours=1)
                            API_REQUESTS.labels(endpoint=endpoint, status='rate_limited').inc()
                            return await self._get_mock_data(endpoint, params)
                        
                        response.raise_for_status()
                        data = await response.json()
                        
                        if "status" in data and data["status"] == "error":
                            if "code" in data and data["code"] == 429:  # Rate limit error
                                logger.warning(f"Rate limit exceeded for {endpoint}, switching to mock data")
                                RATE_LIMIT_HITS.inc()
                                self._use_mock_data = True
                                self._rate_limit_reset_time = datetime.now() + timedelta(hours=1)
                                API_REQUESTS.labels(endpoint=endpoint, status='rate_limited').inc()
                                return await self._get_mock_data(endpoint, params)
                            raise MarketDataError(f"API error: {data.get('message', 'Unknown error')}")
                        
                        self._last_request_time = datetime.now()
                        self._retry_count = 0
                        API_REQUESTS.labels(endpoint=endpoint, status='success').inc()
                        return data
                except aiohttp.ClientError as e:
                    logger.error(f"API request failed for {endpoint}: {str(e)}")
                    self._retry_count += 1
                    
                    if "429" in str(e) or self._retry_count >= self._max_retries:
                        self._use_mock_data = True
                        self._rate_limit_reset_time = datetime.now() + timedelta(hours=1)
                        if "429" in str(e):
                            RATE_LIMIT_HITS.inc()
                            logger.warning(f"Rate limit exceeded for {endpoint}, switching to mock data")
                        else:
                            logger.warning(f"Max retries reached for {endpoint}, switching to mock data")
                    API_REQUESTS.labels(endpoint=endpoint, status='error').inc()
                    return await self._get_mock_data(endpoint, params)
        finally:
            ACTIVE_REQUESTS.dec()
            API_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)

    async def _get_mock_data(self, endpoint: str, params: Dict) -> Dict:
        """Generate mock market data for testing."""
        logger.info(f"Using mock data for {endpoint}")
        MOCK_DATA_USAGE.labels(endpoint=endpoint).inc()
        
        if endpoint == "time_series":
            symbol = params.get("symbol", "AAPL")
            interval = params.get("interval", "1day")
            return self._generate_mock_time_series(symbol, interval)
        elif endpoint == "quote":
            symbol = params.get("symbol", "AAPL")
            return self._generate_mock_quote(symbol)
        else:
            raise MarketDataError(f"Unsupported endpoint for mock data: {endpoint}")

    def _generate_mock_time_series(self, symbol: str, interval: str) -> Dict:
        """Generate mock time series data with realistic price movements."""
        now = datetime.now()
        candles = []
        
        # Initialize base price based on symbol
        if symbol in ["BTC/USD", "ETH/USD"]:
            base_price = random.uniform(20000, 50000) if symbol == "BTC/USD" else random.uniform(1000, 3000)
        elif symbol in ["EUR/USD", "GBP/USD", "USD/JPY"]:
            base_price = random.uniform(1.0, 1.5) if symbol != "USD/JPY" else random.uniform(100, 150)
        else:
            base_price = random.uniform(100, 200)
        
        # Get appropriate volatility for the timeframe
        volatility = self._get_volatility_for_timeframe(interval)
        trend = random.uniform(-0.001, 0.001)  # Slight trend
        
        # Generate price series with realistic movements
        prices = []
        # Allow override of number of candles via environment variable for testing
        num_candles = int(os.environ.get("MOCK_NUM_CANDLES", 30))
        # Adjust number of candles based on interval if not overridden
        if "MOCK_NUM_CANDLES" not in os.environ:
            if interval in ["1min", "5min"]:
                num_candles = 100
            elif interval in ["15min", "30m"]:
                num_candles = 50
        
        for i in range(num_candles):
            # Add random walk with drift
            price_change = np.random.normal(trend, volatility)
            base_price *= (1 + price_change)
            prices.append(base_price)
        
        # Generate OHLC data
        for i, price in enumerate(prices):
            # Calculate timestamp based on interval
            if interval == "1min":
                date = now - timedelta(minutes=i)
            elif interval == "5min":
                date = now - timedelta(minutes=i*5)
            elif interval == "15min":
                date = now - timedelta(minutes=i*15)
            elif interval == "30m":
                date = now - timedelta(minutes=i*30)
            elif interval == "1h":
                date = now - timedelta(hours=i)
            elif interval == "4h":
                date = now - timedelta(hours=i*4)
            else:  # 1day
                date = now - timedelta(days=i)
            
            open_price = price
            high_price = open_price * (1 + random.uniform(0, volatility))
            low_price = open_price * (1 - random.uniform(0, volatility))
            close_price = (high_price + low_price) / 2
            
            # Adjust volume based on price movement and market hours
            base_volume = random.uniform(1000000, 5000000)
            if not self._is_market_open(symbol):
                base_volume *= 0.1  # Reduce volume outside market hours
            volume = int(base_volume * (1 + abs(price_change)))
            
            candles.append({
                "datetime": date.strftime("%Y-%m-%d %H:%M:%S"),
                "open": str(round(open_price, 2)),
                "high": str(round(high_price, 2)),
                "low": str(round(low_price, 2)),
                "close": str(round(close_price, 2)),
                "volume": str(volume)
            })
        
        return {
            "meta": {
                "symbol": symbol,
                "interval": interval,
                "currency": "USD",
                "exchange_timezone": "America/New_York",
                "exchange": "NASDAQ",
                "type": "Common Stock"
            },
            "values": candles
        }

    def _generate_mock_quote(self, symbol: str) -> Dict:
        """Generates a mock quote for a given symbol."""
        mock_price = round(random.uniform(100.0, 2000.0), 2)  # Simulate a price
        mock_volume = random.randint(10000, 1000000)
        return {
            "symbol": symbol,
            "name": f"{symbol} Mock Stock",
            "exchange": "MOCK",
            "mic_code": "XMOS",
            "currency": symbol.split("/")[-1] if "/" in symbol else "USD",
            "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "open": mock_price * 0.99,
            "high": mock_price * 1.01,
            "low": mock_price * 0.98,
            "close": mock_price,  # Ensure 'close' key is always present
            "volume": mock_volume,
            "previous_close": mock_price * 0.995,
            "change": mock_price - (mock_price * 0.995),
            "percent_change": (mock_price - (mock_price * 0.995)) / (mock_price * 0.995) * 100,
            "fifty_two_week": {
                "low": mock_price * 0.8,
                "high": mock_price * 1.2,
                "change": 0.0,
                "change_percent": 0.0
            }
        }

    async def get_stock_price(self, symbol: str) -> float:
        """Get the current stock price for a given symbol."""
        try:
            data = await self._make_request("quote", {"symbol": symbol})
            return float(data["close"])
        except (KeyError, ValueError) as e:
            raise MarketDataError(f"Failed to parse stock price: {str(e)}")

    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "1day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get historical price data for a given symbol."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": "30"
        }

        # Set default date range if not provided
        if not start_date and not end_date:
            end = datetime.utcnow()
            # Estimate 30 bars back for the interval
            if interval.endswith("h"):
                try:
                    hours = int(interval[:-1]) * 30
                except Exception:
                    hours = 30
                start = end - timedelta(hours=hours)
            elif interval.endswith("m"):
                try:
                    minutes = int(interval[:-1]) * 30
                except Exception:
                    minutes = 30
                start = end - timedelta(minutes=minutes)
            else:
                start = end - timedelta(days=30)
            params["start_date"] = start.strftime("%Y-%m-%d %H:%M:%S")
            params["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        else:
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

        try:
            data = await self._make_request("time_series", params)
            return data["values"]
        except KeyError as e:
            raise MarketDataError(f"Failed to parse historical data: {str(e)}")

    async def get_live_price(self, pair: str) -> float:
        """Get the latest price for a trading pair with robust failover."""
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported trading pair: {pair}")

        # Check cache first
        if pair in self._price_cache:
            last_update = self._last_update.get(pair)
            if last_update and (datetime.utcnow() - last_update).total_seconds() <= self._max_latency:
                return self._price_cache[pair]
        
        # If mock/demo mode is active, always use mock data
        if self._use_mock_data:
            logger.warning(f"[MarketData] Using mock data for {pair} (MOCK_MARKET_DATA enabled)")
            MOCK_DATA_USAGE.labels(endpoint="get_live_price").inc()
            price = float(self._generate_mock_quote(pair)["close"])
            self._price_cache[pair] = price
            self._last_update[pair] = datetime.utcnow()
            self._last_data_source = "mock"
            return price

        # Try primary provider (TwelveData)
        try:
            price = await self._fetch_live_price(pair)
            self._price_cache[pair] = price
            self._last_update[pair] = datetime.utcnow()
            self._last_data_source = "twelvedata"
            await update_price(pair, price)
            await callback_manager.trigger('price_update', pair=pair, price=price)
            logger.info(f"Fetched live price for {pair} from TwelveData: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching live price for {pair} from TwelveData: {str(e)}. Trying Alpha Vantage...")
            # Try Alpha Vantage if API key is set
            alpha_key = os.getenv("ALPHAVANTAGE_API_KEY")
            if alpha_key:
                try:
                    # Alpha Vantage fallback logic (pseudo, replace with real call)
                    # price = await self._fetch_alpha_vantage_price(pair, alpha_key)
                    # For now, just log and skip
                    logger.warning(f"[MarketData] Alpha Vantage fallback not implemented for {pair}")
                except Exception as av_e:
                    logger.error(f"Alpha Vantage error for {pair}: {av_e}")
            # Try Yahoo Finance/Binance as last fallback
            try:
                if pair in ["XAU/USD", "GBP/USD", "GBP/JPY", "EUR/USD", "USD/JPY"]:
                    yf_symbol = pair.replace("/", "") + "=X" if pair != "XAU/USD" else "XAUUSD=X"
                    ticker = yf.Ticker(yf_symbol)
                    price = ticker.history(period="1d").tail(1)["Close"].iloc[0]
                    self._price_cache[pair] = price
                    self._last_update[pair] = datetime.utcnow()
                    self._last_data_source = "yfinance"
                    logger.info(f"Fetched live price for {pair} from Yahoo Finance: {price}")
                    return price
                elif pair in ["BTC/USD", "ETH/USD"]:
                    binance_symbol = pair.replace("/", "") + "T"  # e.g., BTCUSDT
                    client = BinanceClient()
                    ticker = client.get_symbol_ticker(symbol=binance_symbol)
                    price = float(ticker["price"])
                    self._price_cache[pair] = price
                    self._last_update[pair] = datetime.utcnow()
                    self._last_data_source = "binance"
                    logger.info(f"Fetched live price for {pair} from Binance: {price}")
                    return price
                else:
                    raise HTTPException(status_code=500, detail=f"No fallback source for {pair}")
            except Exception as fallback_e:
                logger.error(f"Fallback error fetching live price for {pair}: {str(fallback_e)}")
                raise HTTPException(status_code=500, detail="Failed to fetch live price from all sources")

    async def get_candle_data(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get candle data for a trading pair with robust failover."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        # If mock/demo mode is active, always use mock data
        if self._use_mock_data:
            logger.warning(f"[MarketData] Using mock candle data for {pair} (MOCK_MARKET_DATA enabled)")
            MOCK_DATA_USAGE.labels(endpoint="get_candle_data").inc()
            df = pd.DataFrame(self._generate_mock_time_series(pair, timeframe)["values"])
            self._last_data_source = "mock"
            return df
        # Try primary provider (TwelveData)
        try:
            df = await self._fetch_candle_data(pair, timeframe, limit)
            self._last_data_source = "twelvedata"
            return df
        except Exception as e:
            logger.error(f"Error fetching candle data for {pair} from TwelveData: {str(e)}. Trying Alpha Vantage...")
            alpha_key = os.getenv("ALPHAVANTAGE_API_KEY")
            if alpha_key:
                try:
                    # Alpha Vantage fallback logic (pseudo, replace with real call)
                    # df = await self._fetch_alpha_vantage_candles(pair, timeframe, limit, alpha_key)
                    logger.warning(f"[MarketData] Alpha Vantage fallback not implemented for {pair}")
                except Exception as av_e:
                    logger.error(f"Alpha Vantage error for {pair}: {av_e}")
            # Try Yahoo Finance/Binance as last fallback
            try:
                if pair in ["XAU/USD", "GBP/USD", "GBP/JPY", "EUR/USD", "USD/JPY"]:
                    yf_symbol = pair.replace("/", "") + "=X" if pair != "XAU/USD" else "XAUUSD=X"
                    ticker = yf.Ticker(yf_symbol)
                    hist = ticker.history(period="1mo")
                    df = pd.DataFrame({
                        "timestamp": hist.index,
                        "open": hist["Open"],
                        "high": hist["High"],
                        "low": hist["Low"],
                        "close": hist["Close"],
                        "volume": hist["Volume"]
                    })
                    self._last_data_source = "yfinance"
                    logger.info(f"Fetched candle data for {pair} from Yahoo Finance")
                    return df
                elif pair in ["BTC/USD", "ETH/USD"]:
                    binance_symbol = pair.replace("/", "") + "T"
                    client = BinanceClient()
                    klines = client.get_klines(symbol=binance_symbol, interval="1h", limit=limit)
                    df = pd.DataFrame(klines, columns=[
                        "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ])
                    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    self._last_data_source = "binance"
                    logger.info(f"Fetched candle data for {pair} from Binance")
                    return df
                else:
                    raise HTTPException(status_code=500, detail=f"No fallback source for {pair}")
            except Exception as fallback_e:
                logger.error(f"Fallback error fetching candle data for {pair}: {str(fallback_e)}")
                raise HTTPException(status_code=500, detail="Failed to fetch candle data from all sources")

    async def subscribe_to_live_price(self, pair: str, callback):
        """Subscribe to live price updates for a trading pair."""
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported trading pair: {pair}")
        
        self._live_price_subscribers.add((pair, callback))
        if pair not in self._ws_connections:
            await self._setup_websocket_connection(pair)

    async def subscribe_to_candles(self, pair: str, timeframe: str, callback):
        """Subscribe to candle updates for a trading pair."""
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported trading pair: {pair}")
        
        self._candle_subscribers.add((pair, timeframe, callback))
        if pair not in self._ws_connections:
            await self._setup_websocket_connection(pair)

    async def _update_candles_periodically(self):
        """Periodically update candle data for all subscribed pairs."""
        while True:
            try:
                for pair, timeframe, callback in self._candle_subscribers:
                    df = await self.get_candle_data(pair, timeframe)
                    await callback(df)
                
                # Update live prices
                for pair, callback in self._live_price_subscribers:
                    price = await self.get_live_price(pair)
                    await callback(price)
                
                await asyncio.sleep(self._update_interval)
            except Exception as e:
                logger.error(f"Error in periodic update: {str(e)}")
                await asyncio.sleep(self._update_interval)

    async def _setup_websocket_connection(self, pair: str):
        """Set up WebSocket connection for a trading pair."""
        try:
            # Implementation depends on your WebSocket provider
            # This is a placeholder for the actual implementation
            pass
        except Exception as e:
            logger.error(f"Error setting up WebSocket for {pair}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to set up WebSocket connection")

    async def _fetch_live_price(self, pair: str) -> float:
        symbol = pair  # e.g., "BTC/USD"
        url = f"{self.base_url}/price"
        params = {"symbol": symbol, "apikey": self.api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if "price" in data:
                    return float(data["price"])
                else:
                    raise Exception(f"TwelveData error: {data}")

    async def _fetch_candle_data(self, pair: str, interval: str, limit: int) -> pd.DataFrame:
        """Fetch candle data from TwelveData API."""
        symbol = pair  # e.g., "BTC/USD"
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    
                    if "values" in data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data["values"])
                        
                        # TwelveData returns: datetime, open, high, low, close, volume
                        # Check the actual number of columns and map accordingly
                        if len(df.columns) == 6:
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        elif len(df.columns) == 5:
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                            df['volume'] = 0  # Add volume column if not present
                        else:
                            logger.error(f"Unexpected number of columns in TwelveData response: {len(df.columns)}")
                            return pd.DataFrame()
                        
                        # Convert types
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        
                        return df
                    else:
                        logger.error(f"TwelveData error fetching candle data: {data}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching candle data for {pair}: {str(e)}")
            return pd.DataFrame()

    async def broadcast_price_update(self, pair: str, price: float):
        """Broadcasts a live price update to connected WebSocket clients."""
        message = {"type": "market_data", "symbol": pair, "price": price, "timestamp": datetime.utcnow().isoformat()}
        await self.websocket_manager.broadcast_message(pair, message)

    def _start_fallback_polling(self, pair: str):
        """Starts polling for market data if WebSocket connection fails."""
        if self._fallback_attempts[pair] >= self._max_fallback_attempts:
            logger.warning(f"Max fallback attempts reached for {pair}. Stopping polling.")
            return

        if pair not in self._fallback_polling or self._fallback_polling[pair].done():
            logger.info(f"Starting fallback polling for {pair}...")
            self._fallback_attempts[pair] += 1
            loop = asyncio.get_event_loop()
            self._fallback_polling[pair] = loop.create_task(self._poll_price_updates(pair))

    async def _poll_price_updates(self, pair: str):
        """Polls for price updates and broadcasts them via WebSocket."""
        while True:
            try:
                price = await self.get_live_price(pair)
                await self.websocket_manager.broadcast_market_data_update(pair, price)
                await asyncio.sleep(self._fallback_interval)  # Poll every X seconds
            except Exception as e:
                logger.error(f"Error during fallback polling for {pair}: {e}")
                # If polling also fails, consider stopping after a few retries or notify an admin
                await asyncio.sleep(self._fallback_interval * 2) # Wait longer on error

    def _stop_fallback_polling(self, pair: str):
        """Stops fallback polling for a given pair."""
        if pair in self._fallback_polling and not self._fallback_polling[pair].done():
            self._fallback_polling[pair].cancel()
            logger.info(f"Stopped fallback polling for {pair}.")

    def register_websocket_client(self, pair: str, client: Any):
        """Registers a new WebSocket client for a given pair."""
        if pair not in self._ws_connections:
            self._ws_connections[pair] = []
        self._ws_connections[pair].append(client)
        logger.info(f"Registered WebSocket client for {pair}. Total clients: {len(self._ws_connections[pair])}")
        # If this is the first client, start real-time data fetching
        if len(self._ws_connections[pair]) == 1 and not settings.MOCK_MARKET_DATA:
            # In a real scenario, you'd connect to a live data feed (e.g., Binance WebSocket)
            # For now, we rely on polling or the existing signal automation to generate data
            logger.info(f"Starting live data fetching for {pair} (placeholder).")

    def unregister_websocket_client(self, pair: str, client: Any):
        """Unregisters a WebSocket client for a given pair."""
        if pair in self._ws_connections and client in self._ws_connections[pair]:
            self._ws_connections[pair].remove(client)
            logger.info(f"Unregistered WebSocket client for {pair}. Total clients: {len(self._ws_connections[pair])}")
            if not self._ws_connections[pair]:
                del self._ws_connections[pair]
                logger.info(f"No more WebSocket clients for {pair}. Stopping live data fetching.")
                self._stop_fallback_polling(pair) # Stop polling if no more clients

    def reset_mock_mode(self):
        """Reset the service to use real data again."""
        self._use_mock_data = False
        self._rate_limit_reset_time = None
        self._retry_count = 0
        logger.info("Market data service reset to use real data")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a given DataFrame."""
        if df.empty or 'close' not in df.columns:
            return df

        # Ensure columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values that result from indicator calculations
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        if df.empty:
            return df

        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

        # MACD
        macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # EMA (50-period)
        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

        # Volume for plotting (ensure it's numeric)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        else:
            df['volume'] = 0.0 # Default to 0 if no volume data
            
        return df.fillna(0) # Fill any remaining NaNs with 0

    async def broadcast_candle_data(self, pair: str, timeframe: str, candles_df: pd.DataFrame) -> None:
        """Broadcasts comprehensive candle data with indicators to subscribed clients."""
        if self.websocket_manager is None:
            logger.error("WebSocket manager is not initialized in MarketDataService.")
            return

        if candles_df.empty:
            logger.warning(f"No candle data to broadcast for {pair} {timeframe}")
            return

        # Prepare data for JSON serialization
        # Convert DataFrame to a list of dictionaries
        # Ensure timestamps are ISO format strings and NaNs are handled
        candles_data = candles_df.replace({np.nan: None}).to_dict(orient='records')
        for item in candles_data:
            if 'timestamp' in item and isinstance(item['timestamp'], pd.Timestamp):
                item['timestamp'] = item['timestamp'].isoformat()

        message = {
            "type": "candle_data",
            "pair": pair,
            "timeframe": timeframe,
            "data": {
                "timestamps": [item['timestamp'] for item in candles_data],
                "open": [item['open'] for item in candles_data],
                "high": [item['high'] for item in candles_data],
                "low": [item['low'] for item in candles_data],
                "close": [item['close'] for item in candles_data],
                "volume": [item['volume'] for item in candles_data],
                "rsi": [item['rsi'] for item in candles_data if 'rsi' in item],
                "macd": [item['macd'] for item in candles_data if 'macd' in item],
                "macd_signal": [item['macd_signal'] for item in candles_data if 'macd_signal' in item],
                "macd_diff": [item['macd_diff'] for item in candles_data if 'macd_diff' in item],
                "ema_50": [item['ema_50'] for item in candles_data if 'ema_50' in item],
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await self.websocket_manager.broadcast_message(pair, message)
            logger.debug(f"Broadcasted candle data for {pair} {timeframe}")
        except Exception as e:
            logger.error(f"Error broadcasting candle data for {pair} {timeframe}: {str(e)}")

    def set_websocket_manager(self, manager: "WebSocketManager"):
        self.websocket_manager = manager

    def _validate_pair(self, pair: str):
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported trading pair: {pair}")


# Create singleton instance
market_data_service = MarketDataService()

async def initialize():
    """Initialize the market data service."""
    await market_data_service.initialize()

async def close():
    """Close the market data service."""
    await market_data_service.close()

async def get_available_pairs() -> List[str]:
    """Get list of available trading pairs"""
    return list(SUPPORTED_PAIRS.keys())
