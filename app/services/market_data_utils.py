from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Shared cache
price_cache = {}
candle_cache = {}

SUPPORTED_PAIRS = [
    "XAU/USD",
    "GBP/USD",
    "GBP/JPY",
    "EUR/USD",
    "USD/JPY",
    "BTC/USD",
    "ETH/USD",
]

async def get_market_data(pair: str) -> Optional[Dict[str, Any]]:
    """Get market data for a pair."""
    if pair in candle_cache:
        return candle_cache[pair]
    return None

async def update_market_data(pair: str, data: Dict[str, Any]) -> None:
    """Update market data for a pair."""
    candle_cache[pair] = data

async def get_price(pair: str) -> Optional[float]:
    """Get current price for a pair."""
    if pair in price_cache:
        return price_cache[pair]
    return None

async def update_price(pair: str, price: float) -> None:
    """Update price for a pair."""
    price_cache[pair] = price

async def get_available_pairs() -> List[str]:
    """Get list of available trading pairs."""
    return SUPPORTED_PAIRS