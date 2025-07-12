from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Shared constants
SUPPORTED_PAIRS = [
    "XAU/USD", "GBP/USD", "GBP/JPY", "EUR/USD", "USD/JPY", "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "BNB/USD", "DOGE/USD", "XRP/USD", "LTC/USD", "DOT/USD", "AVAX/USD", "SHIB/USD", "MATIC/USD", "TRX/USD", "LINK/USD", "UNI/USD", "BCH/USD", "EOS/USD", "ATOM/USD", "XMR/USD", "ETC/USD", "FIL/USD", "XTZ/USD", "AAVE/USD", "NEO/USD", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/CAD", "EUR/AUD", "EUR/NZD", "GBP/CHF", "GBP/AUD", "GBP/NZD", "AUD/JPY", "AUD/NZD", "AUD/CAD", "AUD/CHF", "NZD/JPY", "NZD/CAD", "NZD/CHF", "CAD/JPY", "CHF/JPY"
]

# Shared cache
price_cache = {}
candle_cache = {}

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

def get_available_pairs() -> list:
    return SUPPORTED_PAIRS 