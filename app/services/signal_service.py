from typing import List, Dict, Optional
import logging
import asyncio
from datetime import datetime
import random  # For demo purposes, replace with real market data

logger = logging.getLogger(__name__)

class SignalService:
    def __init__(self):
        self.supported_pairs = ["BTC/USD"]
        self.price_cache: Dict[str, float] = {}
        self._price_update_task = None
        
    async def start(self):
        """Start the price update service."""
        if not self._price_update_task:
            self._price_update_task = asyncio.create_task(self._update_prices())
            logger.info("Signal service started")
            
    async def stop(self):
        """Stop the price update service."""
        if self._price_update_task:
            self._price_update_task.cancel()
            try:
                await self._price_update_task
            except asyncio.CancelledError:
                pass
            self._price_update_task = None
            logger.info("Signal service stopped")
            
    async def _update_prices(self):
        """Background task to update prices periodically."""
        while True:
            try:
                for pair in self.supported_pairs:
                    # Simulate price updates (replace with real market data)
                    current_price = self.price_cache.get(pair, 0)
                    change = random.uniform(-0.01, 0.01) * current_price
                    new_price = current_price + change if current_price else random.uniform(1000, 100000)
                    self.price_cache[pair] = new_price
                    
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating prices: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a trading pair."""
        return self.price_cache.get(pair)
        
    def get_all_prices(self) -> Dict[str, float]:
        """Get current prices for all supported pairs."""
        return self.price_cache.copy()
        
    async def generate_signal(self, pair: str) -> Dict:
        """Generate a trading signal for the given pair."""
        current_price = self.get_current_price(pair)
        if not current_price:
            return None
            
        # Simulate signal generation (replace with real signal logic)
        signal_type = random.choice(["BUY", "SELL"])
        confidence = random.uniform(0.5, 0.95)
        
        return {
            "pair": pair,
            "type": signal_type,
            "price": current_price,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def get_signals(self, pairs: List[str]) -> List[Dict]:
        """Get signals for multiple pairs."""
        signals = []
        for pair in pairs:
            if pair in self.supported_pairs:
                signal = await self.generate_signal(pair)
                if signal:
                    signals.append(signal)
        return signals 