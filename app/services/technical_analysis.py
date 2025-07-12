import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from app.schemas.signal_schema import SignalDirection

@dataclass
class SupportResistanceLevel:
    price: float
    strength: int  # Number of touches
    type: str  # 'support' or 'resistance'
    last_touch: int  # Index of last touch

class TechnicalAnalysis:
    def __init__(self):
        self._min_touches = 2  # Minimum number of touches for S/R level
        self._fractal_deviation = 0.05  # 5% deviation for fractal detection
        self._trend_leg_candles = 20  # Number of candles for trend leg detection

    def find_micro_support_resistance(self, df: pd.DataFrame) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Find micro support and resistance levels based on recent price action."""
        supports = []
        resistances = []
        
        # Get recent candles
        recent_df = df.tail(20)
        
        # Find swing highs and lows
        highs = self._find_swing_highs(recent_df)
        lows = self._find_swing_lows(recent_df)
        
        # Group nearby levels
        support_levels = self._group_price_levels(lows, recent_df['low'].min(), recent_df['low'].max())
        resistance_levels = self._group_price_levels(highs, recent_df['high'].min(), recent_df['high'].max())
        
        # Create SupportResistanceLevel objects
        for level in support_levels:
            if level['touches'] >= self._min_touches:
                supports.append(SupportResistanceLevel(
                    price=level['price'],
                    strength=level['touches'],
                    type='support',
                    last_touch=level['last_touch']
                ))
        
        for level in resistance_levels:
            if level['touches'] >= self._min_touches:
                resistances.append(SupportResistanceLevel(
                    price=level['price'],
                    strength=level['touches'],
                    type='resistance',
                    last_touch=level['last_touch']
                ))
        
        return supports, resistances

    def detect_trend_leg(self, df: pd.DataFrame) -> Optional[SignalDirection]:
        """Detect trend leg based on higher highs/lows or lower highs/lows."""
        # Get recent candles
        recent_df = df.tail(self._trend_leg_candles)
        
        # Calculate EMA
        recent_df['ema20'] = recent_df['close'].ewm(span=20).mean()
        
        # Find swing highs and lows
        highs = self._find_swing_highs(recent_df)
        lows = self._find_swing_lows(recent_df)
        
        # Check for higher highs and higher lows
        if len(highs) >= 2 and len(lows) >= 2:
            higher_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
            higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
            
            if higher_highs and higher_lows:
                return SignalDirection.BUY
        
        # Check for lower highs and lower lows
        if len(highs) >= 2 and len(lows) >= 2:
            lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
            lower_lows = all(lows[i] < lows[i-1] for i in range(1, len(lows)))
            
            if lower_highs and lower_lows:
                return SignalDirection.SELL
        
        return None

    def _find_swing_highs(self, df: pd.DataFrame) -> List[float]:
        """Find swing highs using fractal logic."""
        highs = []
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                highs.append(df['high'].iloc[i])
        return highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[float]:
        """Find swing lows using fractal logic."""
        lows = []
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                lows.append(df['low'].iloc[i])
        return lows

    def _group_price_levels(self, levels: List[float], min_price: float, max_price: float) -> List[Dict]:
        """Group nearby price levels and count touches."""
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Calculate price range for grouping
        price_range = max_price - min_price
        group_threshold = price_range * self._fractal_deviation
        
        # Group levels
        grouped_levels = []
        current_group = {
            'price': sorted_levels[0],
            'touches': 1,
            'last_touch': 0
        }
        
        for i in range(1, len(sorted_levels)):
            if abs(sorted_levels[i] - current_group['price']) <= group_threshold:
                current_group['price'] = (current_group['price'] * current_group['touches'] + sorted_levels[i]) / (current_group['touches'] + 1)
                current_group['touches'] += 1
                current_group['last_touch'] = i
            else:
                grouped_levels.append(current_group)
                current_group = {
                    'price': sorted_levels[i],
                    'touches': 1,
                    'last_touch': i
                }
        
        grouped_levels.append(current_group)
        return grouped_levels

    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        diff = high - low
        return {
            '0.0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1.0': high
        }

    def find_zigzag_points(self, df: pd.DataFrame, deviation: float = 0.05) -> Tuple[List[float], List[float]]:
        """Find zigzag swing points based on percentage deviation."""
        highs = []
        lows = []
        last_high = df['high'].iloc[0]
        last_low = df['low'].iloc[0]
        
        for i in range(1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check for new high
            if current_high > last_high * (1 + deviation):
                highs.append(current_high)
                last_high = current_high
                last_low = current_low
            
            # Check for new low
            elif current_low < last_low * (1 - deviation):
                lows.append(current_low)
                last_low = current_low
                last_high = current_high
        
        return highs, lows

# Create singleton instance
technical_analysis = TechnicalAnalysis() 