import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
from app.services.market_data import market_data_service, SIGNAL_TYPE_TO_INTERVAL
from app.services.market_data_utils import SUPPORTED_PAIRS
from app.models.signal import Signal as SignalModel, SignalType, SignalDirection, SignalStatus, SignalTimeframe, SignalSetup
from typing import Optional, Tuple, List, Dict, Any
from app.config import settings
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.user import User
from fastapi import HTTPException
from app.schemas.signal_schema import SignalCreate
from app.services.websocket import websocket_manager
from app.core.notifier import notifier
import asyncio


logger = logging.getLogger(__name__)

# Risk:Reward configurations for each trading style
TRADING_CONFIGS = {
    SignalType.SCALPING: {
        "tp1_pips": 30,
        "tp2_pips": 60,
        "sl_pips": 50,
        "timeframes": [SignalTimeframe.M5, SignalTimeframe.M15],
        "indicators": {
            "ema_fast": 50,
            "ema_slow": 200,
            "volume_avg_period": 10
        },
        "setup_type": "Breakout + Pullback Confirmation",
        "required_conditions": [
            "Trend Filter: 50 EMA > 200 EMA",
            "Breakout Candle: Strong candle close above recent micro S/R",
            "Volume Spike: Above 10-bar avg volume",
            "Entry Signal: Pullback to broken level + bullish confirmation wick",
            "MACD Confirmation: Cross or histogram above 0"
        ]
    },
    SignalType.INTRADAY: {
        "tp1_pips": 50,
        "tp2_pips": 100,
        "sl_pips": 80,
        "timeframes": [SignalTimeframe.M30, SignalTimeframe.H1],
        "indicators": {
            "ema_period": 50,
            "fib_levels": [0.382, 0.5, 0.618]
        },
        "setup_type": "Trend Continuation + Fibonacci + S/R",
        "required_conditions": [
            "Trend: 50 EMA slope",
            "Entry Zone: 38.2–61.8 fib pullback on prior swing",
            "Confirmation: Inside bar → breakout",
            "RSI above 50 or MACD agreement"
        ]
    },
    SignalType.SWING: {
        "tp1_pips": 150,
        "tp2_pips": 300,
        "sl_pips": 200,
        "timeframes": [SignalTimeframe.H4, SignalTimeframe.D1],
        "indicators": {
            "ema_period": 200,
            "weekly_sr_lookback": 20
        },
        "setup_type": "Major Breakout + Retest",
        "required_conditions": [
            "Weekly S/R zone",
            "Trendline or 200 EMA test",
            "Reversal Candle (Pin bar, Engulfing)",
            "Divergence (RSI, MACD)"
        ]
    }
}

# News events to filter
MAJOR_NEWS_EVENTS = [
    "NFP",  # Non-Farm Payrolls
    "CPI",  # Consumer Price Index
    "FOMC", # Federal Open Market Committee
    "ECB",  # European Central Bank
    "BOE",  # Bank of England
    "BOJ",  # Bank of Japan
    "RBA",  # Reserve Bank of Australia
    "BOC"   # Bank of Canada
]

def calculate_pips(pair: str, price: float) -> float:
    """Calculate pip value based on pair type."""
    if "JPY" in pair:
        return price * 0.01  # 0.01 for JPY pairs
    return price * 0.0001  # 0.0001 for other pairs

def calculate_targets(pair: str, entry: float, direction: SignalDirection, 
                     signal_type: SignalType) -> Dict[str, float]:
    """Calculate TP1, TP2, and SL based on trading style specifications."""
    
    # Get pip value for the pair
    pip_value = calculate_pips(pair, entry)
    
    # Trading style specific configurations
    STYLE_TARGETS = {
        SignalType.SCALPING: {
            "tp1_pips": 30,
            "tp2_pips": 60, 
            "sl_pips": 50
        },
        SignalType.INTRADAY: {
            "tp1_pips": 50,
            "tp2_pips": 100,
            "sl_pips": 80
        },
        SignalType.SWING: {
            "tp1_pips": 150,
            "tp2_pips": 300,
            "sl_pips": 200
        }
    }
    
    config = STYLE_TARGETS.get(signal_type, STYLE_TARGETS[SignalType.INTRADAY])
    
    if direction == SignalDirection.BUY:
        tp1 = entry + (config["tp1_pips"] * pip_value)
        tp2 = entry + (config["tp2_pips"] * pip_value)
        sl = entry - (config["sl_pips"] * pip_value)
    else:  # SELL
        tp1 = entry - (config["tp1_pips"] * pip_value)
        tp2 = entry - (config["tp2_pips"] * pip_value)
        sl = entry + (config["sl_pips"] * pip_value)
        
    return {
        "tp1": round(tp1, 5),
        "tp2": round(tp2, 5),
        "sl": round(sl, 5)
    }

def detect_micro_sr(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    """Detect micro support and resistance levels."""
    recent_df = df.tail(lookback)
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(1, len(recent_df) - 1):
        # Check for swing high
        if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and 
            recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]):
            swing_highs.append(recent_df['high'].iloc[i])
        
        # Check for swing low
        if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and 
            recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]):
            swing_lows.append(recent_df['low'].iloc[i])
    
    # Find levels with at least 2 touches
    def find_touch_levels(levels, tolerance=0.001):
        touch_levels = {}
        for level in levels:
            found_match = False
            for existing_level in touch_levels:
                if abs(level - existing_level) / existing_level < tolerance:
                    touch_levels[existing_level] += 1
                    found_match = True
                    break
            if not found_match:
                touch_levels[level] = 1
        
        return [level for level, touches in touch_levels.items() if touches >= 2]
    
    resistance_levels = find_touch_levels(swing_highs)
    support_levels = find_touch_levels(swing_lows)
    
    sr_high = max(resistance_levels) if resistance_levels else df['high'].max()
    sr_low = min(support_levels) if support_levels else df['low'].min()
    
    return sr_high, sr_low

def check_scalping_setup(df: pd.DataFrame) -> Optional[Dict]:
    """Check for scalping setup based on requirements."""
    config = TRADING_CONFIGS[SignalType.SCALPING]
    
    # Calculate EMAs
    ema50 = EMAIndicator(close=df['close'], window=config["indicators"]["ema_fast"])
    ema200 = EMAIndicator(close=df['close'], window=config["indicators"]["ema_slow"])
    df['ema50'] = ema50.ema_indicator()
    df['ema200'] = ema200.ema_indicator()
    
    # Calculate MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Calculate volume
    df['volume_avg'] = df['volume'].rolling(window=config["indicators"]["volume_avg_period"]).mean()
    
    # Get last candle
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    # Detect micro S/R levels
    sr_high, sr_low = detect_micro_sr(df)
    
    # --- Bullish (BUY) setup ---
    if last_candle['close'] > last_candle['open']:
        if (df['ema50'].iloc[-1] > df['ema200'].iloc[-1] and  # Uptrend
            last_candle['high'] > sr_high and  # Breakout above resistance
            last_candle['close'] > last_candle['open'] and  # Strong bullish candle
            last_candle['volume'] > df['volume_avg'].iloc[-1] and  # Volume spike
            last_candle['low'] <= sr_high and  # Pullback to broken level
            df['macd_hist'].iloc[-1] > 0):  # MACD confirmation
            return {
                "direction": SignalDirection.BUY,
                "entry": last_candle['close'],
                "confidence": 0.8,
                "reason": "Breakout + Pullback with Volume and MACD Confirmation",
                "setup_type": config["setup_type"],
                "conditions_met": config["required_conditions"]
            }
    # --- Bearish (SELL) setup ---
    if last_candle['close'] < last_candle['open']:
        if (df['ema50'].iloc[-1] < df['ema200'].iloc[-1] and  # Downtrend
            last_candle['low'] < sr_low and  # Breakout below support
            last_candle['close'] < last_candle['open'] and  # Strong bearish candle
            last_candle['volume'] > df['volume_avg'].iloc[-1] and  # Volume spike
            last_candle['high'] >= sr_low and  # Pullback to broken level
            df['macd_hist'].iloc[-1] < 0):  # MACD confirmation
            return {
                "direction": SignalDirection.SELL,
                "entry": last_candle['close'],
                "confidence": 0.8,
                "reason": "Breakdown + Pullback with Volume and MACD Confirmation",
                "setup_type": config["setup_type"],
                "conditions_met": config["required_conditions"]
            }
    return None

def check_intraday_setup(df: pd.DataFrame) -> Optional[Dict]:
    # Alternate between BUY and SELL for demo
    if int(datetime.utcnow().timestamp()) % 2 == 0:
        direction = SignalDirection.BUY
    else:
        direction = SignalDirection.SELL
    return {
        "direction": direction,
        "entry": float(df['close'].iloc[-1]),
        "confidence": 0.99,
        "reason": "Demo signal (forced)",
        "setup_type": "Demo",
        "conditions_met": ["Always true for demo"]
    }

def check_swing_setup(df: pd.DataFrame) -> Optional[Dict]:
    """Check for swing setup based on requirements."""
    config = TRADING_CONFIGS[SignalType.SWING]
    
    # Calculate EMA
    ema = EMAIndicator(close=df['close'], window=config["indicators"]["ema_period"])
    df['ema'] = ema.ema_indicator()
    
    # Calculate RSI
    rsi = RSIIndicator(close=df['close'])
    df['rsi'] = rsi.rsi()
    
    # Calculate MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Get last candles
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    # Detect weekly S/R levels
    sr_high, sr_low = detect_micro_sr(df, lookback=config["indicators"]["weekly_sr_lookback"])
    
    # --- Bullish (BUY) setup ---
    if (last_candle['close'] > df['ema'].iloc[-1] and  # Above EMA
        last_candle['low'] <= df['ema'].iloc[-1] and   # Retested EMA
        last_candle['close'] > last_candle['open'] and # Bullish candle
        df['rsi'].iloc[-1] > 50):                      # RSI above 50
        
        # Check for reversal candle
        if (last_candle['close'] > last_candle['open'] and
            last_candle['close'] > prev_candle['close'] and
            last_candle['low'] < prev_candle['low'] and
            df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]):  # MACD agreement
            
            return {
                "direction": SignalDirection.BUY,
                "entry": last_candle['close'],
                "confidence": 0.85,
                "reason": "Major Breakout + EMA Retest with Reversal Candle",
                "setup_type": config["setup_type"],
                "conditions_met": config["required_conditions"]
            }
    
    # --- Bearish (SELL) setup ---
    if (last_candle['close'] < df['ema'].iloc[-1] and  # Below EMA
        last_candle['high'] >= df['ema'].iloc[-1] and  # Retested EMA from below
        last_candle['close'] < last_candle['open'] and # Bearish candle
        df['rsi'].iloc[-1] < 50):                      # RSI below 50
        # Check for reversal candle
        if (last_candle['close'] < last_candle['open'] and
            last_candle['close'] < prev_candle['close'] and
            last_candle['high'] > prev_candle['high'] and
            df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]):  # MACD agreement
            return {
                "direction": SignalDirection.SELL,
                "entry": last_candle['close'],
                "confidence": 0.85,
                "reason": "Major Breakdown + EMA Retest with Reversal Candle",
                "setup_type": config["setup_type"],
                "conditions_met": config["required_conditions"]
            }
    
    return None

async def check_news_filter() -> bool:
    """Check if current time is near major news events."""
    # TODO: Implement news filter logic for NFP, CPI, FOMC, etc.
    # This should check against a news calendar database
    return False

async def check_active_signals(db: Session, pair: str) -> bool:
    """Check if there are any active signals for the given pair."""
    result = await db.execute(
        select(SignalModel).where(
            SignalModel.pair == pair,
            SignalModel.status.in_([SignalStatus.ACTIVE, SignalStatus.TP1_HIT])
        )
    )
    return result.scalars().first() is not None

async def generate_signal(
    df: pd.DataFrame,
    pair: str,
    current_price: float,
    signal_type: SignalType,
    user_id: int,
    db: Session,
    timeframe: SignalTimeframe = SignalTimeframe.H1
) -> Optional[SignalModel]:
    """Generate a new trading signal based on the specified type with memory lock protection."""
    
    # Check memory lock to prevent repainting
    if await check_memory_lock(db, pair, signal_type):
        logger.info(f"Memory lock active for {pair} {signal_type.value}, skipping signal generation")
        return None

    # Check news filter
    if await check_news_filter():
        logger.info(f"Signal filtered due to news for {pair}")
        return None

    # Generate signal based on type
    setup = None
    if signal_type == SignalType.SCALPING:
        setup = check_scalping_setup(df)
    elif signal_type == SignalType.INTRADAY:
        setup = check_intraday_setup(df)
    elif signal_type == SignalType.SWING:
        setup = check_swing_setup(df)

    if not setup:
        return None

    # Calculate targets
    targets = calculate_targets(pair, setup["entry"], setup["direction"], signal_type)

    # Create signal
    signal = SignalModel(
        user_id=user_id,
        pair=pair,
        type=signal_type,
        timeframe=timeframe,
        direction=setup["direction"],
        entry_price=setup["entry"],
        tp1=targets["tp1"],
        tp2=targets["tp2"],
        stop_loss=targets["sl"],
        confidence=setup["confidence"],
        reason=setup["reason"],
        status=SignalStatus.ACTIVE,
        trading_style=signal_type.value,
        generated_time=datetime.utcnow(),
        logic_note=f"Setup Type: {setup['setup_type']}\nConditions Met:\n" + "\n".join(f"- {cond}" for cond in setup['conditions_met'])
    )

    # Save to database
    db.add(signal)
    await db.commit()
    await db.refresh(signal)

    # Convert to dict while session is still open
    signal_dict = signal.to_dict()

    # Create memory lock to prevent repainting
    await create_memory_lock(db, pair, signal_type, signal.id)

    # Broadcast signal
    logger.info(f"Broadcasting signal update: {signal_dict}")
    await websocket_manager.broadcast_signal_update(signal_dict)
    
    # Send notification
    await notifier.send_signal_notification(signal.to_dict() if hasattr(signal, 'to_dict') else signal)

    return signal

class SignalGenerationService:
    def __init__(self, db: Session):
        self.db = db
        self._initialized = False
        self._active_signals: Dict[str, List[SignalModel]] = {}

    async def initialize(self):
        """Initialize the signal generation service."""
        if not self._initialized:
            # Load active signals from database
            query = select(SignalModel).where(SignalModel.status == SignalStatus.ACTIVE)
            result = await self.db.execute(query)
            active_signals = result.scalars().all()
            
            # Group signals by pair
            for signal in active_signals:
                if signal.pair not in self._active_signals:
                    self._active_signals[signal.pair] = []
                self._active_signals[signal.pair].append(signal)
            
            self._initialized = True
            logger.info("Signal generation service initialized")

    async def generate_signal(
        self,
        pair: str,
        signal_type: SignalType,
        timeframe: SignalTimeframe,
        user_id: int
    ) -> SignalModel:
        """
        Generate a new trading signal.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            signal_type: Type of signal (SCALPING, INTRADAY, SWING)
            timeframe: Timeframe for the signal
            user_id: ID of the user requesting the signal
            
        Returns:
            SignalModel: Generated signal
            
        Raises:
            HTTPException: If signal generation fails
        """
        if pair not in SUPPORTED_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")

        if timeframe not in TRADING_CONFIGS[signal_type]["timeframes"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe {timeframe} for {signal_type} trading"
            )

        try:
            # Get market data
            candles = await market_data_service.get_candle_data(pair, timeframe, limit=100)
            if candles.empty:
                raise HTTPException(status_code=404, detail="No market data available")

            # Calculate indicators
            indicators = self._calculate_indicators(candles)
            
            # Check for setup conditions
            setup_conditions = self._check_setup_conditions(
                candles, indicators, signal_type, timeframe
            )
            
            if not setup_conditions["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid setup found: {setup_conditions['reason']}"
                )

            # Create signal
            signal = SignalModel(
                pair=pair,
                type=signal_type,
                timeframe=timeframe,
                direction=setup_conditions["direction"],
                entry_price=setup_conditions["entry_price"],
                tp1=setup_conditions["take_profit"],
                tp2=setup_conditions["take_profit"] * 1.5,  # Second target at 1.5x the first
                stop_loss=setup_conditions["stop_loss"],
                setup_conditions=setup_conditions["conditions"],
                user_id=user_id,
                status=SignalStatus.ACTIVE,
                created_at=datetime.utcnow()
            )

            # Save to database
            self.db.add(signal)
            await self.db.commit()
            await self.db.refresh(signal)

            # Add to active signals
            if pair not in self._active_signals:
                self._active_signals[pair] = []
            self._active_signals[pair].append(signal)

            # Broadcast signal
            logger.info(f"Broadcasting signal update: {signal.to_dict() if hasattr(signal, 'to_dict') else signal}")
            await websocket_manager.broadcast_signal_update(signal.to_dict())
            
            # Send notification
            await notifier.send_signal_notification(signal.to_dict() if hasattr(signal, 'to_dict') else signal)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']

        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        return {
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'signal': df['signal'].iloc[-1],
            'histogram': df['histogram'].iloc[-1],
            'volume_sma': df['volume_sma'].iloc[-1],
            'current_volume': df['volume'].iloc[-1]
        }

    def _check_setup_conditions(
        self,
        candles: pd.DataFrame,
        indicators: dict,
        signal_type: SignalType,
        timeframe: SignalTimeframe
    ) -> dict:
        """Check for valid setup conditions based on trading style."""
        
        # Trading style configurations
        STYLE_CONFIGS = {
            SignalType.SCALPING: {
                "timeframes": ["5m", "15m"],
                "tp1_pips": 30,
                "tp2_pips": 60,
                "sl_pips": 50,
                "setup_type": "breakout_pullback",
                "indicators": ["ema_50", "ema_200", "macd", "volume"]
            },
            SignalType.INTRADAY: {
                "timeframes": ["30m", "1h"],
                "tp1_pips": 50,
                "tp2_pips": 100,
                "sl_pips": 80,
                "setup_type": "trend_continuation_fib",
                "indicators": ["ema_50", "fibonacci", "rsi", "macd"]
            },
            SignalType.SWING: {
                "timeframes": ["4h", "1d"],
                "tp1_pips": 150,
                "tp2_pips": 300,
                "sl_pips": 200,
                "setup_type": "major_breakout_retest",
                "indicators": ["ema_200", "trendline", "rsi", "macd"]
            }
        }
        
        config = STYLE_CONFIGS.get(signal_type)
        if not config:
            return {"valid": False, "reason": "Invalid signal type"}
        
        # Check timeframe compatibility
        if timeframe.value.lower() not in config["timeframes"]:
            return {"valid": False, "reason": f"Timeframe {timeframe.value} not suitable for {signal_type.value}"}
        
        # Get current price and calculate pip values
        current_price = candles["close"].iloc[-1]
        pip_value = self._get_pip_value(current_price)
        
        # Apply trading style specific logic
        if signal_type == SignalType.SCALPING:
            return self._check_scalping_setup(candles, indicators, current_price, pip_value, config)
        elif signal_type == SignalType.INTRADAY:
            return self._check_intraday_setup(candles, indicators, current_price, pip_value, config)
        elif signal_type == SignalType.SWING:
            return self._check_swing_setup(candles, indicators, current_price, pip_value, config)
        
        return {"valid": False, "reason": "No valid setup found"}

    def _check_scalping_setup(self, candles: pd.DataFrame, indicators: dict, current_price: float, pip_value: float, config: dict) -> dict:
        """Check for Scalping setup conditions."""
        try:
            # Trend Filter: 50 EMA > 200 EMA → Uptrend only
            ema_50 = indicators.get("ema_50", [])
            ema_200 = indicators.get("ema_200", [])
            
            if len(ema_50) < 2 or len(ema_200) < 2:
                return {"valid": False, "reason": "Insufficient EMA data"}
            
            # Check uptrend condition
            if ema_50[-1] <= ema_200[-1]:
                return {"valid": False, "reason": "Not in uptrend (50 EMA <= 200 EMA)"}
            
            # Breakout detection
            recent_highs = candles["high"].rolling(window=10).max()
            recent_lows = candles["low"].rolling(window=10).min()
            
            # Check for breakout above recent resistance
            resistance_level = recent_highs.iloc[-2]  # Previous resistance
            breakout_candle = candles.iloc[-1]
            
            if breakout_candle["close"] <= resistance_level:
                return {"valid": False, "reason": "No breakout detected"}
            
            # Volume confirmation
            volume_avg = candles["volume"].rolling(window=10).mean().iloc[-1]
            if breakout_candle["volume"] <= volume_avg:
                return {"valid": False, "reason": "Insufficient volume for breakout"}
            
            # Pullback confirmation
            pullback_low = candles["low"].iloc[-1]
            if pullback_low > resistance_level:
                return {"valid": False, "reason": "No pullback to broken level"}
            
            # MACD confirmation
            macd = indicators.get("macd", [])
            if len(macd) >= 2:
                if macd[-1] <= 0 or macd[-1] <= macd[-2]:
                    return {"valid": False, "reason": "MACD not confirming bullish momentum"}
            
            # Calculate entry, TP, and SL
            entry_price = breakout_candle["close"]
            stop_loss = entry_price - (config["sl_pips"] * pip_value)
            tp1 = entry_price + (config["tp1_pips"] * pip_value)
            tp2 = entry_price + (config["tp2_pips"] * pip_value)

            return {
                "valid": True,
                "direction": "BUY",
                "entry_price": entry_price,
                "take_profit": tp1,
                "stop_loss": stop_loss,
                "tp2": tp2,
                "conditions": [
                    "50 EMA > 200 EMA (Uptrend)",
                    "Breakout above resistance",
                    "Volume spike confirmation",
                    "Pullback to broken level",
                    "MACD bullish confirmation"
                ]
            }
            
        except Exception as e:
           return {"valid": False, "reason": f"Scalping setup error: {str(e)}"}

    def _check_intraday_setup(self, candles: pd.DataFrame, indicators: dict, current_price: float, pip_value: float, config: dict) -> dict:
        """Check for Intraday setup conditions."""
        try:
            # Trend continuation with Fibonacci
            ema_50 = indicators.get("ema_50", [])
            if len(ema_50) < 20:
                return {"valid": False, "reason": "Insufficient EMA data"}
            
            # Check EMA slope for trend
            ema_slope = (ema_50[-1] - ema_50[-5]) / ema_50[-5]
            if ema_slope <= 0:
                return {"valid": False, "reason": "EMA slope not bullish"}
            
            # Fibonacci retracement levels
            swing_high = candles["high"].rolling(window=20).max().iloc[-1]
            swing_low = candles["low"].rolling(window=20).min().iloc[-1]
            
            fib_38_2 = swing_high - (swing_high - swing_low) * 0.382
            fib_61_8 = swing_high - (swing_high - swing_low) * 0.618
            
            # Check if price is in Fibonacci zone
            if not (fib_61_8 <= current_price <= fib_38_2):
                return {"valid": False, "reason": "Price not in Fibonacci retracement zone"}
            
            # Inside bar pattern
            prev_candle = candles.iloc[-2]
            current_candle = candles.iloc[-1]
            
            if not (current_candle["high"] <= prev_candle["high"] and current_candle["low"] >= prev_candle["low"]):
                return {"valid": False, "reason": "No inside bar pattern"}
            
            # RSI confirmation
            rsi = indicators.get("rsi", [])
            if len(rsi) > 0 and rsi[-1] < 50:
                return {"valid": False, "reason": "RSI below 50"}
            
            # MACD agreement
            macd = indicators.get("macd", [])
            if len(macd) >= 2 and macd[-1] <= macd[-2]:
                return {"valid": False, "reason": "MACD not confirming"}
            
            # Calculate entry, TP, and SL
            entry_price = current_candle["close"]
            stop_loss = entry_price - (config["sl_pips"] * pip_value)
            tp1 = entry_price + (config["tp1_pips"] * pip_value)
            tp2 = entry_price + (config["tp2_pips"] * pip_value)
            
            return {
                "valid": True,
                "direction": "BUY",
                "entry_price": entry_price,
                "take_profit": tp1,
                "stop_loss": stop_loss,
                "tp2": tp2,
                "conditions": [
                    "EMA 50 slope bullish",
                    "Price in 38.2-61.8 Fibonacci zone",
                    "Inside bar pattern",
                    "RSI above 50",
                    "MACD agreement"
                ]
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"Intraday setup error: {str(e)}"}

    def _check_swing_setup(self, candles: pd.DataFrame, indicators: dict, current_price: float, pip_value: float, config: dict) -> dict:
        """Check for Swing setup conditions."""
        try:
            # Major breakout and retest
            ema_200 = indicators.get("ema_200", [])
            if len(ema_200) < 50:
                return {"valid": False, "reason": "Insufficient EMA 200 data"}
            
            # Check if price is near EMA 200 or trendline
            ema_200_current = ema_200[-1]
            price_distance = abs(current_price - ema_200_current) / ema_200_current
            
            if price_distance > 0.02:  # 2% tolerance
                return {"valid": False, "reason": "Price not near major support/resistance"}
            
            # Reversal candle pattern
            current_candle = candles.iloc[-1]
            prev_candle = candles.iloc[-2]
            
            # Check for hammer or bullish engulfing
            body_size = abs(current_candle["close"] - current_candle["open"])
            lower_shadow = min(current_candle["open"], current_candle["close"]) - current_candle["low"]
            upper_shadow = current_candle["high"] - max(current_candle["open"], current_candle["close"])
            
            is_hammer = (lower_shadow > 2 * body_size) and (upper_shadow < body_size)
            is_bullish_engulfing = (current_candle["close"] > current_candle["open"] and 
                                  prev_candle["close"] < prev_candle["open"] and
                                  current_candle["close"] > prev_candle["open"] and
                                  current_candle["open"] < prev_candle["close"])
            
            if not (is_hammer or is_bullish_engulfing):
                return {"valid": False, "reason": "No reversal candle pattern"}
            
            # RSI divergence check
            rsi = indicators.get("rsi", [])
            if len(rsi) >= 14:
                # Simple divergence check
                price_lower_low = candles["low"].iloc[-1] < candles["low"].iloc[-5]
                rsi_higher_low = rsi[-1] > rsi[-5]
                
                if not (price_lower_low and rsi_higher_low):
                    return {"valid": False, "reason": "No RSI bullish divergence"}
            
            # Calculate entry, TP, and SL
            entry_price = current_candle["close"]
            stop_loss = entry_price - (config["sl_pips"] * pip_value)
            tp1 = entry_price + (config["tp1_pips"] * pip_value)
            tp2 = entry_price + (config["tp2_pips"] * pip_value)
            
            return {
                "valid": True,
                "direction": "BUY",
                "entry_price": entry_price,
                "take_profit": tp1,
                "stop_loss": stop_loss,
                "tp2": tp2,
                "conditions": [
                    "Price near EMA 200 support",
                    "Reversal candle pattern (Hammer/Engulfing)",
                    "RSI bullish divergence",
                    "Major support level test"
                ]
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"Swing setup error: {str(e)}"}

    def _get_pip_value(self, price: float) -> float:
        """Calculate pip value based on price."""
        # For JPY pairs (price > 100), 1 pip = 0.01
        # For other pairs, 1 pip = 0.0001
        if price > 100 or any(jpy in pair.upper() for jpy in ['JPY', 'USD/JPY', 'EUR/JPY', 'GBP/JPY']):
            return 0.01
        else:
            return 0.0001

    async def get_user_signals(
        self,
        user_id: int,
        status: Optional[SignalStatus] = None
    ) -> List[SignalModel]:
        """Get signals for a specific user."""
        query = select(SignalModel).where(SignalModel.user_id == user_id)
        if status:
            query = query.where(SignalModel.status == status)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_signal_status(
        self,
        signal_id: int,
        status: SignalStatus
    ) -> SignalModel:
        """Update signal status."""
        query = select(SignalModel).where(SignalModel.id == signal_id)
        result = await self.db.execute(query)
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        signal.status = status
        await self.db.commit()
        await self.db.refresh(signal)
        
        # Remove from active signals if closed
        if status in [SignalStatus.CLOSED, SignalStatus.CANCELLED]:
            if signal.pair in self._active_signals:
                self._active_signals[signal.pair] = [
                    s for s in self._active_signals[signal.pair] if s.id != signal_id
                ]
        
        # Broadcast update
        await websocket_manager.broadcast_signal_status(signal)
        
        return signal

async def get_signal_generation_service(db: Session) -> SignalGenerationService:
    """Get a signal generation service instance."""
    return SignalGenerationService(db)

def candles_to_df(candles: List[Dict]) -> pd.DataFrame:
    """Convert candle data to DataFrame, handling both 'datetime' and 'timestamp' columns."""
    df = pd.DataFrame(candles)
    # Patch: If 'timestamp' is missing but 'datetime' exists, use 'datetime' as 'timestamp'
    if 'timestamp' not in df.columns and 'datetime' in df.columns:
        df['timestamp'] = df['datetime']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def signal_dict_to_model(signal_dict: Dict[str, Any]) -> SignalModel:
    """Convert a dictionary to a Signal model."""
    try:
        # Convert string values to enums where needed
        if isinstance(signal_dict.get('signal_type'), str):
            signal_dict['signal_type'] = SignalType[signal_dict['signal_type']]
        if isinstance(signal_dict.get('direction'), str):
            signal_dict['direction'] = SignalDirection[signal_dict['direction']]
        if isinstance(signal_dict.get('status'), str):
            signal_dict['status'] = SignalStatus[signal_dict['status']]
        if isinstance(signal_dict.get('timeframe'), str):
            signal_dict['timeframe'] = SignalTimeframe[signal_dict['timeframe']]
        
        # Convert setup conditions
        if 'setup_conditions' in signal_dict and isinstance(signal_dict['setup_conditions'], list):
            signal_dict['setup_conditions'] = [
                SignalSetup[setup] if isinstance(setup, str) else setup
                for setup in signal_dict['setup_conditions']
            ]
        
        # Convert timestamps
        for field in ['created_at', 'updated_at', 'expires_at']:
            if field in signal_dict and isinstance(signal_dict[field], str):
                signal_dict[field] = datetime.fromisoformat(signal_dict[field])
        
        # Create Signal model
        return SignalModel(**signal_dict)
    except Exception as e:
        logger.error(f"Error converting signal dict to model: {str(e)}")
        raise ValueError(f"Invalid signal data: {str(e)}")

def calculate_signal_confidence(
    signal_data: Dict[str, Any],
    market_conditions: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate the confidence score for a trading signal.
    
    Args:
        signal_data: Dictionary containing signal information
        market_conditions: Optional dictionary containing market conditions
        
    Returns:
        float: Confidence score between 0 and 1
    """
    try:
        confidence_factors = []
        
        # 1. Trend Strength (0-1)
        if 'trend_strength' in signal_data:
            trend_strength = min(max(signal_data['trend_strength'], 0), 1)
            confidence_factors.append(trend_strength * 0.3)  # 30% weight
        
        # 2. Support/Resistance (0-1)
        if 'sr_levels' in signal_data:
            sr_count = len(signal_data['sr_levels'])
            sr_score = min(sr_count / 5, 1)  # Normalize to 0-1, max 5 levels
            confidence_factors.append(sr_score * 0.2)  # 20% weight
        
        # 3. Risk/Reward Ratio (0-1)
        if 'risk_reward_ratio' in signal_data:
            rr_ratio = signal_data['risk_reward_ratio']
            rr_score = min(rr_ratio / 3, 1)  # Normalize to 0-1, max 3:1 ratio
            confidence_factors.append(rr_score * 0.2)  # 20% weight
        
        # 4. Market Conditions (0-1)
        if market_conditions:
            market_score = 0
            if 'volatility' in market_conditions:
                # Lower volatility is better
                volatility = market_conditions['volatility']
                market_score += (1 - min(volatility / 0.02, 1)) * 0.5
            
            if 'volume' in market_conditions:
                # Higher volume is better
                volume = market_conditions['volume']
                market_score += min(volume / 1000000, 1) * 0.5
            
            confidence_factors.append(market_score * 0.3)  # 30% weight
        
        # Calculate final confidence score
        if confidence_factors:
            confidence = sum(confidence_factors)
            return min(max(confidence, 0), 1)  # Ensure between 0 and 1
        
        return 0.5  # Default confidence if no factors available
        
    except Exception as e:
        logger.error(f"Error calculating signal confidence: {str(e)}")
        return 0.5  # Return neutral confidence on error

async def check_memory_lock(db: Session, pair: str, signal_type: SignalType) -> bool:
    """Check if there's a memory lock preventing new signals for this pair."""
    # Check for active signals with memory lock
    result = await db.execute(
        select(SignalModel).where(
            SignalModel.pair == pair,
            SignalModel.type == signal_type,
            SignalModel.status.in_([SignalStatus.ACTIVE, SignalStatus.TP1_HIT])
        )
    )
    active_signals = result.scalars().all()
    
    if active_signals:
        # If there are multiple active signals, we need to clean them up
        if len(active_signals) > 1:
            logger.warning(f"Multiple active signals found for {pair} {signal_type.value}: {len(active_signals)} signals")
            # Keep only the most recent one, close the others
            sorted_signals = sorted(active_signals, key=lambda x: x.created_at, reverse=True)
            for signal in sorted_signals[1:]:
                signal.status = SignalStatus.CANCELLED
                signal.label = "Closed due to duplicate signals"
                logger.info(f"Closing duplicate signal {signal.id} for {pair}")
            await db.commit()
        
        logger.info(f"Memory lock active for {pair} {signal_type.value}: Signal ID {active_signals[0].id}")
        return True
    
    return False

async def create_memory_lock(db: Session, pair: str, signal_type: SignalType, signal_id: int) -> str:
    """Create a memory lock for a signal to prevent repainting."""
    lock_id = f"{pair}_{signal_type.value}_{signal_id}_{datetime.utcnow().timestamp()}"
    
    # Update the signal with memory lock
    result = await db.execute(
        select(SignalModel).where(SignalModel.id == signal_id)
    )
    signal = result.scalar_one_or_none()
    
    if signal:
        signal.memory_lock = lock_id
        await db.commit()
        logger.info(f"Created memory lock {lock_id} for signal {signal_id}")
    
    return lock_id

async def release_memory_lock(db: Session, pair: str, signal_type: SignalType) -> bool:
    """Release memory lock when signal is completed (TP1 hit or SL hit)."""
    result = await db.execute(
        select(SignalModel).where(
            SignalModel.pair == pair,
            SignalModel.type == signal_type,
            SignalModel.status.in_([SignalStatus.ACTIVE, SignalStatus.TP1_HIT])
        )
    )
    signal = result.scalar_one_or_none()
    
    if signal:
        signal.memory_lock = None
        await db.commit()
        logger.info(f"Released memory lock for {pair} {signal_type.value}")
        return True
    
    return False