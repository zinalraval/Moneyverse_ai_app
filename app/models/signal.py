from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base_class import Base
import enum
import json

class SignalType(str, enum.Enum):
    SCALPING = "SCALPING"
    INTRADAY = "INTRADAY"
    SWING = "SWING"

class SignalDirection(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

class SignalStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    SL_HIT = "SL_HIT"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    WAITING_FOR_NEXT = "WAITING_FOR_NEXT"

class SignalTimeframe(str, enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M45 = "45m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H8 = "8h"
    D1 = "1day"
    W1 = "1week"
    MN1 = "1month"

class SignalSetup(str, enum.Enum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    MACD_CROSSOVER = "MACD_CROSSOVER"
    MACD_DIVERGENCE = "MACD_DIVERGENCE"
    RSI_OVERSOLD_OVERBOUGHT = "RSI_OVERSOLD_OVERBOUGHT"
    SR_LEVELS = "SR_LEVELS"
    MICRO_SR = "MICRO_SR"
    TREND_ALIGNMENT = "TREND_ALIGNMENT"
    NEWS_FILTER = "NEWS_FILTER"

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pair = Column(String, nullable=False)
    type = Column(Enum(SignalType), nullable=False)
    timeframe = Column(Enum(SignalTimeframe), nullable=False)
    direction = Column(Enum(SignalDirection), nullable=False)
    entry_price = Column(Float, nullable=False)
    tp1 = Column(Float, nullable=False)
    tp2 = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    status = Column(Enum(SignalStatus), default=SignalStatus.ACTIVE)
    confidence = Column(Float, default=0.5)
    reason = Column(String, default="")
    label = Column(String)  # For status messages like "Move SL to Breakeven"
    logic_note = Column(String, default="")
    trading_style = Column(String)
    setup_conditions = Column(JSON, default=list)  # Store as JSON
    is_news_filtered = Column(Boolean, default=False)  # To track if signal was filtered due to news
    trend_leg_id = Column(String)  # To track which trend leg this signal belongs to
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    exit_price = Column(Float)
    pnl = Column(Float)
    exit_time = Column(DateTime(timezone=True))
    generated_time = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    memory_lock = Column(String)  # To prevent repainting and track signal sequence

    # Relationship with User model
    user = relationship("User", back_populates="signals")

    def update_status(self, new_status: SignalStatus, exit_price: float = None):
        """Update signal status and exit price if provided."""
        self.status = new_status
        self.updated_at = func.now()
        self.last_updated = func.now()
        
        # Update label based on status
        if new_status == SignalStatus.TP1_HIT:
            self.label = "Move SL to Breakeven"
        elif new_status == SignalStatus.COMPLETED:
            self.label = "Trade Completed. Wait for Next Signal"
        elif new_status == SignalStatus.WAITING_FOR_NEXT:
            self.label = "Waiting for Next Signal"
        
        if exit_price is not None:
            self.exit_price = exit_price
            if self.exit_price and self.entry_price:
                self.pnl = (self.exit_price - self.entry_price) / self.entry_price * 100

    def to_dict(self) -> dict:
        """Convert signal to dictionary for frontend."""
        return {
            "id": self.id,
            "pair": self.pair,
            "type": self.type,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": self.entry_price,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "stop_loss": self.stop_loss,
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "label": self.label,
            "logic_note": self.logic_note,
            "trading_style": self.trading_style,
            "setup_conditions": self.setup_conditions,
            "is_news_filtered": self.is_news_filtered,
            "trend_leg_id": self.trend_leg_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "generated_time": self.generated_time.isoformat() if self.generated_time else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "memory_lock": self.memory_lock
        }
