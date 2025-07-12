from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List, Union
from datetime import datetime
from app.models.signal import (
    SignalType, SignalDirection, SignalStatus,
    SignalTimeframe, SignalSetup
)
import json

class SignalBase(BaseModel):
    pair: str = Field(..., description="Trading pair (e.g., XAU/USD)")
    type: SignalType = Field(..., description="Signal type (SCALPING, INTRADAY, SWING)")
    direction: SignalDirection = Field(..., description="Signal direction (BUY, SELL)")
    entry_price: float = Field(..., alias="entry", description="Entry price")
    tp1: float = Field(..., description="First take profit level")
    tp2: float = Field(..., description="Second take profit level")
    stop_loss: float = Field(..., alias="sl", description="Stop loss level")
    status: SignalStatus = Field(default=SignalStatus.ACTIVE, description="Current signal status")
    confidence: float = Field(..., ge=0, le=100, description="Signal confidence percentage")
    reason: str = Field(..., description="Reason for signal generation")
    label: Optional[str] = Field(None, description="Current signal label/instruction")
    logic_note: Optional[str] = Field(None, description="Technical analysis notes")
    timeframe: SignalTimeframe
    trading_style: Optional[str] = None
    setup_conditions: List[Union[str, SignalSetup]] = Field(default_factory=list)
    is_news_filtered: bool = Field(default=False)
    trend_leg_id: Optional[str] = None

    @validator('setup_conditions', pre=True)
    def validate_setup_conditions(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    @field_validator('timeframe', mode='before')
    @classmethod
    def normalize_timeframe(cls, v):
        if not v:
            return v
        v = str(v).upper()
        mapping = {
            "M1": "1M", "1M": "1M", "1MIN": "1M",
            "M5": "5M", "5M": "5M", "5MIN": "5M",
            "M15": "15M", "15M": "15M", "15MIN": "15M",
            "M30": "30M", "30M": "30M", "30MIN": "30M",
            "H1": "1H", "1H": "1H", "60M": "1H",
            "H4": "4H", "4H": "4H",
            "D1": "1D", "1D": "1D", "DAILY": "1D"
        }
        # Accept both 'H1' and '1H' as '1H', etc.
        return mapping.get(v, v)

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        from_attributes = True
        populate_by_name = True
        by_alias = False
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class SignalCreate(SignalBase):
    pass

class SignalUpdate(BaseModel):
    status: Optional[SignalStatus] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_time: Optional[datetime] = None
    label: Optional[str] = None
    stop_loss: Optional[float] = Field(None, alias="sl", validation_alias="sl")
    setup_conditions: Optional[List[Union[str, SignalSetup]]] = None
    is_news_filtered: Optional[bool] = None
    trend_leg_id: Optional[str] = None
    last_updated: Optional[datetime] = None

    @validator('setup_conditions', pre=True)
    def validate_setup_conditions(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        from_attributes = True
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class SignalRead(BaseModel):
    id: int
    user_id: int
    pair: str
    type: SignalType
    timeframe: SignalTimeframe
    direction: SignalDirection
    entry: float = Field(..., validation_alias="entry_price")
    tp1: float
    tp2: float
    stop_loss: float
    status: SignalStatus
    confidence: Optional[float] = None
    reason: Optional[str] = None
    label: Optional[str] = None
    logic_note: Optional[str] = None
    trading_style: Optional[str] = None
    setup_conditions: Optional[list] = None
    is_news_filtered: Optional[bool] = None
    trend_leg_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_time: Optional[datetime] = None
    generated_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    memory_lock: Optional[str] = None

    class Config:
        orm_mode = True
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

# For backward compatibility
SignalInDB = SignalRead
