from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class LicenseBase(BaseModel):
    code: str
    is_active: bool = True
    expires_at: Optional[datetime] = None
    expiration_date: Optional[datetime] = Field(None, alias="expires_at")

class LicenseCreate(LicenseBase):
    user_id: int

class LicenseUpdate(BaseModel):
    code: Optional[str] = None
    is_active: Optional[bool] = None
    expires_at: Optional[datetime] = None
    expiration_date: Optional[datetime] = Field(None, alias="expires_at")

class License(LicenseBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True 
        allow_population_by_field_name = True 