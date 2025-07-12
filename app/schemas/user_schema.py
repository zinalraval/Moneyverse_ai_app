from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    preferences: Optional[dict] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)
    is_active: Optional[bool] = None
    preferences: Optional[dict] = None


class UserOut(UserBase):
    id: int
    is_superuser: bool = False
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    preferences: Optional[dict] = None

    model_config = {
        "from_attributes": True  # replaces orm_mode for Pydantic v2
    }


# Alias for compatibility with previous code expecting `UserRead`
UserRead = UserOut

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)