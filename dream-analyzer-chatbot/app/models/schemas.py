from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from enum import Enum

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(max_length=10000)
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_history: List[ChatMessage] = Field(default_factory=list, max_length=100)

class UIEventType(str, Enum):
    dream_staged = "dream_staged"
    toast = "toast"
    navigate = "navigate"
    refresh = "refresh"
    badge_update = "badge_update"
    state_patch = "state_patch"

class UIEvent(BaseModel):
    type: UIEventType
    payload: Dict[str, Any] = {}
    key: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    timestamp: datetime
    conversation_id: Optional[str] = None
    ui_events: List[UIEvent] = []


class StreamChunk(BaseModel):
    content: str
    is_complete: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
