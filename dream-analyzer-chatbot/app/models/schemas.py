from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

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
