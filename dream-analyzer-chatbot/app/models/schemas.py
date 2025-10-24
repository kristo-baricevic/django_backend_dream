from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    message: str
    timestamp: datetime
    conversation_id: Optional[str] = None

class StreamChunk(BaseModel):
    content: str
    is_complete: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
