
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatSessionCreateResponse(BaseModel):
    session_id: str


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., description="Existing chat session ID")
    message: str = Field(..., min_length=1, description="User message")


class ChatMessageItem(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatMessageResponse(BaseModel):
    session_id: str
    mode: str
    routing_label: str
    answer: str
    retrieved_chunks: Optional[List[dict]] = []
    sources: Optional[List[dict]] = []
    messages: Optional[List[ChatMessageItem]] = []