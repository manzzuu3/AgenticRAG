from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 5

class Citation(BaseModel):
    source: str
    page: Optional[int]
    excerpt: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    assessment: str
    reasoning: str
    citations: List[Citation]

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Message]
