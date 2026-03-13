from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    text: str
    document_name: str
    chunk_id: int

class SourceInfo(BaseModel):
    document_name: str
    chunk_id: int


class QuestionResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    mode: str
    routing_label: str
    answer: str
    sources: List[SourceInfo]