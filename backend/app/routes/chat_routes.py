from fastapi import APIRouter, HTTPException
from app.services.chat_service import process_chat_message
from app.schemas.chat_schema import (
    ChatSessionCreateResponse,
    ChatMessageRequest,
)
from app.store.chat_store import (
    create_session,
    session_exists,
    get_messages,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/session", response_model=ChatSessionCreateResponse)
def create_chat_session():
    session_id = create_session()
    return ChatSessionCreateResponse(session_id=session_id)


@router.post("/message")
def send_chat_message(payload: ChatMessageRequest):
    if not session_exists(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    result = process_chat_message(payload.session_id, payload.message)

    return {
        "session_id": result["session_id"],
        "mode": result["mode"],
        "routing_label": result["routing_label"],
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "sources": result["sources"],
        "messages": get_messages(payload.session_id),
    }

@router.get("/history/{session_id}")
def get_chat_history(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": get_messages(session_id)
    }