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
import time
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/session", response_model=ChatSessionCreateResponse)
def create_chat_session():
    session_id = create_session()
    logger.info("Session created: %s", session_id)
    return ChatSessionCreateResponse(session_id=session_id)


@router.post("/message")
def send_chat_message(payload: ChatMessageRequest):
    if not session_exists(payload.session_id):
        logger.warning("Session not found: %s", payload.session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info("Message received | session=%s | length=%d | preview=%.80s", payload.session_id, len(payload.message), payload.message)
    start = time.time()
    result = process_chat_message(payload.session_id, payload.message)
    elapsed = round((time.time() - start) * 1000)
    logger.info("Response sent | session=%s | mode=%s | routing=%s | answer_len=%d | elapsed=%dms", result["session_id"], result["mode"], result["routing_label"], len(result["answer"]), elapsed)
    

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
    messages = get_messages(session_id)
    logger.info("History fetched | session=%s | message_count=%d", session_id, len(messages))

    return {
        "session_id": session_id,
        "messages": messages,
    }