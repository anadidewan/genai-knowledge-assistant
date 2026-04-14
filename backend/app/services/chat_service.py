from fastapi import HTTPException

from app.store.chat_store import (
    save_message,
    get_recent_messages,
)
from app.services.retrieval_service import hybrid_retrieve, graph_expand, get_graph_context
from app.services.llm_service import generate_answer, generate_direct_answer, generate_critique_answer, _call_gemini
from app.services.router_service import should_use_retrieval
from app.config import settings
import time
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

def rewrite_query(history: list[dict], current_message: str) -> str:
    if not history:
        return current_message

    recent = history[-4:]
    history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    prompt = f"""Rewrite the user's latest message as a standalone question 
that makes sense without conversation history. 
If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation:
{history_text}

Latest message: {current_message}

Standalone question:"""

    return _call_gemini(prompt, caller="rewrite_query").strip()

# def build_retrieval_query(history: list[dict], current_message: str) -> str:

#     previous_user_messages = [
#         msg["content"] for msg in history if msg["role"] == "user"
#     ]

#     if previous_user_messages:
#         last_user_message = previous_user_messages[-1]
#         return f"{last_user_message} {current_message}"

#     return current_message


def process_chat_message(session_id: str, user_message: str) -> dict:
    # Save current user message
    save_message(session_id, "user", user_message)

    # Load recent history
    history = get_recent_messages(session_id, limit=6)
    # Build retrieval query
    retrieval_query = rewrite_query(history, user_message)
    logger.debug("Retrieval query built | session=%s | query=%.120s", session_id, retrieval_query)

    


    try:
        routing = should_use_retrieval(user_message)
        logger.info("Routing decision | session=%s | decision=%s | label=%s", session_id, routing["decision"], routing["raw_label"])


        if routing["decision"] in ("retrieve", "critique"):
            retrieved_chunks = hybrid_retrieve(retrieval_query, top_k=5)
            retrieved_chunks = graph_expand(retrieved_chunks, top_k=3)
            graph_context = get_graph_context(retrieved_chunks)
            confidence = (
                retrieved_chunks[0].get("retrieval_confidence", 0.0)
                if retrieved_chunks
                else 0.0
            )
            if confidence < settings.RETRIEVAL_CONFIDENCE_THRESHOLD:
                answer = generate_direct_answer(user_message, history)
                mode = "direct_low_confidence"
                sources = []
                retrieved_chunks = []
            elif routing["decision"] == "critique":
                answer = generate_critique_answer(user_message, retrieved_chunks, history, graph_context)

                mode = "retrieved"
                sources = [
                    {
                        "document_name": chunk["document_name"],
                        "chunk_id": chunk["chunk_id"],
                    }
                    for chunk in retrieved_chunks
                ]
            else:
                answer = generate_answer(user_message, retrieved_chunks, history, graph_context)

                mode = "retrieved"
                sources = [
                    {
                        "document_name": chunk["document_name"],
                        "chunk_id": chunk["chunk_id"],
                    }
                    for chunk in retrieved_chunks
                ]
        else:
            retrieved_chunks = []
            answer = generate_direct_answer(user_message, history)
            sources = []
            mode = "direct"
 

        # Save assistant reply
        save_message(session_id, "assistant", answer)

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "mode": mode,
            "routing_label": routing["raw_label"],
        }
    except ValueError as e:
        logger.error("Chat processing ValueError | session=%s | error=%s", session_id, e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Chat processing failed | session=%s | error=%s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Save assistant reply
