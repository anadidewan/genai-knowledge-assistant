from fastapi import HTTPException

from app.store.chat_store import (
    save_message,
    get_recent_messages,
)
from app.services.retrieval_service import hybrid_retrieve
from app.services.llm_service import generate_answer, generate_direct_answer, generate_critique_answer
from app.services.router_service import should_use_retrieval



def build_retrieval_query(history: list[dict], current_message: str) -> str:

    previous_user_messages = [
        msg["content"] for msg in history if msg["role"] == "user"
    ]

    if previous_user_messages:
        last_user_message = previous_user_messages[-1]
        return f"{last_user_message} {current_message}"

    return current_message


def process_chat_message(session_id: str, user_message: str) -> dict:
    # Save current user message
    save_message(session_id, "user", user_message)

    # Load recent history
    history = get_recent_messages(session_id, limit=6)

    # Build retrieval query
    retrieval_query = build_retrieval_query(history[:-1], user_message)

    


    try:
        routing = should_use_retrieval(retrieval_query)


        if routing["decision"] == "retrieve":
            retrieved_chunks = hybrid_retrieve(retrieval_query, top_k=5)
            answer = generate_answer(user_message, retrieved_chunks, history)
            sources = [
                {
                    "document_name": chunk["document_name"],
                    "chunk_id": chunk["chunk_id"]
                }
                for chunk in retrieved_chunks
            ]

            mode = "retrieved"
        elif routing["decision"] == "critique":

            retrieved_chunks = hybrid_retrieve(retrieval_query, top_k=5)
            answer = generate_critique_answer(user_message, retrieved_chunks, history)
            sources = [
                {
                    "document_name": chunk["document_name"],
                    "chunk_id": chunk["chunk_id"]
                }
                for chunk in retrieved_chunks
            ]

            mode = "retrieved"
        else:
            retrieved_chunks = []
            answer = generate_direct_answer(user_message,history)
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
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save assistant reply
