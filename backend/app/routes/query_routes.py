from fastapi import APIRouter , HTTPException
from app.schemas.question_schema import QuestionRequest, QuestionResponse
from app.services.retrieval_service import hybrid_retrieve
from app.services.llm_service import generate_answer, generate_direct_answer
from app.services.router_service import should_use_retrieval


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(payload: QuestionRequest):
    try:
        routing = should_use_retrieval(payload.question)

        if routing["decision"] == "retrieve":
            retrieved_chunks = hybrid_retrieve(payload.question, top_k=5)
            answer = generate_answer(payload.question, retrieved_chunks)

            sources = [
                {
                    "document_name": chunk["document_name"],
                    "chunk_id": chunk["chunk_id"]
                }
                for chunk in retrieved_chunks
            ]

            return {
                "question": payload.question,
                "mode": "retrieved",
                "routing_label": routing["raw_label"],
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "sources": sources
            }

        else:
            answer = generate_direct_answer(payload.question)

            return {
                "question": payload.question,
                "mode": "direct",
                "routing_label": routing["raw_label"],
                "answer": answer,
                "retrieved_chunks": [],
                "sources": []
            }


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))