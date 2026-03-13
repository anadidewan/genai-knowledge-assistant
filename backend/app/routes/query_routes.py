from fastapi import APIRouter , HTTPException
from app.schemas.question_schema import QuestionRequest, QuestionResponse
from app.services.retrieval_service import hybrid_retrieve
from app.services.llm_service import generate_answer


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        retrieved_chunks = hybrid_retrieve(request.question, top_k=5)
        answer = generate_answer(request.question, retrieved_chunks)

        sources = [
            {
            "document_name" : chunk["document_name"],
            "chunk_id" : chunk["chunk_id"]
            }
            for chunk in retrieved_chunks[:3]
        ]
        return QuestionResponse(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            sources=sources
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))