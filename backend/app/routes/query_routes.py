from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.question_schema import QuestionRequest
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.llm_service import generate_answer


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = retrieve_relevant_chunks(request.question)
        answer = generate_answer(request.question, result)
        return {
            "question": request.question,
            "retrieved_chunks": result,
            "answer": answer
        }

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))
