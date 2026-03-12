from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import process_uploaded_document

router = APIRouter(prefix="/documents", tags=["documents"])


stored_chunks = []
stored_index = None

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    return process_uploaded_document(file)