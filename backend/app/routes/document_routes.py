from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import process_uploaded_document
from app.store.document_store import store

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

@router.get("/graph")
def get_graph_data():
    return {
        "graph_records_count": len(store.graph_data),
        "graph_data": store.graph_data[:5]
    }