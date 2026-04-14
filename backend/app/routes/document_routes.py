from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import process_uploaded_document
from app.store.document_store import store
import time
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


stored_chunks = []
stored_index = None

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        logger.warning("Upload rejected: no filename provided")
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Upload rejected: unsupported file type '%s'", file.filename)

        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")
    logger.info("Upload started: %s", file.filename)
    start = time.time()
    try:
        result = process_uploaded_document(file)
        elapsed = round((time.time() - start) * 1000)
        logger.info("Upload complete | file=%s | new_chunks=%d | total_chunks=%d | elapsed=%dms", file.filename, result["num_new_chunks"], result["total_chunks"], elapsed)
        return result
    except Exception as e:
        logger.error("Upload failed | file=%s | error=%s", file.filename, e, exc_info=True)
        raise


@router.get("/graph")
def get_graph_data():
    return {
        "graph_records_count": len(store.graph_data),
        "graph_data": store.graph_data[:5]
    }