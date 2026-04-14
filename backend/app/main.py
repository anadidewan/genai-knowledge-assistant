from fastapi import FastAPI
from app.config import settings
from app.routes.document_routes import router as document_router
from app.store.document_store import store
from app.routes.chat_routes import router as chat_router

from app.utils.custom_logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)


app = FastAPI(title=settings.app_name, version=settings.version)
@app.on_event("startup")
def startup_event():
    logger.info("Starting %s v%s | model=%s | gemini=%s", settings.app_name, settings.version, settings.MODEL, settings.GEMINI_MODEL)
    try:
        store.load_from_disk()
        logger.info("Store loaded: %d chunks, index=%s", len(store.stored_chunks), "ready" if store.stored_index is not None else "empty")
    except Exception as e:
        logger.error("Store load failed, starting empty: %s", e, exc_info=True)

@app.get("/")
def read_root():
    return {"message": "Backend is running"}


app.include_router(document_router)
app.include_router(chat_router)