from fastapi import FastAPI
from app.config import settings
from app.routes.document_routes import router as document_router
from app.routes.query_routes import router as query_router
from app.store.document_store import store

app = FastAPI(title=settings.app_name, version=settings.version)
@app.on_event("startup")
def startup_event():
    store.load_from_disk()

@app.get("/")
def read_root():
    return {"message": "Backend is running"}


app.include_router(document_router)
app.include_router(query_router)