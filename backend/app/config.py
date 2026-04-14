from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()



class Settings(BaseModel):
    app_name: str = "GenAI Knowledge Assistant API"
    version: str = "1.0.0"
    UPLOAD_DIR: str = Path("uploads")
    MODEL: str = "all-MiniLM-L6-v2"
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str | None = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    

    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_BASE_DELAY: float = float(os.getenv("LLM_BASE_DELAY", "1.0"))
    LLM_MAX_DELAY: float = float(os.getenv("LLM_MAX_DELAY", "30.0"))
    LLM_BACKOFF_FACTOR: float = float(os.getenv("LLM_BACKOFF_FACTOR", "2.0"))
 
    # Retrieval confidence threshold (0.0 - 1.0)
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.6")
    )
 


settings = Settings()
