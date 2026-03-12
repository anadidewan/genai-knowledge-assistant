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


settings = Settings()
