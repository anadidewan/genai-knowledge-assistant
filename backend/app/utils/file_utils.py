import shutil
from fastapi import UploadFile
from app.config import settings
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

settings.UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(file: UploadFile) -> str:
    file_path = settings.UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("File written | name=%s | path=%s", file.filename, file_path)

    return str(file_path)


