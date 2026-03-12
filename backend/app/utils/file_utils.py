import shutil
from fastapi import UploadFile
from app.config import settings

settings.UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(file: UploadFile) -> str:
    file_path = settings.UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(file_path)


