from pypdf import PdfReader
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
        else:
            logger.warning("No text on page %d of %s (possibly scanned)", page + 1, file_path)
    logger.info("PDF extracted | file=%s | pages=%d | chars=%d", file_path, len(reader.pages), len("\n".join(text_parts)))

    return "\n".join(text_parts)