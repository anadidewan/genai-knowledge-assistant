from app.utils.file_utils import save_uploaded_file
from app.utils.pdf_utils import extract_text_from_pdf
from app.utils.text_utils import chunk_text
from app.utils.vector_utils import create_embeddings, build_faiss_index
from app.store.document_store import store

stored_chunks = []
stored_index = None


def process_uploaded_document(file):
    global stored_chunks, stored_index

    saved_path = save_uploaded_file(file)
    extracted_text = extract_text_from_pdf(saved_path)

    if not extracted_text.strip():
        raise ValueError("Could not extract text from this PDF")

    chunks = chunk_text(extracted_text)

    if not chunks:
        raise ValueError("No chunks could be created from this PDF")

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    
    store.stored_chunks = chunks
    store.stored_index = index

    return {
        "message": "File uploaded and indexed successfully",
        "num_chunks": len(chunks),
        "saved_path": saved_path,
    }