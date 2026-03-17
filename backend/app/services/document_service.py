from app.utils.file_utils import save_uploaded_file
from app.utils.pdf_utils import extract_text_from_pdf
from app.utils.text_utils import chunk_text
from app.utils.vector_utils import create_embeddings, build_faiss_index
from app.store.document_store import store
from app.services.graph_service import build_graph_data



def process_uploaded_document(file):

    saved_path = save_uploaded_file(file)
    extracted_text = extract_text_from_pdf(saved_path)

    if not extracted_text.strip():
        raise ValueError("Could not extract text from this PDF")

    chunks = chunk_text(extracted_text)

    if not chunks:
        raise ValueError("No chunks could be created from this PDF")


    chunk_records = []
    for i, chunk in enumerate(chunks):
        chunk_records.append({
            "text": chunk,
            "document_name": file.filename,
            "chunk_id": i
        })

    store.stored_chunks.extend(chunk_records)
    all_chunk_texts = [record["text"] for record in store.stored_chunks]
    embeddings = create_embeddings(all_chunk_texts)
    store.stored_index = build_faiss_index(embeddings)
    # new_graph_records = build_graph_data(chunk_records[:30])
    # store.graph_data.extend(new_graph_records)

    store.save_to_disk()

    
    

    return {
        "message": "File uploaded and indexed successfully",
        "uploaded_file": file.filename,
        "num_new_chunks": len(chunk_records),
        "total_chunks": len(store.stored_chunks),
        "saved_path": saved_path,
    }