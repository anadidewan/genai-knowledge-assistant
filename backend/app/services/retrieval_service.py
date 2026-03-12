from app.utils.vector_utils import embed_query, search_index
from app.store.document_store import store




def retrieve_relevant_chunks(question: str):
    if store.stored_index is None:
        raise ValueError("No document uploaded yet")

    query_embedding = embed_query(question)
    top_indices = search_index(store.stored_index, query_embedding)
    results = [store.stored_chunks[i] for i in top_indices]

    return {
        "question": question,
        "relevant_chunks": results
    }