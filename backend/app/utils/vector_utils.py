import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

model = SentenceTransformer(settings.MODEL)


def create_embeddings(chunks: list[str]) -> np.ndarray:
    embeddings = model.encode(chunks)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def embed_query(query: str) -> np.ndarray:
    embedding = model.encode([query])
    return np.array(embedding, dtype="float32")


def search_index(index, query_embedding, k=3):
    distances, indices = index.search(query_embedding, k)
    return indices[0]