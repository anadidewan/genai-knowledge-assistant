import re
from typing import List, Dict, Any
from app.utils.vector_utils import embed_query
from app.store.document_store import store
from app.config import settings

import time
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)


def tokenize(text: str) -> set:
    text = text.lower()
    return set(re.findall(r"\b\w+\b", text))

def keyword_score(query: str, chunk_text: str) -> float:
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk_text)

    if not query_tokens or not chunk_tokens:
        return 0.0

    overlap = query_tokens.intersection(chunk_tokens)
    return len(overlap) / len(query_tokens)


def semantic_retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    logger.debug("Semantic retrieve | query=%.80s | top_k=%d", question, top_k)
    if store.stored_index is None or not store.stored_chunks:
        raise ValueError("No documents uploaded yet")

    query_embedding = embed_query(question)

    distances, indices = store.stored_index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(store.stored_chunks):
            continue

        chunk = store.stored_chunks[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "document_name": chunk["document_name"],
            "text": chunk["text"],
            "semantic_score": float(distances[0][rank]),
        })
    logger.debug("Semantic retrieve returned %d results", len(results))

    return results

def keyword_retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    logger.debug("Keyword retrieve | query=%.80s | top_k=%d", question, top_k)

    if not store.stored_chunks:
        raise ValueError("No documents uploaded yet")

    scored_chunks = []

    for chunk in store.stored_chunks:
        score = keyword_score(question, chunk["text"])

        if score > 0:
            scored_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "text": chunk["text"],
                "keyword_score": score,
            })

    scored_chunks.sort(key=lambda x: x["keyword_score"], reverse=True)
    logger.debug("Keyword retrieve: %d chunks with score > 0", len(scored_chunks))
    return scored_chunks[:top_k]


def normalize_semantic_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return results

    scores = [item["semantic_score"] for item in results]
    max_score = max(scores)
    min_score = min(scores)

    for item in results:
        if max_score == min_score:
            item["semantic_score_normalized"] = 1.0
        else:
            item["semantic_score_normalized"] = (
                (item["semantic_score"] - min_score) / (max_score - min_score)
            )

    return results

def compute_retrieval_confidence(results: List[Dict[str, Any]]) -> float:
    
    if not results:
        return 0.0
 
    scores = [item["hybrid_score"] for item in results]
    top_score = scores[0]
 
    if len(scores) == 1:
        return top_score
 
    rest_avg = sum(scores[1:]) / len(scores[1:])
    gap = top_score - rest_avg
 
    confidence = 0.6 * top_score + 0.4 * gap
    return round(min(max(confidence, 0.0), 1.0), 4)


def hybrid_retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    semantic_results = semantic_retrieve(question, top_k=top_k * 2)
    semantic_results = normalize_semantic_scores(semantic_results)

    keyword_results = keyword_retrieve(question, top_k=top_k * 2)

    merged = {}

    for item in semantic_results:
        key = (item["document_name"], item["chunk_id"])
        merged[key] = {
            "chunk_id": item["chunk_id"],
            "document_name": item["document_name"],
            "text": item["text"],
            "semantic_score": item.get("semantic_score_normalized", 0.0),
            "keyword_score": 0.0,
        }

    for item in keyword_results:
        key = (item["document_name"], item["chunk_id"])
        if key not in merged:
            merged[key] = {
                "chunk_id": item["chunk_id"],
                "document_name": item["document_name"],
                "text": item["text"],
                "semantic_score": 0.0,
                "keyword_score": item["keyword_score"],
            }
        else:
            merged[key]["keyword_score"] = item["keyword_score"]

    final_results = []
    for item in merged.values():
        item["hybrid_score"] = 0.7 * item["semantic_score"] + 0.3 * item["keyword_score"]
        final_results.append(item)

    final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    confidence = compute_retrieval_confidence(final_results[:top_k])


    print("\nTop hybrid results:")
    for item in final_results[:5]:
        print(
            item["chunk_id"],
            "semantic=", item["semantic_score"],
            "keyword=", item["keyword_score"],
            "hybrid=", item["hybrid_score"]
        )
    top_results = final_results[:top_k]
    for item in top_results:
        item["retrieval_confidence"] = confidence

    return top_results