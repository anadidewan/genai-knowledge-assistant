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

def graph_expand(retrieved_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:

    if not store.graph_data:
        return retrieved_chunks

    # Collect chunk IDs we already have
    existing_keys = {
        (c["document_name"], c["chunk_id"]) for c in retrieved_chunks
    }

    # Step 1: Find all entities mentioned in retrieved chunks
    matched_entities = set()
    for chunk in retrieved_chunks:
        for record in store.graph_data:
            if record["document_name"] == chunk["document_name"] and record["chunk_id"] == chunk["chunk_id"]:
                matched_entities.update(e.lower() for e in record["entities"])

    if not matched_entities:
        logger.debug("Graph expand: no entities found in retrieved chunks")
        return retrieved_chunks

    # Step 2: Find other chunks that contain those entities
    candidate_chunks = []
    for record in store.graph_data:
        key = (record["document_name"], record["chunk_id"])
        if key in existing_keys:
            continue

        record_entities = {e.lower() for e in record["entities"]}
        overlap = matched_entities.intersection(record_entities)

        if overlap:
            candidate_chunks.append({
                "document_name": record["document_name"],
                "chunk_id": record["chunk_id"],
                "overlap_count": len(overlap),
                "matched_entities": list(overlap),
            })

    # Rank by how many shared entities
    candidate_chunks.sort(key=lambda x: x["overlap_count"], reverse=True)

    # Step 3: Look up the actual chunk text and append
    expanded = list(retrieved_chunks)
    for candidate in candidate_chunks[:top_k]:
        for stored_chunk in store.stored_chunks:
            if (stored_chunk["document_name"] == candidate["document_name"]
                    and stored_chunk["chunk_id"] == candidate["chunk_id"]):
                expanded.append({
                    "chunk_id": stored_chunk["chunk_id"],
                    "document_name": stored_chunk["document_name"],
                    "text": stored_chunk["text"],
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "hybrid_score": 0.0,
                    "retrieval_confidence": 0.0,
                    "graph_expanded": True,
                    "matched_entities": candidate["matched_entities"],
                })
                break

    logger.info(
        "Graph expand: %d entities matched, %d new chunks added",
        len(matched_entities),
        len(expanded) - len(retrieved_chunks),
    )
    return expanded

def get_graph_context(retrieved_chunks: List[Dict[str, Any]], max_triplets: int = 15) -> str:
    """
    Build a text block of entity-relationship triplets relevant to
    the retrieved chunks, for injection into the LLM prompt.
    """
    if not store.graph_data:
        return ""

    chunk_keys = {
        (c["document_name"], c["chunk_id"]) for c in retrieved_chunks
    }

    triplets = []
    seen = set()

    for record in store.graph_data:
        if (record["document_name"], record["chunk_id"]) not in chunk_keys:
            continue

        for rel in record.get("relationships", []):
            triplet_key = (rel["source"].lower(), rel["relation"], rel["target"].lower())
            if triplet_key not in seen:
                seen.add(triplet_key)
                triplets.append(f'{rel["source"]} —[{rel["relation"]}]→ {rel["target"]}')

    if not triplets:
        return ""

    logger.debug("Graph context: %d triplets for prompt", len(triplets[:max_triplets]))
    return "\n".join(triplets[:max_triplets])

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