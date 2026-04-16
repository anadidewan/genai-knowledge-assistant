# GenAI Knowledge Assistant – Backend

A modular FastAPI-based backend implementing a multi-document Retrieval-Augmented Generation (RAG) system with hybrid retrieval, knowledge graph enrichment, and LLM-based answer generation.

---

## Overview

This backend enables:

* PDF ingestion and indexing
* Semantic + keyword hybrid retrieval
* Knowledge graph extraction using NLP
* Multi-turn chat with context-aware query rewriting
* Confidence-aware answer generation
* Persistent storage of vectors, documents, and chat sessions

---

## Architecture

### High-Level Flow

```
User Query
   ↓
Chat Service
   ↓
Query Rewriting + Routing
   ↓
Retrieval Layer (Hybrid)
   ↓
Graph Expansion + Context Building
   ↓
LLM Generation (Gemini)
   ↓
Response (Answer + Sources)
```

---

## Project Structure

```
app/
│
├── routes/
│   ├── document_routes.py     # Upload + graph endpoints
│   └── chat_routes.py         # Chat APIs
│
├── services/
│   ├── chat_service.py        # Main orchestration logic
│   ├── retrieval_service.py   # Hybrid retrieval + scoring
│   ├── llm_service.py         # Gemini integration
│   ├── router_service.py      # Query rewriting + routing
│   ├── document_service.py    # Ingestion pipeline
│   └── graph_service.py       # Knowledge graph extraction
│
├── store/
│   ├── document_store.py      # FAISS + chunks persistence
│   └── chat_store.py          # Chat session persistence
│
├── utils/
│   ├── vector_utils.py        # Embeddings + FAISS
│   ├── text_utils.py          # Chunking
│   ├── pdf_utils.py           # PDF parsing
│   ├── retry_utils.py         # LLM retry logic
│   └── custom_logger.py       # Logging setup
│
├── schemas/
│   └── chat_schema.py         # Request/response models
│
├── config.py                  # Environment + settings
└── main.py                    # FastAPI entry point
```

---

## Core Components

### 1. Document Ingestion

* Extracts text from PDF
* Splits into overlapping chunks
* Generates embeddings using Sentence Transformers
* Builds FAISS index
* Extracts entities + relationships using spaCy

---

### 2. Hybrid Retrieval

Combines:

* **Semantic search** (vector similarity)
* **Keyword matching** (token overlap)

Final score:

```
hybrid_score = 0.7 * semantic + 0.3 * keyword
```

Also computes a **retrieval confidence score** to decide fallback behavior.

---

### 3. Knowledge Graph Enrichment

* Extracts named entities from chunks
* Builds relationships using dependency parsing
* Expands retrieval results based on shared entities
* Injects entity relationships into LLM prompt

---

### 4. Query Rewriting + Routing

Uses LLM to:

* Convert conversational query → standalone query
* Classify into:

  * `RETRIEVE`
  * `CRITIQUE`
  * `DIRECT`

---

### 5. Answer Generation

Three modes:

* **Retrieved Answer**

  * Uses document context only
  * Enforces grounded responses

* **Critique Mode**

  * Provides feedback on document quality

* **Direct Answer**

  * General knowledge fallback

---

### 6. Chat System

* Session-based conversations
* Stores history in JSON
* Maintains last N messages for context

---

## Configuration

Environment variables:

```
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-3-flash-preview
RETRIEVAL_CONFIDENCE_THRESHOLD=0.6
```

---

## ▶️ Running the Backend

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API Endpoints

### Document APIs

* `POST /documents/upload`

  * Upload PDF
* `GET /documents/graph`

  * Inspect graph data

### Chat APIs

* `POST /chat/session`

  * Create new chat session

* `POST /chat/message`

  * Send message and get response

* `GET /chat/history/{session_id}`

  * Retrieve chat history

---

## Design Decisions

* Hybrid retrieval improves recall over pure vector search
* Retrieval confidence prevents hallucinated grounded answers
* Knowledge graph adds relational context beyond chunk-level similarity
* Modular services allow easy extension into agent-based systems

---

## Current Limitations

* Full FAISS index rebuild on each upload
* Graph extraction limited to subset of chunks
* Character-based chunking (not semantic)
* Fixed routing instead of dynamic tool selection

---

## Future Work

* Agentic architecture using tool-based reasoning
* Retrieval reranking (cross-encoder)
* Semantic chunking (sentence/section-aware)
* Incremental indexing
* Answer verification loop
* Evaluation metrics (grounding, recall@k)

---

## 👨‍💻 Author Notes

This system is intentionally built without heavy frameworks to deeply understand:

* Retrieval pipelines
* Ranking strategies
* LLM orchestration
* System design trade-offs in real-world GenAI systems

---

## 🏁 Summary

This backend is not just a chatbot — it is a retrieval + ranking system with an LLM on top, designed to prioritize grounded, explainable answers over raw generation.
