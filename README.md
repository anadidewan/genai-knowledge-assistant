# GenAI Knowledge Assistant

A full-stack AI-powered knowledge assistant that enables users to upload documents, ask questions, and receive grounded answers using a hybrid retrieval + LLM pipeline.

---

## Overview

This system allows users to:

* Upload PDF documents
* Ask questions in a chat interface
* Get answers grounded in document content
* Maintain multi-turn conversations
* View retrieved sources and supporting context

Under the hood, it combines:

* Vector search (FAISS)
* Keyword matching
* Knowledge graph enrichment
* LLM-based answer generation (Gemini)

---

## Project Structure

```
project-root/
│
├── frontend/          # React / UI layer (chat interface)
├── backend/           # FastAPI + RAG system
│
└── README.md          # You are here
```

---

## Tech Stack

### Frontend

* React (or your framework)
* API integration with backend

### Backend

* FastAPI
* FAISS (vector search)
* Sentence Transformers (embeddings)
* Gemini (LLM)
* spaCy (knowledge graph extraction)

---

## System Flow

1. User uploads a PDF
2. Backend:

   * extracts text
   * chunks content
   * generates embeddings
   * builds FAISS index
   * extracts entities + relationships
3. User asks a question
4. Backend:

   * rewrites query using chat history
   * routes request (retrieve / critique / direct)
   * retrieves relevant chunks
   * enriches with graph context
   * generates answer using LLM
5. Response returned with sources and context

---

## Running the Project

### Backend

```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```
cd frontend
npm install
npm start
```

---

## API Base URL

```
http://localhost:8000
```

---

## Key Features

* Multi-document RAG
* Hybrid retrieval (semantic + keyword)
* Knowledge graph augmentation
* Retrieval confidence-based fallback
* Chat session persistence
* Modular backend architecture

---

## Future Enhancements

* Agentic orchestration (tool-based reasoning)
* Retrieval reranking (cross-encoder)
* Knowledge graph visualization
* Streaming responses
* Evaluation metrics (Recall@k, grounding score)
* More tha pdf support

---

## 👨‍💻 Author

Built as part of advanced GenAI systems exploration focusing on retrieval, ranking, and LLM orchestration.
