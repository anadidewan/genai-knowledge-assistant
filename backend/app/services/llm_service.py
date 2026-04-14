from google import genai
from app.config import settings
from app.utils.retry_utils import retry_with_backoff
from typing import List

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

def format_history(history):
    if not history:
        return ""

    formatted = []
    for msg in history[-6:]:  # limit to last few
        role = msg["role"].capitalize()
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

 
 
@retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
def _call_gemini(prompt: str) -> str:
    """Centralized Gemini call with retry logic."""
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL, contents=prompt
    )
    return response.text

def generate_critique_answer(question: str, retrieved_chunks: list[dict], history: list[dict] = None) -> str:
    history_text = format_history(history)

    context_parts = []

    for item in retrieved_chunks[:3]:
        context_parts.append(
            f"Document: {item['document_name']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Text: {item['text']}"
        )

    context = "\n\n".join(context_parts)


    prompt = f"""
    You are a helpful AI assistant reviewing a document.

    Your task is to analyze the provided document excerpts and suggest how the document can be improved.

    Instructions:
    - Base your feedback on the provided document context.
    - You may make reasonable writing and structure suggestions based on that context.
    - Do not invent missing sections or claim the document says something it does not.
    - Give practical, specific suggestions.
    - If possible, mention clarity, structure, completeness, tone, and technical depth where relevant.

    Recent conversation:
    {history_text}

    Document context:
    {context}

    User request:
    {question}
    """
    
    return _call_gemini(prompt)

def generate_answer(question: str, retrieved_chunks: list[dict], history: list[dict] = None) -> str:

    history_text = format_history(history)
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded document."

    context_parts = []

    for item in retrieved_chunks[:3]:
        context_parts.append(
            f"Document: {item['document_name']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Text: {item['text']}"
        )

    context = "\n\n".join(context_parts)


    prompt = f"""
    You are a helpful AI assistant that answers questions using ONLY the provided document context.

    Rules:
    1. Use only the provided context.
    2. Do not make up information.
    3. If the answer is not present in the context say:
    "I could not find the answer in the uploaded document."
    Recent conversation:
    {history_text}
    Context:
    {context}

    Question:
    {question}
    """

    return _call_gemini(prompt)

def generate_direct_answer(question: str, history: List[dict] = None) -> str:
    history_text = format_history(history)
    prompt = f"""
    You are a helpful AI assistant.
    Use the provided context when possible.
    Answer the user's question clearly and accurately.
    If you are unsure, say so briefly instead of making things up.
    Recent conversation:
    {history_text}
    Question:
    {question}
    """

    return _call_gemini(prompt)

