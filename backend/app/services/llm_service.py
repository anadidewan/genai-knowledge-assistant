from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
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

    Context:
    {context}

    Question:
    {question}
    """

    response = client.models.generate_content(
    model=settings.GEMINI_MODEL, contents=prompt
    )

    return response.text