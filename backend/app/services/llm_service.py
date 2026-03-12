from google import genai
from app.config import settings

print(settings.GOOGLE_API_KEY)
client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded document."

    context = "\n\n".join(
        [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

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