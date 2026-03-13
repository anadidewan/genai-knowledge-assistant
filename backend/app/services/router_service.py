from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def should_use_retrieval(question: str) -> dict:
    if settings.plenty_available == 0:
        q = question.lower()

        retrieval_triggers = [
            "uploaded document",
            "uploaded documents",
            "uploaded pdf",
            "pdf",
            "paper",
            "report",
            "file",
            "document",
            "documents",
            "according to the document",
            "according to the paper",
            "from the document",
            "from the paper",
            "in the document",
            "in the paper",
            "summarize the document",
            "summarize the paper",
            "what does the document say",
            "what does the paper say",
        ]

        for phrase in retrieval_triggers:
            if phrase in q:
                return {
                    "decision": "retrieve",
                    "raw_label": f"matched rule: {phrase}"
                }

        return {
            "decision": "direct",
            "raw_label": "no retrieval trigger matched"
        }
    else:
        prompt = f"""
    You are a routing assistant for a RAG system.

    Your job is to decide whether a user question requires retrieving information from uploaded documents.

    Return ONLY one of these two labels:
    - RETRIEVE
    - DIRECT

    Use RETRIEVE if:
    - the user is asking about the uploaded document(s)
    - the answer likely depends on document-specific content
    - the user asks to summarize, extract, explain, compare, or locate information from uploaded files

    Use DIRECT if:
    - the question is general knowledge
    - the answer does not require any uploaded document context

    Question:
    {question}
    """

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt
        )

        label = response.text.strip().upper()

        if "RETRIEVE" in label:
            decision = "RETRIEVE"
        else:
            decision = "DIRECT"

        return {
            "decision": decision,
            "raw_label": label
        }
