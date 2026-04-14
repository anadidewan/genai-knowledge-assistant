
import json
from app.services.llm_service import _call_gemini

from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

def rewrite_and_route(history: list[dict], current_message: str) -> dict:
    if not history:
        return {
            "rewritten_query": current_message,
            "mode": "direct",
        }

    recent = history[-4:]
    history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    prompt = f"""You are a routing assistant for a RAG system.

Given the conversation history and the user's latest message, do two things:

1. Rewrite the latest message as a standalone question (resolve pronouns, add context from history). If it's already standalone, keep it unchanged.

2. Classify the intent into one of these modes:
   - RETRIEVE: user is asking about uploaded documents (summarize, extract, explain, compare content)
   - CRITIQUE: user wants feedback on the document (improve, rewrite, make better, is this good)
   - DIRECT: general knowledge question, not about any uploaded document

Return ONLY valid JSON in this exact format, nothing else:
{{"query": "the rewritten standalone question", "mode": "retrieve"}}

Conversation:
{history_text}

Latest message: {current_message}"""

    raw = _call_gemini(prompt, caller="rewrite_and_route").strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    try:
        parsed = json.loads(raw.strip())
        mode = parsed.get("mode", "direct").lower()
        if mode not in ("retrieve", "critique", "direct"):
            mode = "direct"
        return {
            "rewritten_query": parsed.get("query", current_message),
            "mode": mode,
        }
    except Exception as e:
        logger.warning("Rewrite+route parse failed: %s | raw=%s", e, raw)
        return {
            "rewritten_query": current_message,
            "mode": "direct",
        }