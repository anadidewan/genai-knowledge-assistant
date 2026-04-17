import json
import re
from app.services.llm_service import _call_gemini
from app.store.document_store import store
from app.utils.custom_logger import get_logger

logger = get_logger(__name__)


DOC_SIGNAL_PATTERNS = [
    r"\bthis document\b",
    r"\bthis file\b",
    r"\buploaded document\b",
    r"\buploaded file\b",
    r"\battached document\b",
    r"\battached file\b",
    r"\bthe document\b",
    r"\bthe file\b",
    r"\bthis pdf\b",
    r"\bthe pdf\b",
    r"\buploaded pdf\b",
    r"\battached pdf\b",
    r"\bwhat is this document about\b",
    r"\bwhat does this document say\b",
    r"\bsummarize this document\b",
    r"\bsummarize this file\b",
    r"\bexplain this document\b",
    r"\bexplain this file\b",
]

CRITIQUE_SIGNAL_PATTERNS = [
    r"\bimprove\b",
    r"\brewrite\b",
    r"\bmake (it|this) better\b",
    r"\bis this good\b",
    r"\bfeedback\b",
    r"\bcritique\b",
    r"\breview this\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _has_uploaded_docs() -> bool:
    return bool(store.stored_chunks)


def _is_followup(message: str) -> bool:
    msg = message.lower().strip()

    followup_starts = (
        "summarize that",
        "is that good",
        
        
    )

    return msg.startswith(followup_starts)


def rewrite_and_route(history: list[dict], current_message: str) -> dict:
    message = current_message.lower().strip()
    has_docs = _has_uploaded_docs()

    # 1. If there are no uploaded docs, do not waste an LLM routing call.
    if not has_docs:
        logger.info("Routing decided by rule | mode=direct | reason=no_uploaded_docs")
        return {
            "rewritten_query": current_message,
            "mode": "direct",
        }

    # 2. Deterministic retrieve for strong document-reference signals.
    if _matches_any(message, DOC_SIGNAL_PATTERNS):
        logger.info(
            "Routing decided by rule | mode=retrieve | message=%.100s",
            current_message,
        )
        return {
            "rewritten_query": current_message,
            "mode": "retrieve",
        }

    # 3. Deterministic critique for strong critique signals when docs exist.
    if _matches_any(message, CRITIQUE_SIGNAL_PATTERNS):
        logger.info(
            "Routing decided by rule | mode=critique | message=%.100s",
            current_message,
        )
        return {
            "rewritten_query": current_message,
            "mode": "critique",
        }

    # 4. Follow-ups are ambiguous. Let the LLM decide with context.
    is_followup = _is_followup(current_message)
    if is_followup:
        logger.debug(
            "Routing marked ambiguous follow-up | message=%.100s",
            current_message,
        )

    # 5. LLM fallback for ambiguous cases only.
    recent = history[-4:] if history else []
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    )

    prompt = f"""You are a routing assistant for a RAG system.

Given the conversation history and the user's latest message, do two things:

1. Rewrite the latest message as a standalone question (resolve pronouns, add context from history). If it's already standalone, keep it unchanged.

2. Classify the intent into one of these modes:
   - RETRIEVE: user is asking about uploaded documents
   - CRITIQUE: user wants feedback on the uploaded document
   - DIRECT: general knowledge question, not about uploaded documents

Important routing rules:
- If the user clearly refers to an uploaded document, file, or PDF, choose RETRIEVE.
- If the user asks for feedback, rewriting, review, or improvement of uploaded content, choose CRITIQUE.
- Only choose DIRECT for clearly general questions not about uploaded documents.
- Follow-up questions may depend on conversation context, so use the history carefully.

Return ONLY valid JSON in this exact format:
{{"query": "the rewritten standalone question", "mode": "retrieve"}}

Conversation:
{history_text}

Latest message: {current_message}"""

    raw = _call_gemini(prompt, caller="rewrite_and_route").strip()
    logger.debug("Router raw response | raw=%.300s", raw)

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    try:
        parsed = json.loads(raw.strip())
        mode = parsed.get("mode", "direct").lower()

        if mode not in ("retrieve", "critique", "direct"):
            mode = "direct"

        rewritten_query = parsed.get("query", current_message)

        logger.info(
            "Routing decided by llm | mode=%s | rewritten_query=%.120s",
            mode,
            rewritten_query,
        )

        return {
            "rewritten_query": rewritten_query,
            "mode": mode,
        }

    except Exception as e:
        logger.warning("Rewrite+route parse failed: %s | raw=%s", e, raw)
        return {
            "rewritten_query": current_message,
            "mode": "retrieve",
        }