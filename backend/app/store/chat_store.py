import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

# Path: backend/data/chat_history.json
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_storage_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CHAT_HISTORY_FILE.exists():
        initial_data = {
            "sessions": {}
        }
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)


def _load_data() -> dict[str, Any]:
    _ensure_storage_file()

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_data(data: dict[str, Any]) -> None:
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_session() -> str:
    data = _load_data()

    session_id = str(uuid.uuid4())
    logger.debug("Chat session created: %s", session_id)
    data["sessions"][session_id] = {
        "created_at": _utc_now_iso(),
        "messages": []
    }

    _save_data(data)

    return session_id


def session_exists(session_id: str) -> bool:
    data = _load_data()
    return session_id in data.get("sessions", {})


def save_message(session_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be either 'user' or 'assistant'")

    data = _load_data()

    if session_id not in data["sessions"]:
        logger.warning("Save message failed — session not found: %s", session_id)
        raise ValueError(f"Session '{session_id}' does not exist")

    message = {
        "role": role,
        "content": content,
        "timestamp": _utc_now_iso()
    }

    data["sessions"][session_id]["messages"].append(message)
    logger.debug("Message saved | session=%s | role=%s | length=%d", session_id, role, len(content))

    _save_data(data)


def get_messages(session_id: str) -> list[dict[str, str]]:
    data = _load_data()

    if session_id not in data["sessions"]:
        raise ValueError(f"Session '{session_id}' does not exist")

    return data["sessions"][session_id]["messages"]


def get_recent_messages(session_id: str, limit: int = 6) -> list[dict[str, str]]:
    messages = get_messages(session_id)
    return messages[-limit:]


def delete_session(session_id: str) -> None:
    data = _load_data()

    

    if session_id not in data["sessions"]:
        raise ValueError(f"Session '{session_id}' does not exist")

    del data["sessions"][session_id]
    logger.info("Session deleted: %s", session_id)
    _save_data(data)