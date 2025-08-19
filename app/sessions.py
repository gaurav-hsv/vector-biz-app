import uuid
from typing import Dict, Any, Optional

# NOTE: prod me Redis replace karein
_store: Dict[str, Dict[str, Any]] = {}

def ensure_session(session_id: Optional[str]) -> str:
    sid = session_id or str(uuid.uuid4())
    _store.setdefault(sid, {"messages": []})
    return sid

def append_message(session_id: str, role: str, text: str) -> None:
    entry = {"role": role, "text": text}
    _store[session_id]["messages"].append(entry)
    # keep only last 6 msgs for convenience
    _store[session_id]["messages"] = _store[session_id]["messages"][-6:]

def get_tail(session_id: str, n: int = 6):
    return _store.get(session_id, {}).get("messages", [])[-n:]
