import json, time, uuid
from typing import Dict, Any, List, Optional
import redis
from .config import settings

_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# keys: sess:{sid} -> JSON {messages:[{role,text,ts}]}

def ensure_session(session_id: Optional[str]) -> str:
    sid = session_id or str(uuid.uuid4())
    key = f"sess:{sid}"
    _r.setnx(key, json.dumps({"messages": []}))
    _r.expire(key, 60 * 60 * 6)  # 6h TTL
    return sid

def append_message(session_id: str, role: str, text: str) -> None:
    key = f"sess:{session_id}"
    raw = _r.get(key)
    if not raw:
        ensure_session(session_id)
        raw = _r.get(key) or '{"messages": []}'
    obj = json.loads(raw)
    obj.setdefault("messages", []).append({"role": role, "text": text, "ts": int(time.time())})
    obj["messages"] = obj["messages"][-6:]  # keep tail
    _r.set(key, json.dumps(obj))
    _r.expire(key, 60 * 60 * 6)

def get_tail(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    key = f"sess:{session_id}"
    raw = _r.get(key)
    if not raw:
        return []
    obj = json.loads(raw)
    return obj.get("messages", [])[-n:]
