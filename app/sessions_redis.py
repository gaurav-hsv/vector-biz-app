# app/sessions_redis.py
import json, time, uuid
from typing import Optional, Dict, Any, List
import redis
from .config import settings

_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Redis key & constants
_TTL_SECONDS = 1800  # 30 min sliding session
_KEY = lambda sid: f"sess:{sid}"
_MAX_MESSAGES = 200  # hard cap to prevent growth

# Schema:
# {
#   "session_id": str,
#   "messages": [{"role": "user"|"assistant"|"system", "text": str, "ts": int}],
#   "memory": {
#       "summary": str,                # 3–6 bullets plain text
#       "state": {                     # small, typed JSON
#           "last_topic": str|null,
#           "last_intent": str|null,
#           "last_calc": { ... }|null
#       }
#   }
# }

def _new_session(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "messages": [],
        "memory": {"summary": "", "state": {}},
    }

def get_session(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_id = str(uuid.uuid4())
    key = _KEY(session_id)
    raw = _r.get(key)
    if not raw:
        session = _new_session(session_id)
        _r.setex(key, _TTL_SECONDS, json.dumps(session))
        return session
    return json.loads(raw)

def _trim_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(msgs) > _MAX_MESSAGES:
        # keep the newest window
        return msgs[-_MAX_MESSAGES:]
    return msgs

def append_message(session_id: str, role: str, text: str) -> None:
    key = _KEY(session_id)
    for _ in range(4):  # small retry budget
        with _r.pipeline() as p:
            try:
                p.watch(key)
                raw = p.get(key)
                if not raw:
                    session = _new_session(session_id)
                else:
                    session = json.loads(raw)

                session["messages"].append({
                    "role": role,
                    "text": text,
                    "ts": int(time.time())
                })
                session["messages"] = _trim_messages(session["messages"])

                p.multi()
                p.setex(key, _TTL_SECONDS, json.dumps(session))
                p.execute()
                return
            except redis.WatchError:
                # concurrent writer; retry
                continue
    raise RuntimeError("append_message failed due to concurrent updates")

def get_memory(session_id: str) -> Dict[str, Any]:
    key = _KEY(session_id)
    raw = _r.get(key)
    if not raw:
        return {"summary": "", "state": {}}
    session = json.loads(raw)
    return session.get("memory") or {"summary": "", "state": {}}

def set_memory(session_id: str, *, summary: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> None:
    key = _KEY(session_id)
    for _ in range(4):
        with _r.pipeline() as p:
            try:
                p.watch(key)
                raw = p.get(key)
                if not raw:
                    session = _new_session(session_id)
                else:
                    session = json.loads(raw)

                mem = session.get("memory") or {"summary": "", "state": {}}
                if summary is not None:
                    # hard cap to keep prompts small
                    mem["summary"] = summary.strip()[:2000]
                if state is not None:
                    # shallow merge to avoid accidental schema blow-up
                    s = mem.get("state") or {}
                    s.update(state)
                    mem["state"] = s

                session["memory"] = mem
                p.multi()
                p.setex(key, _TTL_SECONDS, json.dumps(session))
                p.execute()
                return
            except redis.WatchError:
                continue
    raise RuntimeError("set_memory failed due to concurrent updates")
