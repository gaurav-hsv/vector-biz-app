import json, time, uuid
from typing import Optional, Dict, Any
import redis
from .config import settings

_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# keys: sess:{sid} -> JSON {messages:[{role,text,ts}]}

def get_session(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_id = str(uuid.uuid4())
    key = f"sess:{session_id}"
    if not _r.exists(key):
        session = {
            "session_id": session_id,
            "messages": []  # list of objects like {"role": "user"/"assistant", "text": "..."}
        }
        _r.setex(key, 1800, json.dumps(session))
    else:
        session = json.loads(_r.get(key))

    return session


def append_message(session_id: str, role: str, text: str) -> None:
    key = f"sess:{session_id}"
    if not _r.exists(key):
        raise ValueError(f"Session {session_id} does not exist.")
    
    message = {
        "role": role,
        "text": text,
        "ts": int(time.time())
    }
    
    session = json.loads(_r.get(key))
    session["messages"].append(message)
    _r.setex(key, 1800, json.dumps(session))  # reset expiration

